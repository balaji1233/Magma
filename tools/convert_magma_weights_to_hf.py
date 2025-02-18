# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LLaVa-NeXT (LLaVa-1.6) checkpoints from the original repository.

URL: https://github.com/haotian-liu/LLaVA/tree/main.


The command used to obtain original logits is the following:
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-mistral-7b" --image-file "images/llava_v1_5_radar.jpg" --query "What is shown in this image?" --max_new_tokens 100 --temperature 0

Note: logits are tested with torch==2.1.2.
"""

import os
import argparse
import glob
import json
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open
import open_clip
from llava.conversation import conv_templates, SeparatorStyle

from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from magma.configuration_magma import MagmaConfig
from magma.modeling_magma import MagmaForConditionalGeneration

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
)

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.model.model.backbone.": "vision_tower.",
    "model.mm_projector.": "multi_modal_projector.",
    "vision_tower.clip_model.visual.": "vision_tower.clip_vision_model.",
    "model.layers.": "language_model.model.layers.",
    "model.embed_tokens.": "language_model.model.embed_tokens.",
    "model.norm.": "language_model.model.norm.",
    "lm_head": "language_model.lm_head",
    # "model.model": "language_model.model",
    # "multi_modal_projector.0": "multi_modal_projector.linear_1",
    # "multi_modal_projector.2": "multi_modal_projector.linear_2",
}

KEYS_TO_REMOVE = [
    'sem_seg_head', 
    'clip_model.transformers', 
    'dilation_kernel', 
    'logit_scale', 
]


def load_original_state_dict(model_id):
    directory_path = model_id

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # if key does not contain any of the keys to remove
                    if not any([key_to_remove in key for key_to_remove in KEYS_TO_REMOVE]):
                        original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "./image_00002.png"
    image = Image.open(url)
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    filepath = os.path.join(model_id, "config.json")
    # read json
    with open(filepath) as f:
        model_config = json.load(f)
        model_config['vision_feature_layer'] = 'clip_vis_dense'
        print(model_config)

    text_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    vision_model_id = model_config["mm_vision_tower"]

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)

    # Llama-3 doesn't have a padding token set yet
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    text_config.pad_token_id = tokenizer.pad_token_id
    text_config.vocab_size = len(tokenizer)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    if "xxlarge" in vision_model_id:
        _, _, open_clip_image_processor = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg')
    elif "large" in vision_model_id:
        _, _, open_clip_image_processor = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large-laion2B-s34B-b82K-augreg')

    import pdb; pdb.set_trace()
    image_processor = MagmaImageProcessor(
        anyres_strategy=model_config['img_anyres_strategy'], 
        base_img_size=model_config['img_size'], 
        num_crops=model_config['max_num_crops'],
    )
    processor = MagmaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # prepare inputs: image
    image = load_image()
    inputs = processor(images=image, texts="this is a test", return_tensors="pt")

    config = MagmaConfig(
        vision_config=model_config,
        text_config=text_config.to_dict(),
        image_token_index=image_token_index,
        # image_grid_pinpoints=image_processor.image_grid_pinpoints,
        # use_image_newline_parameter=True,
    )

    with init_empty_weights():
        model = MagmaForConditionalGeneration(config)
    
    # load original state dict
    import pdb; pdb.set_trace()
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    if state_dict['language_model.model.embed_tokens.weight'].shape[0] != config.text_config.vocab_size:
        state_dict['language_model.model.embed_tokens.weight'] = torch.cat(
            [
                state_dict['language_model.model.embed_tokens.weight'],
                torch.zeros(
                    config.text_config.vocab_size - state_dict['language_model.model.embed_tokens.weight'].shape[0],
                    state_dict['language_model.model.embed_tokens.weight'].shape[1],
                ).type_as(state_dict['language_model.model.embed_tokens.weight']),
            ],
            dim=0,
        )
    if state_dict['language_model.lm_head.weight'].shape[0] != config.text_config.vocab_size:
        state_dict['language_model.lm_head.weight'] = torch.cat(
            [
                state_dict['language_model.lm_head.weight'],
                torch.zeros(
                    config.text_config.vocab_size - state_dict['language_model.lm_head.weight'].shape[0],
                    state_dict['language_model.lm_head.weight'].shape[1],
                ).type_as(state_dict['language_model.lm_head.weight']),
            ],
            dim=0,
        )
    # TODO: remove strict=False
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # Pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(vocab_size, pad_to_multiple_of=pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    device = "cuda:1"
    model.to(device)

    # prepare inputs: image
    image = load_image()
    # inputs = processor(images=image, text=prompt, return_tensors="pt")

    # prepare inputs: text
    magma_template = {
        # 'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\nFor you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",
        'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\n
                            For you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",        
        'prompt_type': "Now the task is:\n{}\n\nPlease localize the visual marks and predict their {} future positions with moving speed {}."
    }
    system_prompt_type = magma_template['system_prompt_type']
    prompt_type = magma_template['prompt_type']
    conv_mode = "llama_3_instruct"
    speed = 8
    num_marks = 9
    task_description = "push the chip bag to left."
    mark_ids = [i+1 for i in range(num_marks)]    
    prompt = system_prompt_type.format(mark_ids) + prompt_type.format(task_description, 16, speed)
    prompt = "<image>\n" + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = processor(images=image, texts=prompt, return_tensors="pt")
    inputs = inputs.to(device)

    # # verify single forward pass
    # print("Single forward pass")
    # with torch.inference_mode():
    #     outputs = model(**inputs)

    # verify generation
    output_ids = model.generate(
        **inputs,
        temperature=0.0,
        do_sample=False,
        num_beams=1,
        max_new_tokens=1000,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))
    print("Generated text is ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = model_id.split("/")[-1]
        model.push_to_hub(f"llava-hf/{repo_id}-hf")
        processor.push_to_hub(f"llava-hf/{repo_id}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="location of the model to convert",
        # default="/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/llava/checkpoints/finetune-llava-Meta-Llama-3-8B-Instruct-segtokv9-convnextxxlarge-none-bs1-as1-pt-ep1-bimsz768-ncrops4-anyresglobal-seqlen4096-1e-5-0.0_820k_openx-286k_bridge+36k_bau+18k_bfm+300k_fractal_v14",
        default="/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/llava/checkpoints/finetune-llava-Meta-Llama-3-8B-Instruct-segtokv9-convnextxxlarge-encoder-possinusoidal-tunesegdec-none-bs1-as1-pt-ep3-bimsz768-ncrops4-anyresglobal-seqlen4096-imsFalse-imlFalse-1e-5-0_820k_openx-286k_bridge_v14+36k_bau_v14+18k_bfm_v14", 
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="magma-llama-3-8b-instruct-bridge-bau-bfm-hf", type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
