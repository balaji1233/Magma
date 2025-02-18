from PIL import Image 
import requests 
import ast, re
import torch
import torchvision
from magma.utils.conversation import conv_templates, SeparatorStyle
from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForConditionalGeneration
from cotracker.utils.visualizer import Visualizer, read_video_from_path

model_id = "checkpoints/magma_8b_model_coin_v9_hf"
# model_id = "/home/jianwyan/projects/Magma/magma-llama-3-8b-instruct-bridge-bau-bfm-hf"
# model_id = "/home/jianwyan/projects/Magma/magma_8b_model_bugfixed_hf"
# model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma-hf/magma-llama-3-8b-instruct-hf"
# model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma-hf/magma-llama-3-8b-instruct-base-hf"

processor = MagmaProcessor.from_pretrained(model_id, trust_remote_code=True) 

model = MagmaForConditionalGeneration.from_pretrained(model_id, device_map="cuda", low_cpu_mem_usage=True) # use _attn_implementation='eager' to disable flash attention

# copy model.vision_tower.clip_vision_model.trunk.stages[0].blocks[0].weight to model.vision_tower.clip_vision_model.trunk.stages[0].blocks[0].gamma for all blocks
# for stage in model.vision_tower.clip_vision_model.trunk.stages:
#     for block in stage.blocks:
#         block.gamma = block.weight

device = "cuda:0"

# prepare inputs: image
image = Image.open("image_00002.png")
# inputs = processor(images=image, text=prompt, return_tensors="pt")

# prepare inputs: text
magma_template_openx = {
    # 'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\nFor you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",
    'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\n
                        For you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",        
    # 'prompt_type': "Now the task is:\n{}\n\nPlease localize the visual marks and predict their {} future positions with moving speed {}."
    'prompt_type': "Now the task is:\n{}\n\n. In the image with 200x200 grids, please localize the visual marks and predict their {} future positions."
}

system_prompt_type = magma_template_openx['system_prompt_type']
prompt_type = magma_template_openx['prompt_type']
conv_mode = "llama_3_instruct"
speed = 8
num_marks = 9
steps = 16
task_description = "I want to drink"
mark_ids = [i+1 for i in range(num_marks)]    
# mark_ids = [1,15,16]
# prompt = system_prompt_type.format(mark_ids) + prompt_type.format(task_description, steps, speed)
prompt = system_prompt_type.format(mark_ids) + prompt_type.format(task_description, steps)
prompt = "<image>\n" + prompt

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

inputs = processor(images=image, texts=prompt, return_tensors="pt")
inputs = inputs.to(device)

# verify generation
output_ids = model.generate(
    **inputs,
    temperature=0.0,
    do_sample=False,
    num_beams=1,
    max_new_tokens=5000,
    use_cache=True,
)

# only decode generated text
prompt_decoded = processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# remove prompt from generated text
response = generated_text[len(prompt_decoded):]

print("Generated text:", repr(response))

def draw_visual_trace(video_chunk, pred_tracks, pred_visibility, save_dir="./", backward_tracking=False, grid_query_frame=0, file_name="visual_trace_salient"):
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, tracks_leave_trace=-1)
    vis.visualize(
        video_chunk,
        pred_tracks,
        pred_visibility,
        query_frame=0 if backward_tracking else grid_query_frame,
        filename=file_name,
    )     

import pdb; pdb.set_trace()
# extract traces from response
if "and their future positions are:" in response:
    selected_marks_str, traces_str = response.split("and their future positions are:\n")
else:
    traces_str = response
    selected_marks_str = None

traces_dict = ast.literal_eval(traces_str)
overlay_traces = torch.cat([torch.tensor((trace)).unsqueeze(1) for mark_id, trace in traces_dict.items()], dim=1).unsqueeze(0)
overlay_traces = overlay_traces.float() / 200
overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
images = [image] * overlay_traces.shape[1]
overlay_visibility = overlay_traces.new(overlay_traces.shape[0], overlay_traces.shape[1], overlay_traces.shape[2]).fill_(True)
video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
draw_visual_trace(video, overlay_traces, overlay_visibility, file_name=task_description.replace(' ', '_'), save_dir="./videos")