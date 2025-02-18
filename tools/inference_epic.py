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
from PIL import Image, ImageDraw, ImageFont
from data_utils import *
import os
import json
import cv2
import re
import logging
import ast
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob

def draw_visual_trace(video_chunk, pred_tracks, pred_visibility, save_dir="./", backward_tracking=False, grid_query_frame=0, file_name="visual_trace_salient"):
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=2, tracks_leave_trace=-1)
    vis.visualize(
        video_chunk,
        pred_tracks,
        pred_visibility,
        query_frame=0 if backward_tracking else grid_query_frame,
        filename=file_name,
    )     

model_id = "checkpoints/magma_8b_model_epic_coin_v2_hf"
processor = MagmaProcessor.from_pretrained(model_id, trust_remote_code=True) 
model = MagmaForConditionalGeneration.from_pretrained(model_id, device_map="cuda", low_cpu_mem_usage=True) # use _attn_implementation='eager' to disable flash attention

device = "cuda"

prompt_type = 'You are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the historical moving traces for all marks on the image.\n'
# magma_template_coin['prompt_type']
conv_mode = "llama_3_instruct"
# Data
root_folder = "/home/jianwyan/projects/ProjectWillow/azureblobs/echelondata/datasets/epic_kitchens"
# root_folder = "/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/COIN"
sft_data_list = os.path.join(root_folder, "sft_data_list.json")
sft_data_files = json.load(open(sft_data_list, 'r'))

# random shuffle sft_data_files
import random
# random.seed(123)
random.shuffle(sft_data_files)
# conversation_constructor = Conversation(remove_static_trace_pts=True)
conversation_constructor = epic(
    mm_use_trace_start_end=False,
    mm_use_image_start_end=True,
    remove_static_trace_pts=True,
    dataset_folder=None,
    show_trace=True
)

for sft_data_file in sft_data_files:
    ann = torch.load(sft_data_file, map_location='cpu')
    if 'video' not in ann:
        continue
    video_path = ann['video']
    frame_start, frame_end = ann['frame_interval']
    video_path = os.path.join(root_folder, video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    visual_trace_path = os.path.join(root_folder, 'visual_trace' if 'epic' in root_folder else 'visual_traces', video_name, f'trace_{frame_start:09d}_{frame_end:09d}.pth')
    if os.path.exists(visual_trace_path):
        visual_traces = torch.load(visual_trace_path, map_location='cpu')
    else:
        visual_traces = None
    sources = conversation_constructor(item=ann, video_path=video_path, visual_traces=visual_traces)    
    image = sources['image']
    conversations = sources['conversations']
    if len(conversations) < 4:
        continue

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], conversations[0]['value'])
    conv.append_message(conv.roles[1], conversations[1]['value'])
    conv.append_message(conv.roles[0], conversations[2]['value'])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = processor(images=image, texts=prompt, return_tensors="pt")
    inputs = inputs.to(device)

    # verify generation
    output_ids = model.generate(
        **inputs,
        temperature=0.2,
        do_sample=False,
        num_beams=1,
        max_new_tokens=1000,
        use_cache=True,
    )

    # only decode generated text
    prompt_decoded = processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # remove prompt from generated text
    response = generated_text[len(prompt_decoded):]

    print("Generated text:", repr(response))
    # exit()    
    # # extract traces from response
    import ast
    traces_pos =  response.split("future movements are:")[0]
    # extract all [ ]s including [] in traces_pos
    traces_pos = re.findall(r'\[(.*?)\]', traces_pos)
    traces_pos = [f'[{trace}]' for trace in traces_pos]
    traces_pos = '[' + ','.join(traces_pos) + ']'
    traces_pos = ast.literal_eval(traces_pos)
    traces_pos = torch.tensor(traces_pos).unsqueeze(0).unsqueeze(0).float()
    traces_str = response.split("future movements are:")[-1]
    # else:
    #     traces_str = response.split("future positions are:")[-1]
    
    # extract x, y coordinates in all [ ] in the string    
    traces_str = '{' + traces_str.strip().replace('\n\n', ',') + '}'                    
    traces_dict = ast.literal_eval(traces_str)
    overlay_traces = [torch.tensor(ast.literal_eval('[' + trace[2:] if trace.startswith(']') else trace)) for mark_id, trace in traces_dict.items()]
    # pad to the same length
    max_len = max([trace.shape[0] for trace in overlay_traces])
    # pad to the same length with the last frame
    overlay_traces = [torch.cat([trace, trace[-1].unsqueeze(0).repeat(max_len - trace.shape[0], 1)], dim=0) for trace in overlay_traces]
    overlay_traces = torch.stack(overlay_traces, dim=1).unsqueeze(0)
    
    overlay_traces = overlay_traces.float() / 512 
    overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
    overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
    images = [image] * overlay_traces.shape[1]
    overlay_visibility = overlay_traces.new(*overlay_traces.shape[:-1]).bool().fill_(True)
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    draw_visual_trace(video, overlay_traces, overlay_visibility, file_name=model_id.split('/')[1], save_dir="./videos")
    import pdb; pdb.set_trace()