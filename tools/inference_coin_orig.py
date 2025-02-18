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

import re
import logging
import ast
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob

def draw_visual_trace(video_chunk, pred_tracks, pred_visibility, save_dir="./", backward_tracking=False, grid_query_frame=0, file_name="visual_trace_salient"):
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, tracks_leave_trace=-1)
    # track_length = visual_trace_length(pred_tracks)
    # # only keep tracks with length > 0.2
    # if (track_length > 0.2).sum() > 0:
    #     pred_tracks = pred_tracks[:, :, track_length > 0.2]
    #     pred_visibility = pred_visibility[:, :, track_length > 0.2]

    vis.visualize(
        video_chunk,
        pred_tracks,
        pred_visibility,
        query_frame=0 if backward_tracking else grid_query_frame,
        filename=file_name,
    )     

def som_prompting(image, pos_traces, neg_traces):
    """
    draw marks on the image
    """
    image_size = image.size
    draw = ImageDraw.Draw(image)

    def get_text_size(text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def expand_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return [x1-4, y1-4, x2+4, y2+4]
    
    def draw_marks(draw, points, text_size, id, font_size):
        txt = str(id)
        draw.ellipse(((points[0]-max(text_size)//2-1, points[1]-max(text_size)//2-1, points[0]+max(text_size)//2+1, points[1]+max(text_size)//2+1)), fill='red')
        draw.text((points[0]-text_size[0] // 2, points[1]-text_size[1] // 2-3), txt, fill='white', font=font_size)
        
    fontsize = 1
    font = ImageFont.truetype("arial.ttf", fontsize)
    txt = "55"    
    while min(get_text_size(txt, image, font)) < 0.02*640: # image_size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("arial.ttf", fontsize)

    text_size_2digits = get_text_size('55', image, font)
    text_size_1digit = get_text_size('5', image, font)
    text_size = {
        1: text_size_1digit,
        2: text_size_2digits
    }

    # draw the starting point of positive traces on image
    num_pos = pos_traces.shape[2]
    pos_idx = torch.arange(num_pos)
    pos_traces_to_mark = pos_traces

    # random sample at most 10 negative traces
    num_neg = neg_traces.shape[2]
    num_sampled = int(10 * float(image.size[0]) / 640)
    if num_neg > num_sampled:
        neg_idx = torch.randperm(num_neg)[:num_sampled]
        neg_traces_to_mark = neg_traces[:,:,neg_idx]
    else:
        neg_idx = torch.arange(num_neg)
        neg_traces_to_mark = neg_traces

    num_traces_total = pos_traces_to_mark.shape[2] + neg_traces_to_mark.shape[2]
    # shuffle the indices
    all_idx = torch.randperm(num_traces_total)

    pos_mark_ids = []; neg_mark_ids = []
    pos_traces_som = {}
    for i in range(pos_traces_to_mark.shape[2]):
        pos = pos_traces_to_mark[:,:,i]
        mark_id = all_idx[i].item()
        text_size = get_text_size(str(mark_id+1), image, font)
        draw_marks(draw, pos[0][0], text_size, mark_id+1, font)
        pos_traces_som[mark_id+1] = pos
        pos_mark_ids.append(mark_id+1)

    for i in range(neg_traces_to_mark.shape[2]):
        neg = neg_traces_to_mark[:,:,i]
        mark_id = all_idx[pos_traces_to_mark.shape[2]+i].item()
        text_size = get_text_size(str(mark_id+1), image, font)
        draw_marks(draw, neg[0][0], text_size, mark_id+1, font)
        neg_mark_ids.append(mark_id+1)

    # save drawn image
    # image.save("test.png")
    return pos_traces_som, pos_mark_ids, neg_mark_ids

def som_prompting_with_priors(image, pos_traces_som, neg_traces_som, step_offset=0, draw_som_positive=True, draw_som_negative=True):
    """
    draw marks on the image
    """
    image_size = image.size
    draw = ImageDraw.Draw(image)

    def get_text_size(text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def expand_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return [x1-4, y1-4, x2+4, y2+4]
    
    def draw_marks(draw, points, text_size, id, font_size):
        txt = str(id)
        draw.ellipse(((points[0]-max(text_size)//2-1, points[1]-max(text_size)//2-1, points[0]+max(text_size)//2+1, points[1]+max(text_size)//2+1)), fill='red')
        draw.text((points[0]-text_size[0] // 2, points[1]-text_size[1] // 2-3), txt, fill='white', font=font_size)
        
    fontsize = 1
    font = ImageFont.truetype("/home/jianwyan/projects/ProjectWillow/agent_eval/arial.ttf", fontsize)
    txt = "55"    
    while min(get_text_size(txt, image, font)) < 0.02*image_size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("/home/jianwyan/projects/ProjectWillow/agent_eval/arial.ttf", fontsize)

    text_size_2digits = get_text_size('55', image, font)
    text_size_1digit = get_text_size('5', image, font)
    text_size = {
        1: text_size_1digit,
        2: text_size_2digits
    }

    for key, val in pos_traces_som.items():
        mark_id = key
        pos = val[:,step_offset if step_offset < val.shape[1] else -1]
        text_size = get_text_size(str(mark_id), image, font)
        if draw_som_positive:
            draw_marks(draw, pos[0], text_size, mark_id, font)
    
    for key, val in neg_traces_som.items():
        mark_id = key
        neg = val[:,step_offset if step_offset < val.shape[1] else -1]
        text_size = get_text_size(str(mark_id), image, font)
        if draw_som_negative:
            draw_marks(draw, neg[0], text_size, mark_id, font)


model_id = "magma_8b_model_coin_v7_hf"
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

device = "cuda"

# prepare inputs: image
image = Image.open("image_00003.png")
# inputs = processor(images=image, text=prompt, return_tensors="pt")

# prepare inputs: text
magma_template_openx = {
    # 'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\nFor you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",
    'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\n
                        For you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",        
    'prompt_type': "Now the task is:\n{}\n\nPlease localize the visual marks and predict their {} future positions with moving speed {}."
}

magma_template_coin = {
    # 'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\nFor you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",
    'system_prompt_type': """You are a robot using end-effector control. To accomplish a task, you need to do a planning on which objects to move and their future position in the image.\n
                        For you convinience, the image is split into 200x200 grids spatially, and the candidates are labeled with numeric marks {} on the image.\n""",        
    # 'prompt_type': "I am working on the task of \"{}\". Please briefly describe what you are seeing in the image and what I am doing.\n",
    # 'prompt_type': "I am working on the task of \"{}\". I have labeled a number of numeric marks with IDs {} on the image. Now please tell me which marks should be moved for current action.\n",
    'prompt_type': "I am working on the task of \"{}\". Now I am doing the step {}. I have labeled a number of numeric marks with IDs {} on the image and split the image into 200x200 grid. Now please tell me which marks should be moved and their positions in the next {} steps.\n"
    # 'prompt_type': "I am working on the task of \"{}\". Now I am doing the step {}. I have labeled a number of numeric marks with IDs {} on the image.\n"
}

system_prompt_type = magma_template_coin['system_prompt_type']
prompt_type = magma_template_coin['prompt_type']
conv_mode = "llama_3_instruct"
speed = 8
num_marks = 16
steps = 16
task_description = "Pick up the bowl."
mark_ids = [i+1 for i in range(num_marks)]    
# mark_ids = [1,15,16]
# # prompt = system_prompt_type.format(mark_ids) + prompt_type.format(task_description)
# prompt = prompt_type.format(task_description, mark_ids, steps-1)
# prompt = "<image>\n" + prompt


import os
import json
import cv2

# Data
# root_folder = "/home/jianwyan/projects/ProjectWillow/azureblobs/echelondata/datasets/epic_kitchens"
root_folder = "/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/COIN"
sft_data_list = os.path.join(root_folder, "sft_data_list.json")
sft_data_files = json.load(open(sft_data_list, 'r'))

# random shuffle sft_data_files
import random
# random.seed(123)
random.shuffle(sft_data_files)

for sft_data_file in sft_data_files:
    ann = torch.load(sft_data_file, map_location='cpu')
    if 'video' not in ann:
        continue
    vid_path = ann['video']
    vid_path_full = os.path.join(root_folder, vid_path)
    frame_start, frame_end = ann['frame_interval']
    pos_traces_to_mark, neg_traces_to_mark = ann['pos_traces_to_mark'], ann['neg_traces_to_mark']
    width, height = ann['image_size']
    step_to_predict = ann['step_to_predict']
    global_task = ann['global_instructions']
    if 'error' in ann['gpt_response'].lower():
        continue
    task_description = ann['gpt_response'].split("What you should do next:")[1]
    # remove all (string)
    # task_description = re.sub(r' \([^)]*\)', '', task_description)
    
    # random sample a frame
    frame_pos = random.randint(frame_start, frame_end-2)

    cap = cv2.VideoCapture(vid_path_full)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    # read frame
    ret, frame = cap.read()

    # convert to rgb and then pil image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((width, height))

    som_prompting_with_priors(image, pos_traces_to_mark, neg_traces_to_mark, step_offset=frame_pos-frame_start, draw_som_positive=True, draw_som_negative=True)
    image.save('temp.png')

    # extract traces from annotation
    pos_traces_gt = torch.stack([val[:,frame_pos-frame_start:] for val in pos_traces_to_mark.values()], dim=2)
    pos_traces_gt = pos_traces_gt.float() / 200 
    pos_traces_gt[:,:,:,0] = pos_traces_gt[:,:,:,0] * image.size[0]
    pos_traces_gt[:,:,:,1] = pos_traces_gt[:,:,:,1] * image.size[1]
    images = [image] * pos_traces_gt.shape[1]
    pos_visibility = pos_traces_gt.new(*pos_traces_gt.shape[:-1]).bool().fill_(True)
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    draw_visual_trace(video, pos_traces_gt.int(), pos_visibility, file_name='visual_trace_generated_v7_gt', save_dir="./videos")

    mark_ids = [key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()]    
    mark_ids = sorted(mark_ids)
    prompt = prompt_type.format(global_task, task_description, mark_ids, step_to_predict)
    print(prompt)
    prompt = '<image>' + '\n' + prompt

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
    import pdb; pdb.set_trace()
    # # extract traces from response
    import ast
    selected_marks_str, traces_str = response.split("are: ")
    # extract x, y coordinates in all [ ] in the string    
    traces_str = '{' + traces_str.strip().replace('\n', ',') + '}'                    
    traces_dict = ast.literal_eval(traces_str)
    overlay_traces = torch.cat([torch.tensor(ast.literal_eval(trace)).unsqueeze(1) for mark_id, trace in traces_dict.items()], dim=1).unsqueeze(0)
    overlay_traces = overlay_traces.float() / 200 
    overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
    overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
    images = [image] * overlay_traces.shape[1]
    overlay_visibility = overlay_traces.new(*overlay_traces.shape[:-1]).bool().fill_(True)
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    draw_visual_trace(video, overlay_traces, overlay_visibility, file_name="visual_trace_generated_v3", save_dir="./videos")
    import pdb; pdb.set_trace()

def draw_visual_trace(video_chunk, pred_tracks, pred_visibility, save_dir="./", backward_tracking=False, grid_query_frame=0, file_name="visual_trace_salient"):
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, tracks_leave_trace=-1)
    vis.visualize(
        video_chunk,
        pred_tracks,
        pred_visibility,
        query_frame=0 if backward_tracking else grid_query_frame,
        filename=file_name,
    )     

# # extract traces from response
# if "and their future positions are:" in response:
#     selected_marks_str, traces_str = response.split("and their future positions are:\n")
# else:
#     traces_str = response
#     selected_marks_str = None

# traces_dict = ast.literal_eval(traces_str)
# overlay_traces = []
# for mark_id, trace in traces_dict.items():
#     # convert list of tuples to tensor
#     trace = torch.tensor(trace).unsqueeze(1)
#     overlay_traces.append(trace)
# overlay_traces = torch.cat(overlay_traces, dim=1).unsqueeze(0)
# if selected_marks_str is not None:
#     selected_marks = re.findall(r'\[(.*?)\]', selected_marks_str)
#     selected_marks = [torch.tensor(ast.literal_eval(mark)).unsqueeze(0) for mark in selected_marks]
#     selected_marks = torch.cat(selected_marks, dim=0).unsqueeze(0)        
#     overlay_traces = torch.cat([selected_marks.unsqueeze(1), overlay_traces], dim=1)
# overlay_traces = overlay_traces.float() / 200
# overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
# overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
# images = [image] * overlay_traces.shape[1]
# overlay_visibility = overlay_traces.new(overlay_traces.shape[0], overlay_traces.shape[1], overlay_traces.shape[2]).fill_(True)
# video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
# draw_visual_trace(video, overlay_traces, overlay_visibility, file_name=task_description.replace(' ', '_'), save_dir="./videos")