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

import os
import json
import cv2

def draw_visual_trace(video_chunk, pred_tracks, pred_visibility, save_dir="./", backward_tracking=False, grid_query_frame=0, file_name="visual_trace_salient"):
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, tracks_leave_trace=-1)
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


# model_id = "checkpoints/magma_8b_model_coin_v9_hf"
# model_id = "/home/jianwyan/projects/Magma/magma-llama-3-8b-instruct-bridge-bau-bfm-hf"
# model_id = "/home/jianwyan/projects/Magma/magma_8b_model_bugfixed_hf"
# model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma-hf/magma-llama-3-8b-instruct-hf"
# model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma-hf/magma-llama-3-8b-instruct-base-hf"
model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma/checkpoints/finetune-none-bs16-ep5-bimsz512-ncrops4-anyresglobal-seqlen2048-1e-5-constant-0.0_epic_coin_ego4d_howto100m_openx_-1_iseFalse_tseFalse_tsdTrue_rtptsTrue_qsz256/checkpoint-9000"

processor = MagmaProcessor.from_pretrained(model_id, trust_remote_code=True) 

model = MagmaForConditionalGeneration.from_pretrained(model_id, device_map="cuda", low_cpu_mem_usage=True) # use _attn_implementation='eager' to disable flash attention

# copy model.vision_tower.clip_vision_model.trunk.stages[0].blocks[0].weight to model.vision_tower.clip_vision_model.trunk.stages[0].blocks[0].gamma for all blocks
# for stage in model.vision_tower.clip_vision_model.trunk.stages:
#     for block in stage.blocks:
#         block.gamma = block.weight

device = "cuda"

spatial_quant_size = 256

conv_mode = "llama_3_instruct"

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
    task_description = re.sub(r' \([^)]*\)', '', task_description)
    
    # random sample a frame
    frame_pos = frame_start # random.randint(frame_start, frame_end-8)

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

    import pdb; pdb.set_trace()
    pos_traces_to_mark_filtered = {}
    for key, val in pos_traces_to_mark.items():
        # perform exponential moving smoothing for val
        trace = val[0]
        trace[:, 0] = spatial_quant_size * trace[:, 0].clamp_(0, width-1) / width
        trace[:, 1] = spatial_quant_size * trace[:, 1].clamp_(0, height-1) / height
        trace = trace.int()

        trace_temp = trace.float()
        # remove (almost) static points
        trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 2]], dim=0)
        # remove invisible points
        trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
        if trace_temp.size(0) <= step_to_predict // 2:
            continue
        # calulate motion speed
        speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean().item()
        if speed < 3.0:
            continue
        pos_traces_to_mark_filtered[key] = trace.unsqueeze(0)
    import pdb; pdb.set_trace()
    if len(pos_traces_to_mark_filtered) == 0:
        continue
    # extract traces from annotation
    pos_traces_gt = torch.stack([val[:,frame_pos-frame_start:] for val in pos_traces_to_mark_filtered.values()], dim=2)
    pos_traces_gt = pos_traces_gt.float() / spatial_quant_size 
    pos_traces_gt[:,:,:,0] = pos_traces_gt[:,:,:,0] * image.size[0]
    pos_traces_gt[:,:,:,1] = pos_traces_gt[:,:,:,1] * image.size[1]
    images = [image] * pos_traces_gt.shape[1]
    pos_visibility = pos_traces_gt.new(*pos_traces_gt.shape[:-1]).bool().fill_(True)
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    draw_visual_trace(video, pos_traces_gt.int(), pos_visibility, file_name=model_id.split('/')[1] + '_gt', save_dir="./videos")

    mark_ids = [key for key in pos_traces_to_mark_filtered.keys()] + [key for key in neg_traces_to_mark.keys()]    
    mark_ids = sorted(mark_ids)
    prompt = prompt_type.format(global_task, task_description, mark_ids)
    print(prompt)
    prompt = '<image>' + '\n' + prompt

    import pdb; pdb.set_trace()
    # get history positions for pos_traces_marks
    mark_trace_history = ""
    for mark_id, trace in pos_traces_to_mark_filtered.items():
        val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace[0, :frame_pos+1-frame_start].int().tolist()]) + ']'
        mark_trace_history += f"\"Mark {mark_id}\":\"{val_str_history}\"\n"
    mark_trace_history = mark_trace_history.strip()

    conv_user = (
        f"In the image with {spatial_quant_size}x{spatial_quant_size} grids. "
        f"The history positions in the last {frame_pos-frame_start+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
        f"Now please tell me the {step_to_predict-frame_pos+frame_start-1} future positions\n"
    )

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt+conv_user)
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
    traces_str = response.split("Their future positions for the target marks are:")[-1]
    
    # extract x, y coordinates in all [ ] in the string    
    traces_str = '{' + traces_str.strip().replace('\n', ',') + '}'                    
    traces_dict = ast.literal_eval(traces_str)
    overlay_traces = torch.cat([torch.tensor(ast.literal_eval('[' + trace[2:] if trace.startswith(']') else trace)).unsqueeze(1) for mark_id, trace in traces_dict.items()], dim=1).unsqueeze(0)
    overlay_traces = overlay_traces.float() / spatial_quant_size 
    overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
    overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
    images = [image] * overlay_traces.shape[1]
    overlay_visibility = overlay_traces.new(*overlay_traces.shape[:-1]).bool().fill_(True)
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    draw_visual_trace(video, overlay_traces, overlay_visibility, file_name=model_id.split('/')[1], save_dir="./videos")
    import pdb; pdb.set_trace()
