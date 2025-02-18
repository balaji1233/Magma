import os
import sys
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import pandas as pd
import torch
import deepspeed
import glob
import pandas as pd
import transformers
import tokenizers
import random
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
import torch.distributed as dist
import collections
from PIL import Image
from io import BytesIO
from magma.utils.mm_utils import tokenizer_image_token
from magma.utils import conversation as conversation_lib
from magma.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from .data_item import DataItem
from . import *
from PIL import Image, ImageFile
from PIL import ImageDraw, ImageFont
from typing import List, Optional, Union
from decord import VideoReader, cpu
import numpy as np

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: None
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    processor: MagmaProcessor,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "agent": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3_orig(
    sources,
    processor: MagmaProcessor,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "user": conv.roles[0], "gpt": conv.roles[1], "assistant": conv.roles[1], "agent": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    tokenizer = processor.tokenizer
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep2
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        conversation_sys = conversation.split(conv.sep)[0]
        conversation = conversation[len(conversation_sys):]
        rounds = conversation.split(conv.sep)[1:]
        cur_len = len(tokenizer_image_token(conversation_sys, tokenizer))
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] = conv.sep + parts[0] + sep
            rou = conv.sep + rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources,
    processor: MagmaProcessor,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "user": conv.roles[0], "gpt": conv.roles[1], "assistant": conv.roles[1], "agent": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    tokenizer = processor.tokenizer
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep2
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        conversation_sys = conversation.split(conv.sep)[0]
        conversation = conversation[len(conversation_sys):]
        rounds = conversation.split(conv.sep)[1:]
        cur_len = len(tokenizer(conversation_sys).input_ids)
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] = conv.sep + parts[0] + sep
            rou = conv.sep + rou

            round_len = len(tokenizer(rou).input_ids) - 1
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "agent": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        new_source = []
        for j in range(0, len(source), 2):
            if len(source[j]["value"]) == 0 or len(source[j+1]["value"]) == 0:
                print("wrong conversation")
                continue
            new_source.append(source[j])
            new_source.append(source[j+1])

        conv.messages = []
        for j, sentence in enumerate(new_source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_mistral(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "agent": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.MISTRAL
    sep = "[/INST]"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    processor: MagmaProcessor, 
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:        
        return preprocess_llama_3(sources, processor, has_image=has_image)        
        # llava_out = preprocess_llama_3_orig(sources, processor, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.MISTRAL:
        return preprocess_mistral(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                processor: MagmaProcessor, 
                data_items: Dict,
                dataset_names: List[str],
                dataset_folders: List[str],
                data_args: None):
        super(LazySupervisedDataset, self).__init__()
        self.processor = processor
        self.data_args = data_args
        self.data_items = []
        self.conv_constructor = {}
        if dataset_names is not None:
            for dataset_name, dataset_folder in zip(dataset_names, dataset_folders):
                if dataset_name in ['llava','seeclick', 'layerstack', 'sharegpt4v']:
                    self.data_items.extend(data_items[dataset_name])
                elif dataset_name in ['llava-video-178k', 'video-llava', 'sharegpt4video']:
                    self.conv_constructor[dataset_name] = eval('video_sft_data')(
                        mm_use_trace_start_end=data_args.mm_use_trace_start_end,
                        mm_use_trace_speed=data_args.mm_use_trace_speed,
                        mm_use_image_start_end=data_args.mm_use_image_start_end,
                        remove_static_trace_pts=data_args.remove_static_trace_pts,
                        spatial_quant_size=data_args.spatial_quant_size,  
                        mm_use_vlm_template=data_args.mm_use_vlm_template,                  
                        dataset_folder=dataset_folder, 
                        show_trace=data_args.show_trace,
                        task=data_args.task,
                        tokenizer=processor.tokenizer,
                    )
                    self.data_items.extend(data_items[dataset_name])
                else:
                    self.conv_constructor[dataset_name] = eval(dataset_name)(
                        mm_use_trace_start_end=data_args.mm_use_trace_start_end,
                        mm_use_trace_speed=data_args.mm_use_trace_speed,
                        mm_use_image_start_end=data_args.mm_use_image_start_end,
                        remove_static_trace_pts=data_args.remove_static_trace_pts,
                        max_num_frames=data_args.max_num_frames,
                        spatial_quant_size=data_args.spatial_quant_size,                    
                        dataset_folder=dataset_folder, 
                        show_trace=data_args.show_trace,
                        task=data_args.task,
                        tokenizer=processor.tokenizer,
                    )
                    final_items = self.conv_constructor[dataset_name].filter_items(data_items[dataset_name])
                    self.data_items.extend(final_items)

    def __len__(self):
        return len(self.data_items)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data_items:
            img_tokens = 128 if ('image' in sample and sample['image'] is not None) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data_items:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample and sample['image'] is not None) else -cur_len
            length_list.append(cur_len)
        return length_list

    def _construct_conversations_video_sft(self, item):
        """
        Construct conversations for SFT on video datasets
        """
        conversations = []
        for curr in item['conversations']:
            image_placeholder = ''.join(['<image_start><image><image_end>'] * self.data_args.max_num_frames)
            if '<video>' in curr['value']:
                curr['value'] = curr['value'].replace('<video>', image_placeholder)
            else:
                curr['value'] = curr['value'].replace('<image>', image_placeholder)

            conversations.append(curr)

        item['conversations'] = conversations
        return item

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = copy.deepcopy(self.data_items[i])

        if 'video' in item and item['video'][0] is not None:
            assert item['image_folder'] is not None or self.data_args.image_folder is not None, "image_folder is not provided"
            image_folder = self.data_args.image_folder if self.data_args.image_folder is not None else item['image_folder']
            if item['dataset_tag'] in ['sthv2', 'howto100m', 'ego4d']:
                visual_trace_path = os.path.join(image_folder, item['trace'])
                if os.path.exists(visual_trace_path):
                    try:
                        visual_traces = torch.load(visual_trace_path, map_location='cpu')
                        video_path = os.path.join(image_folder, item['video'])
                        item.update(visual_traces)
                    except Exception as e:
                        print(f"Error loading: {visual_trace_path}")
                        visual_traces = None
                        video_path = None
                else:                
                    print(f"Error: {visual_trace_path} not found")    
                    visual_traces = None       
                    video_path = None
                item = self.conv_constructor[item['dataset_tag']](item=item, video_path=video_path, visual_traces=visual_traces)
            elif item['dataset_tag'] in ['llava-video-178k', 'video-llava', 'sharegpt4video']:
                if 'valley' in item['video']:
                    video_path = os.path.join(item['image_folder'], 'valley', item['video'])
                elif 'videochatgpt' in item['video']:
                    video_path = os.path.join(item['image_folder'], item['video'])
                elif item['dataset_tag'] == 'sharegpt4video':
                    tmp = item['video'].split('/')
                    video_path = os.path.join(item['image_folder'], tmp[0], 'videos', tmp[1])
                else:
                    source_dir = item['source_dir']

                    if source_dir == 'llava_hound':
                        video_path = os.path.join(item['image_folder'], 'LLaVA-Video-178K',  'videos', item['video'])
                    else:
                        video_path = os.path.join(item['image_folder'], 'LLaVA-Video-178K',  source_dir, 'videos', item['video'])

                item['image_size'] = (self.processor.image_processor.base_img_size, self.processor.image_processor.base_img_size)
                item = self.conv_constructor[item['dataset_tag']](item=item, video_path=video_path, num_frames=self.data_args.max_num_frames)

                frames = item['image']
                images = collections.defaultdict(list)
                for curr in range(len(frames)):
                    curr_frame = Image.fromarray(frames[curr])
                    image_pt = self.processor.image_processor(curr_frame, return_tensors='pt')
                
                    for key, val in image_pt.items():
                        images[key].append(val)

                texts = copy.deepcopy([item["conversations"]])
            else:
                item['video'][0] = item['video'][0].replace('/mnt/data/video_datasets_visual_traces/YouCook2/', '')
                video_path = os.path.join(image_folder, item['video'][0])
                frame_start, frame_end = item['frame_interval'][0].item(), item['frame_interval'][1].item()
                video_name = os.path.basename(video_path).split('.')[0]
                if 'youcook2' in video_path.lower():
                    visual_trace_path = os.path.join(image_folder, 'all_detected_visual_traces_30fps', f'{video_name}_trace_{frame_start:09d}_{frame_end:09d}.pth')
                else:
                    visual_trace_path = os.path.join(image_folder, 'visual_trace' if 'epic' in image_folder else 'visual_traces', video_name, f'trace_{frame_start:09d}_{frame_end:09d}.pth')
                if os.path.exists(visual_trace_path):
                    visual_traces = torch.load(visual_trace_path, map_location='cpu')
                else:
                    visual_traces = None
                item = self.conv_constructor[item['dataset_tag']](item=item, video_path=video_path, visual_traces=visual_traces)    
            
            if item['dataset_tag'] not in ['llava-video-178k', 'video-llava']:
                image = item['image']
                # if image is not a PIL image
                if image is None:  
                    base_img_size = self.processor.image_processor.base_img_size
                    image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
                    item['image'] = image

                image_pt = self.processor.image_processor(image, return_tensors='pt')
                images = collections.defaultdict(list)
                for key, val in image_pt.items():
                    images[key].append(val)
                texts = [item["conversations"]]
        elif 'image' in item and item['image'] is not None:
            # cope with multiple images
            image_folder = item['image_folder']
            image_files = item['image']
            if isinstance(image_files, str):
                image_files = [image_files]

            images = collections.defaultdict(list)
            for image_file in image_files:
                image_file = image_file[1:] if image_file.startswith('/') else image_file
                image_path = os.path.join(image_folder, image_file)

                try:
                    if "trace" in self.data_items[i]:
                        trace_file = self.data_items[i]["trace"]
                        trace_path = os.path.join(image_folder, trace_file)
                        if os.path.exists(trace_path):
                            visual_traces = torch.load(trace_path, map_location='cpu')
                            item.update(visual_traces)
                        else:
                            visual_traces = None
                        video_path = image_path
                        item = self.conv_constructor[item['dataset_tag']](item=item, video_path=image_path, visual_traces=visual_traces)               
                        image = item['image']
                        if image is None:  
                            base_img_size = self.processor.image_processor.base_img_size
                            image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
                        image = self.processor.image_processor(image, return_tensors='pt')
                    else:
                        image = Image.open(image_path).convert('RGB')           
                        image = self.processor.image_processor(image, return_tensors='pt')
                    for key, val in image.items():
                        images[key].append(val)

                except Exception as e:
                    print(f"Error: {e}")
                    base_img_size = self.processor.image_processor.base_img_size
                    image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))                    
                    image = self.processor.image_processor(image, return_tensors='pt')
                    for key, val in image.items():
                        images[key].append(val)

            texts = [item["conversations"]]
        else:
            images = collections.defaultdict(list)
            # image does not exist in the data, but the model is multimodal
            base_img_size = self.processor.image_processor.base_img_size
            image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
            image = self.processor.image_processor(image, return_tensors='pt')
            for key, val in image.items():
                images[key].append(val)
            item["conversations"][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + item["conversations"][0]['value']
            texts = [item["conversations"]]
        data_dict = preprocess(
            texts,
            self.processor,
            has_image=(('image' in item and item['image'] is not None) or ('video' in item and item['video'] is not None))
        )        
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict.update(images)
        data_dict.update(
            {
                "dataset_name": item["dataset_tag"],
                "item_id": i
            }
        )
        del item
        return data_dict


# Custom wrapper to combine Dataset and IterableDataset without loading IterableDataset in memory
class CombinedDataset(Dataset):
    def __init__(self, dataset, iterable_dataset, local_run=False, seed=7):
        self.dataset_len = []
        if dataset is not None:
            self.dataset_len.append(len(dataset)) # Length of the Dataset   
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset, 
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    seed=seed,
                    drop_last=False,
                )
            else:
                sampler = None            
            self.iterable_dataset_a = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0 if local_run else 8, pin_memory=False)  # DataLoader for the Dataset
            self.iterable_iter_a = iter(self.iterable_dataset_a)
        else:
            self.iterable_dataset_a = None
            self.iterable_iter_a = None
            self.dataset_len.append(0)

        if iterable_dataset is not None:  
            self.dataset_len.append(len(iterable_dataset)) # Length of the IterableDataset   
            self.iterable_dataset_b = iterable_dataset
            self.iterable_iter_b = iter(self.iterable_dataset_b)  # Iterator for the IterableDataset
        else:
            self.iterable_dataset_b = None
            self.iterable_iter_b = None
            self.dataset_len.append(0)        
        self.sampling_ratios = [float(item)/sum(self.dataset_len) for item in self.dataset_len]
        print(f"total training data size: {sum(self.dataset_len)}")
        print(f"sampling ratios: {self.sampling_ratios}")

    def __len__(self):
        # Length can be the maximum of both or some other logic
        return sum(self.dataset_len)
    
    def __getitem__(self, index):
        # according to the sampling ratio, choose which dataset to sample
        dataset_choice = random.choices([0, 1], self.sampling_ratios)[0]
        if dataset_choice == 0:
            # Fetch a sample from the IterableDataset using its iterator
            try:
                iterable_sample_a = next(self.iterable_iter_a)
            except StopIteration:
                # Reinitialize the iterator if it exhausts
                self.iterable_iter_a = iter(self.iterable_dataset_a)
                iterable_sample_a = next(self.iterable_iter_a)
            iterable_sample_a['input_ids'] = iterable_sample_a['input_ids'][0]
            iterable_sample_a['labels'] = iterable_sample_a['labels'][0]
            iterable_sample_a['pixel_values'] = [item[0] for item in  iterable_sample_a['pixel_values']]
            iterable_sample_a['image_sizes'] = [item[0] for item in  iterable_sample_a['image_sizes']]
            return iterable_sample_a
        else:
            # Fetch a sample from the IterableDataset using its iterator
            try:
                iterable_sample_b = next(self.iterable_iter_b)
            except StopIteration:
                # Reinitialize the iterator if it exhausts
                self.iterable_iter_b = iter(self.iterable_dataset_b)
                iterable_sample_b = next(self.iterable_iter_b)
            # print(f"oxe-rank-{rank}: {iterable_sample_b['dataset_name']}")
            # Return a combined sample (modify based on your requirement)
            return iterable_sample_b

def build_joint_dataset(
        data_path: str,
        processor: MagmaProcessor,
        data_args: None, 
        is_eval: bool = False
    ) -> torch.utils.data.ConcatDataset:

    if data_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    data_items, dataset_names, dataset_folders = DataItem(training_size=data_args.training_size, local_run=data_args.local_run)(data_path, processor, conversation_lib, is_eval=is_eval)

    # pop out open-x dataset
    openx_dataset = None
    if 'openx_orig' in data_items:
        openx_dataset = data_items.pop('openx_orig')
        _ = dataset_folders.pop(dataset_names.index('openx_orig'))
        _ = dataset_names.pop(dataset_names.index('openx_orig'))

        lazy_dataset = None
        if len(data_items) > 0:
            lazy_dataset = LazySupervisedDataset(processor, data_items, dataset_names, dataset_folders, data_args)

        # concatenate openx dataset and lazy_dataset
        return CombinedDataset(lazy_dataset, openx_dataset, local_run=data_args.local_run)
    else:
        return LazySupervisedDataset(processor, data_items, dataset_names, dataset_folders, data_args)

