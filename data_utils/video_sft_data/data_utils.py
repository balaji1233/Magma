import torch
import torchvision
import re
import cv2
import numpy as np
from magma_utils.visual_trace import visual_trace
import os
import sys
import yaml
import json
from PIL import Image
from magma_utils.visual_trace import visual_trace
from magma_utils.som import som_prompting, tom_prompting
from data_utils.conversations import Constructor

class VideoSFT(Constructor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.mm_use_image_start_end = kwargs.get('mm_use_image_start_end', True)
        self.max_num_frames = kwargs.get('max_num_frames', 16)

    def __call__(self, **kwargs):
        return super()._construct_video_sft_conv(**kwargs)

    def filter_items(self, items):
        """
        filter out items that are not suitable for conversation construction
        """
        ann_file_path =  os.path.join(self.root_dir, self.settings['annotation_file'])
        ann_data = json.load(open(ann_file_path, "r"))['database']
        ann_data = {k: v for i, (k, v) in enumerate(ann_data.items())}
        vid2idx = json.load(open(os.path.join(self.root_dir, "annotations/video2dir_idx.json"), "r"))
        classes = set([elem['class'] for elem in ann_data.values()])

        filtered_items = []
        for item in items:
            video_path = item['video'][0]
            video_name = os.path.basename(video_path)
            video_name = video_name.split('.')[0]
            # remove invalid class
            if ann_data[video_name]['class'] not in self.valid_classes:
                continue
            # remove closeup videos
            if 'closeup' in item['gpt_response'][0] or \
                'close-up' in item['gpt_response'][0] or \
                    'close up' in item['gpt_response'][0] or \
                        'What you should do next' not in item['gpt_response'][0]:
                continue

            filtered_items.append(item)
        print(f"Filtered {len(items) - len(filtered_items)} items from {len(items)} items")
        return filtered_items