import json
import yaml
import torch
import random
import os
import sys
import glob
import pickle
#from .openx import OpenXDataItem
from tqdm import tqdm

class DataItem:
    """
    Curate data items from all data sources
    """
    def __init__(self, training_size=-1, local_run=False):
        self.training_size = training_size
        self.local_run = local_run

    def _get_dataset_tag(self, data_path):
        if "coin" in data_path.lower():
            return "coin"
        elif "epic" in data_path.lower():
            return "epic"
        elif "youcook2" in data_path.lower():
            return "youcook2"       
        # TODO: this is a temporary fix
        elif "baolinpeng/open-x" in data_path:
            return "openx_orig"     
        elif "open-x" in data_path or "openx" in data_path:
            if 'traces' in data_path:
                return "openx_magma"
            else:
                return "openx"
        elif "sthv2" in data_path.lower():
            return "sthv2"
        elif 'ego4d' in data_path.lower():
            return "ego4d"
        elif 'howto100m' in data_path.lower():
            return "howto100m"
        elif 'seeclick' in data_path.lower():
            return "seeclick"
        elif 'llava-video-178k' in data_path.lower():
            return "llava-video-178k"
        elif 'video-llava' in data_path.lower():
            return 'video-llava'
        elif 'llava' in data_path.lower():
            return "llava"
        elif 'layerstack' in data_path.lower():
            return 'layerstack'
        elif 'sharegpt4video' in data_path.lower():
            return "sharegpt4video"
        elif 'sharegpt4v' in data_path.lower():
            return "sharegpt4v"
        else:
            raise ValueError(f"Dataset tag not found for {data_path}")

    def llava_to_caption(self, caption, is_video=False, num_frames=None):
        transformed_data = []

        if is_video:
            video_string = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
            count = 1
            while LLAVA_IMAGE_TOKEN in video_string:
                video_string = video_string.replace(LLAVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
                count += 1
        prompt = 'Given a sequence of key frames from a video, provide a concise yet comprehensive description of the overall content and context, noting any significant events, characters, or objects that appear throughout. Ensure the description follows the temporal sequence of the video, highlighting the progression and key moments as they occur.'
        content = f"{video_string}\n{prompt}"

        transformed_data.append(
            {
                'role': 'user',
                'content': content,
            }
        )
        transformed_data.append(
            {
                'role': 'assistant',
                'content': caption,
            }
        )
        return transformed_data
    
    def _get_items(self, data_path, image_folder=None, processor=None, conversation_lib=None):
        if data_path.endswith(".json"):
            list_data_dict = json.load(open(data_path, "r"))
        elif data_path.endswith(".jsonl"):
            list_data_dict = [json.loads(line) for line in open(data_path, "r")]
        elif data_path.endswith(".pth"):
            list_data_dict = torch.load(data_path, map_location="cpu")
            # random.shuffle(list_data_dict)
        else:
            if self._get_dataset_tag(data_path) == "openx_orig":
                list_data_dict = OpenXDataItem()(data_path, image_folder, processor=processor, conversation_lib=conversation_lib, local_run=self.local_run)
            elif self._get_dataset_tag(data_path) == "llava-video-178k":
                list_data_dict = []
                for idx, curr in enumerate(os.listdir(data_path)):
                    curr_dir = os.path.join(data_path, curr)
                    if os.path.isdir(curr_dir):
                        for curr_file in os.listdir(curr_dir):
                            if '.json' in curr_file:
                                curr_data = json.load(open(os.path.join(curr_dir, curr_file), 'r'))

                                for sample in curr_data:
                                    if 'video' in sample:
                                        sample['source_dir'] = curr
                                        list_data_dict.append(sample)

            elif self._get_dataset_tag(data_path) == "video-llava":
                downsampled_ann_path = os.path.join(data_path, 'train_json', 'downsampled_subset.json')
                list_data_dict = json.load(open(downsampled_ann_path, "r"))
            elif self._get_dataset_tag(data_path) == "sharegpt4v":
                all_files = ['share-captioner_coco_lcs_sam_1246k_1107.json', 'sharegpt4v_instruct_gpt4-vision_cap100k.json', 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json']
                list_data_dict = []
                for curr_file in all_files:
                    curr_file_path = os.path.join(data_path, curr_file)
                    data = json.load(open(curr_file_path, 'r'))
                    list_data_dict.extend(data)
            else:
                data_folder = os.path.dirname(data_path)
                # get file name from data_path
                data_files = data_path.split('/')[-1].split('+')
                list_data_dict = []
                for file in data_files:
                    json_path = os.path.join(data_folder, file + '.json')      
                    list_data_dict.extend(json.load(open(json_path, "r")))

        return list_data_dict
    
    def __call__(self, data_path, processor=None, conversation_lib=None, is_eval=False):
        assert data_path is not None, "Data path is not provided"
        if data_path.endswith(".yaml"):
            data_dict = yaml.load(open(data_path, "r"), Loader=yaml.FullLoader)    
            data_path_key = 'DATA_PATH' if not is_eval else 'DATA_PATH_VAL'
            image_folder_key = 'IMAGE_FOLDER' if not is_eval else 'IMAGE_FOLDER_VAL'
            assert len(data_dict[data_path_key]) == len(data_dict[image_folder_key]), "Data path and image folder mismatch"
            items = {}
            dataset_names = []
            dataset_folders = []
            for i, (data_path, image_folder) in enumerate(zip(data_dict[data_path_key], data_dict[image_folder_key])):
                items_temp = self._get_items(data_path, image_folder, processor, conversation_lib)                
                dataset_tag = self._get_dataset_tag(data_path)                
                if dataset_tag != "openx_orig":
                    if self.training_size > 0:
                        items_temp = items_temp[:self.training_size]             
                    if dataset_tag in ['sthv2', "ego4d", "howto100m"]: 
                        for item in items_temp:
                            item['image_folder'] = image_folder
                            item['dataset_tag'] = dataset_tag
                            item['gpt_response'] = ''
                            item['global_instructions'] = item['annotations']
                    elif dataset_tag in ["openx_magma"]:
                        items_dict_temp = []
                        for item in items_temp:
                            items_dict_temp.append(
                                {
                                    'image': item.replace('traces', 'images').replace('.pth', '.jpg'),
                                    'trace': item,
                                    'image_folder': image_folder,
                                    'dataset_tag': dataset_tag
                                }
                            ) 
                        items_temp = items_dict_temp         
                    else:
                        # add image_foler to each item
                        for item in items_temp:
                            item['image_folder'] = image_folder
                        # add dataset tag to each item
                        for item in items_temp:
                            item['dataset_tag'] = dataset_tag
                items[dataset_tag] = items_temp
                dataset_names.append(dataset_tag)
                dataset_folders.append(image_folder)

        else:
            items = self._get_items(data_path)
            dataset_names = None
            dataset_folders = None 

        return items, dataset_names, dataset_folders