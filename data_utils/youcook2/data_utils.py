import torch
import torchvision
import re
import cv2
import numpy as np
from magma_utils.visual_trace import visual_trace
import os
import yaml
import json
from PIL import Image
from magma_utils.visual_trace import visual_trace
from magma_utils.som import som_prompting, tom_prompting
from data_utils.conversations import Constructor

class YouCook2(Constructor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.valid_classes = self.settings['valid_classes']

    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)

    def filter_items(self, items):
        """
        filter out items that are not suitable for conversation construction
        """
        filtered_items = []
        for item in items:
            # remove closeup videos
            if 'closeup' in item['gpt_response'][0] or \
                'close-up' in item['gpt_response'][0] or \
                    'close up' in item['gpt_response'][0]:
                continue
            filtered_items.append(item)
        return filtered_items

    # def _construct_conv(self, item, image, frame_pos, visual_traces=None):
    #     """
    #     v5->v6: smoothen the traces, 
    #     """
    #     width, height = item['image_size'][0].item(), item['image_size'][1].item()
    #     task_description = item['global_instructions'][0]
    #     gpt_response = item['gpt_response'][0]
    #     gpt_response = gpt_response.replace('What you see', 'What I see')
    #     gpt_response = gpt_response.replace('you', 'the person')
    #     gpt_response = gpt_response.replace('your', '')
    #     gpt_response = gpt_response.replace('In the first image, ', '')
    #     # gpt_response = gpt_response.replace('What the person should do next', 'What you should do next')
    #     # gpt_response = gpt_response.replace('What you should do next', 'What you are doing')
    #     gpt_response = gpt_response.replace('personr', 'person\'s')
    #     # remove all str (marks) from the gpt_response
    #     gpt_response = re.sub(r' \([^)]*\)', '', gpt_response)

    #     gpt_response = gpt_response if len(gpt_response) > 0 else task_description
    #     step_to_predict = item['step_to_predict'].item()
    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(gpt_response) > 0:       
    #         conv_user = f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please look at the image and briefly describe what you are seeing and what the person should do next.\n'     
    #     else:
    #         conv_user = f'<image>\nYou are an expert on world physics. Please look at the image and briefly describe what you are seeing in the image and what happen next.\n'     
    #     # conv_user = f'<image>\nI have labeled a few numeric marks on the image. Please briefly describe what you are seeing in the image.\n'
    #     conv_gpt = gpt_response + '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
    #     pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]

    #     # only keep points that are visible at at least half steps
    #     pred_tracks = pred_tracks[:, :, pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]]
    #     pred_visibility = pred_visibility[:, :, pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]]

    #     # step 1: determine whether there are camera motions
    #     start_pos = pred_tracks[:, 0][0]
    #     end_pos = pred_tracks[:, -1][0]
    #     reference_pts_np = start_pos.cpu().numpy().reshape(-1, 2)
    #     future_pts_np = end_pos.cpu().numpy().reshape(-1, 2)
    #     try:
    #         (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
    #     except Exception as e:
    #         H = None
    #     if H is not None:
    #         camera_motion = False
    #         H = torch.tensor(H, dtype=torch.float32)
    #         rt_magnitude = torch.norm(H[:2, :2] - torch.eye(2)).item()
    #         scale_magnitude = torch.norm(H[:2, 2] / torch.Tensor([width, height])).item()
    #         distorsion_magnitude = torch.norm(H[2, :3]).item()
    #         if rt_magnitude > 0.1 or scale_magnitude > 0.1 or distorsion_magnitude > 0.1:
    #             camera_motion = True
            
    #         if camera_motion:
    #             # remove camera motion using homography transformation
    #             future_pts_transformed = []
    #             for k in range(1, pred_tracks.shape[1]):
    #                 future_pts = pred_tracks[:, k][0]
    #                 future_pts_np = future_pts.cpu().numpy().reshape(-1, 2)
    #                 try:
    #                     (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
    #                 except Exception as e:
    #                     continue
    #                 future_pts_np_transformed = cv2.perspectiveTransform(future_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
    #                 future_pts_transformed_k = torch.tensor(future_pts_np_transformed, dtype=torch.float32)
    #                 future_pts_transformed.append(future_pts_transformed_k)            
    #             pred_tracks = torch.stack([start_pos] + future_pts_transformed, dim=0).unsqueeze(0)           

    #     # step 2: find positive traces and negative traces
    #     track_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (width, height)).squeeze(0)
    #     max_length = track_length.max()
    #     threshold = max_length * 0.3
    #     if (track_length > threshold).sum() <= 1:      
    #         item['conversations'] = conversations
    #         return item
    #     else:
    #         # find the positive traces and negative traces
    #         pos_tracks = pred_tracks[:, :, track_length > threshold]
    #         pos_visibility = pred_visibility[:, :, track_length > threshold]

    #         neg_tracks = pred_tracks[:, :, track_length <= threshold]

    #         # clustering for positive traces
    #         pos_sampled_ids = self.trace.cluster_traces(pos_tracks, n_clusters=self.num_clusters)
    #         if pos_sampled_ids is None:
    #             item['conversations'] = conversations
    #             return item                
    #         pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
    #         pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]

    #         # clustering for negative traces
    #         neg_sampled_ids = self.trace.cluster_traces(neg_tracks, n_clusters=2*self.num_clusters)
    #         if neg_sampled_ids is None:
    #             item['conversations'] = conversations
    #             return item                
    #         neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]

    #         pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)

    #         # visualize the traces
    #         # images = [image] * pos_tracks.shape[1]
    #         # video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #         # self.trace.visualizer.save_dir = "./videos"
    #         # res_video = self.trace.visualize(
    #         #     video, 
    #         #     pos_tracks, 
    #         #     pos_visibility, 
    #         #     filename="visual_trace_to_predict", 
    #         #     mode="rainbow"
    #         # )
    #         # import pdb; pdb.set_trace()

    #         mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])

    #         pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
    #         # speeds = []
    #         # pos_mark_ids = []
        
    #         mark_trace_history = ''
    #         mark_trace_future = ''

    #         valid_marks = {}
    #         for key, val in pos_traces_to_mark.items():
    #             # random select a frame position but not the last frame
    #             # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
    #             trace = val[0]
    #             future_steps_random = trace.shape[0]
    #             trace[:, 0] = self.spatial_quant_size * trace[:, 0].clamp_(0, width-1) / width
    #             trace[:, 1] = self.spatial_quant_size * trace[:, 1].clamp_(0, height-1) / height

    #             # trace_history = trace[0]
    #             # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #             # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
    #             valid_marks[key] = trace[0]

    #             trace_future = trace[1:]
    #             val_str_future = '[' + ','.join([f'[{x[0]:.1f},{x[1]:.1f}]' for x in trace_future.tolist()]) + ']'
    #             mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
            
    #         if len(mark_trace_future) > 0:
    #             """
    #             visualize visual traces for debugging
    #             """          
    #             conv_user = (
    #                 f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with a number of numeric marks {mark_ids}.\n"
    #                 # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #                 f"Given what the person is doing next, please further tell me which marks will move and their {future_steps_random} future movements\n"
    #             )
    #             conversations.append({'from': 'human', 'value': conv_user})
    #             formmated_val = ', '.join([f"mark {key} at location [{val[0].item():.1f},{val[1].item():.1f}]" for key, val in valid_marks.items()])       
    #             conv_gpt = f"{formmated_val} will move, and their future movements are:\n\n{mark_trace_future}"
    #             conversations.append({'from': 'gpt', 'value': conv_gpt})
    #         item['conversations'] = conversations
    #         return item
