import torch
import torchvision
import re
import cv2
import numpy as np
from magma_utils.visual_trace import visual_trace
import os
import yaml
from PIL import Image
from magma_utils.visual_trace import visual_trace
from magma_utils.som import som_prompting, tom_prompting
from data_utils.conversations import Constructor

class Sthv2(Constructor):
    def __init__(self, **kwargs):
        super(Sthv2, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')

    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        return items

    # def _construct_conv(self, item, video_path, visual_traces, show_trace=False):
    #     """
    #     v4->v5: add trace of mark
    #     """        
    #     if isinstance(item['image_size'][0], torch.Tensor):
    #         width, height = item['image_size'][0].item(), item['image_size'][1].item()
    #         frame_start, frame_end = item['frame_interval'][0].item(), item['frame_interval'][1].item()
    #         task_description = item['global_instructions'][0]
    #         gpt_response = item['gpt_response'][0]
    #     else:
    #         width, height = item['image_size']
    #         frame_start, frame_end = item['frame_interval']     
    #         task_description = item['global_instructions']
    #         gpt_response = item['gpt_response']
    #     gpt_response = gpt_response.replace('What you see', 'What I see')
    #     gpt_response = gpt_response.replace('you see ', '').replace('You see ', '')
    #     gpt_response = gpt_response.replace('you', 'the person')
    #     gpt_response = gpt_response.replace('your', '')
    #     gpt_response = gpt_response.replace('In the first image, ', '')
    #     # gpt_response = gpt_response.replace('What the person should do next', 'What you should do next')
    #     # gpt_response = gpt_response.replace('What you should do next', 'What you are doing')
    #     gpt_response = gpt_response.replace('personr', 'person\'s')
    #     # remove all str (marks) from the gpt_response
    #     gpt_response = re.sub(r' \([^)]*\)', '', gpt_response)
    #     gpt_response = gpt_response if len(gpt_response) > 0 else task_description

    #     # step_to_predict = item['step_to_predict'].item()
    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(gpt_response) > 0:       
    #         conv_user = (
    #             f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the moving traces for all marks thus far on the image. Tell me what you are seeing and what specific task the person is doing.\n'
    #         ) 
    #         conv_gpt = gpt_response + '\n'
    #     else:
    #         conv_user = (
    #             f'<image>\nYou do not need to say anything.\n'     
    #         ) 
    #         conv_gpt = '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     video_cap = cv2.VideoCapture(video_path)

    #     if visual_traces is None:
    #         item['conversations'] = conversations
    #         item['image'] = self._get_frame(video_cap, frame_start, (width, height))
    #         video_cap.release()
    #         return item
            
    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     # calculate the trace length for each step
    #     track_length = torch.norm(visual_traces['pred_tracks'][:, 1:] - visual_traces['pred_tracks'][:, :-1], dim=3).mean(2)
    #     # accum_sum track_length
    #     accum_sum = torch.cumsum(track_length, dim=1) / (1e-5 + track_length.sum(1)[:, None])
    #     # find last position that is less than 0.7
    #     frame_rightmost = (accum_sum[0] < 0.7).int().sum().item()

    #     # random select a frame position but not the last frame
    #     frame_pos = torch.randint(0, frame_rightmost, (1,)).item()
    #     pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
    #     pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]
    #     step_to_predict = pred_tracks.size(1)

    #     if step_to_predict == 0:
    #         item['conversations'] = conversations
    #         item['image'] = self._get_frame(video_cap, frame_start, (width, height))    
    #         video_cap.release()
    #         return item
        
    #     pred_tracks_history = visual_traces['pred_tracks'][:, :max(1, frame_pos)]
    #     pred_visibility_history = visual_traces['pred_visibility'][:, :max(1, frame_pos)]

    #     # only keep points that are visible at at least half steps
    #     valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]

    #     if valid_idx.sum() <= 1:
    #         item['conversations'] = conversations
    #         item['image'] = self._get_frame(video_cap, frame_start, (width, height))
    #         video_cap.release()
    #         return item
        
    #     pred_tracks = pred_tracks[:, :, valid_idx]
    #     pred_visibility = pred_visibility[:, :, valid_idx]

    #     pred_tracks_history = pred_tracks_history[:, :, valid_idx]
    #     pred_visibility_history = pred_visibility_history[:, :, valid_idx]

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

    #             if pred_tracks_history.size(1) > 0:
    #                 history_pts_transformed = []
    #                 for k in range(0, pred_tracks_history.shape[1]):
    #                     history_pts = pred_tracks_history[:, k][0]
    #                     history_pts_np = history_pts.cpu().numpy().reshape(-1, 2)
    #                     try:
    #                         (H, status) = cv2.findHomography(history_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
    #                     except Exception as e:
    #                         continue
    #                     history_pts_np_transformed = cv2.perspectiveTransform(history_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
    #                     history_pts_transformed_k = torch.tensor(history_pts_np_transformed, dtype=torch.float32)
    #                     history_pts_transformed.append(history_pts_transformed_k)            
    #                 pred_tracks_history = torch.stack(history_pts_transformed, dim=0).unsqueeze(0)   

    #     # step 2: find positive traces and negative traces
    #     track_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (width, height)).squeeze(0)
    #     max_length = track_length.max()
    #     threshold = max_length * 0.3

    #     # video is almost static
    #     if (track_length > threshold).sum() <= 1:      
    #         item['conversations'] = conversations
    #         item['image'] = self._get_frame(video_cap, frame_start, (width, height))  
    #         video_cap.release()
    #         return item
    #     else:
    #         # find the positive traces and negative traces
    #         pos_tracks = pred_tracks[:, :, track_length > threshold]
    #         pos_visibility = pred_visibility[:, :, track_length > threshold]
    #         pos_tracks_history = pred_tracks_history[:, :, track_length > threshold]
    #         pos_visibility_history = pred_visibility_history[:, :, track_length > threshold]

    #         neg_tracks = pred_tracks[:, :, track_length <= threshold]
    #         neg_tracks_history = pred_tracks_history[:, :, track_length <= threshold]

    #         # clustering for positive traces
    #         pos_sampled_ids = self.trace.cluster_traces(pos_tracks, n_clusters=self.num_clusters)
    #         if pos_sampled_ids is None:
    #             item['conversations'] = conversations
    #             item['image'] = self._get_frame(video_cap, frame_start, (width, height))     
    #             video_cap.release()
    #             return item                
    #         pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
    #         pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]
    #         pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
    #         pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

    #         # clustering for negative traces
    #         neg_sampled_ids = self.trace.cluster_traces(neg_tracks, n_clusters=2*self.num_clusters)
    #         if neg_sampled_ids is None:
    #             item['conversations'] = conversations
    #             item['image'] = self._get_frame(video_cap, frame_start, (width, height))
    #             video_cap.release()
    #             return item                
    #         neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]
    #         neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

    #         image = self._get_frame(video_cap, frame_start + frame_pos, (width, height))
    #         if image is None:
    #             item['conversations'] = conversations
    #             item['image'] = image
    #             video_cap.release()
    #             return item
            
    #         image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)
    #         image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
    #             som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)
            
    #         # visualize the traces
    #         if show_trace:
    #             images = [image] * pos_tracks.shape[1]
    #             video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #             self.trace.visualizer.save_dir = "./videos"
    #             _ = self.trace.visualize(video, pos_tracks, pos_visibility, filename="visual_trace_to_predict", mode="rainbow")

    #         mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])
    #         pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
        
    #         mark_trace_history = ''
    #         mark_trace_future = ''

    #         valid_marks = {}
    #         speeds = {}
    #         for key, val in pos_traces_to_mark.items():
    #             # random select a frame position but not the last frame
    #             # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
    #             trace = val[0]
    #             trace[:, 0] = self.spatial_quant_size * trace[:, 0] / width
    #             trace[:, 1] = self.spatial_quant_size * trace[:, 1] / height

    #             trace_temp = trace.clone()
    #             # remove (almost) static points
    #             trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 1]], dim=0)
    #             # remove invisible points
    #             trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
    #             if trace_temp.size(0) <= step_to_predict // 4:
    #                 continue
    #             # calulate motion speed
    #             # if trace_temp.size(0) < step_to_predict:
    #             #     trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
    #             # elif trace_temp.size(0) > step_to_predict:
    #             #     trace_temp = trace_temp[:step_to_predict]   

    #             # calcualte speed
    #             speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean().item()
    #             speeds[key] = speed
    #             if speed < 1.0:
    #                 continue                
    #             # trace_history = trace[0]
    #             # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #             # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
    #             # round trace_temp
    #             if self.remove_static_trace_pts:
    #                 valid_marks[key] = trace_temp.int()
    #             else:
    #                 valid_marks[key] = trace.int()

    #             # NOTE: there was a bug here
    #             val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in valid_marks[key][1:].tolist()]) + ']'

    #             mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
            
    #         if len(mark_trace_future) > 0:
    #             """
    #             visualize visual traces for debugging
    #             """
    #             # 0923-1am
    #             conv_user = (
    #                 # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #                 f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
    #                 f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please tell me which marks will move and their movements in the next {step_to_predict-1} time window.\n"
    #             )
    #             conversations.append({'from': 'human', 'value': conv_user})
    #             formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move {val.shape[0]-1} steps" for key, val in valid_marks.items()])       
    #             conv_gpt = f"{formmated_val}. Their future movements are:\n\n{mark_trace_future}"
    #             conversations.append({'from': 'gpt', 'value': conv_gpt})          

    #             # 0923-4am
    #             # conv_user = (
    #             #     # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #             #     f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
    #             #     f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please tell me which marks will move and their movements in the next {step_to_predict-1} time window.\n"
    #             # )
    #             # conversations.append({'from': 'human', 'value': conv_user})
    #             # formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move" for key, val in valid_marks.items()])   
    #             # if self.mm_use_trace_start_end:
    #             #     mark_trace_future = f'<trace_start>{mark_trace_future}<trace_end>'      
    #             # conv_gpt = f"{formmated_val}. Their future movements are:\n\n{mark_trace_future}"
    #             # conversations.append({'from': 'gpt', 'value': conv_gpt})      

    #             # 0922:
    #             # conv_user = (
    #             #     # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #             #     f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
    #             #     f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please further tell me which marks will move and their {step_to_predict-1} future movements\n"
    #             # )

    #         item['conversations'] = conversations
    #         item['image'] = image
    #         video_cap.release()
    #         return item

    # def _construct_epic2(self, item, image, frame_pos):
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
        
    #     pos_traces_to_mark = item['pos_traces_to_mark']
    #     neg_traces_to_mark = item['neg_traces_to_mark']
    #     mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])

    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(task_description) > 0:       
    #         conv_user = f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please briefly describe what you are seeing and what the person should do next.\n'     
    #     else:
    #         conv_user = f'<image>\nYou are an expert on world physics. Please briefly describe what you are seeing in the image and what happen next.\n'     
    #     # conv_user = f'<image>\nI have labeled a few numeric marks on the image. Please briefly describe what you are seeing in the image.\n'
    #     conv_gpt = gpt_response + '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model to generate a response to the given task description - language action for the future
    #     # if len(task_description) > 0:            
    #     #     conv_user = f'\nNow the task is {task_description}. Tell me what I should do next.\n'           
    #     # else:        
    #     #     conv_user = f'\nTell me what happen next.\n'
    #     # conv_gpt = gpt_response.split('\n\n')[1] + '\n'
    #     # conversations.append({'from': 'human', 'value': conv_user})
    #     # conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
    #     step_to_predict = item['step_to_predict'].item()
    #     speeds = []
    #     pos_mark_ids = []
     
    #     pos_traces_to_mark_filtered = {}
    #     for key, val in pos_traces_to_mark.items():
    #         # perform exponential moving smoothing for val
    #         trace = val[0, 0]
    #         trace[:, 0] = self.spatial_quant_size * trace[:, 0].clamp_(0, width-1) / width
    #         trace[:, 1] = self.spatial_quant_size * trace[:, 1].clamp_(0, height-1) / height
    #         trace = trace.int()

    #         trace_temp = trace.float()
    #         # remove (almost) static points
    #         trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 0.3]], dim=0)
    #         # remove invisible points
    #         trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
    #         if trace_temp.size(0) <= item['step_to_predict'].item() // 2:
    #             continue
    #         # calulate motion speed
    #         if trace_temp.size(0) < step_to_predict:
    #             trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
    #         elif trace_temp.size(0) > step_to_predict:
    #             trace_temp = trace_temp[:step_to_predict]         
    #         pos_traces_to_mark_filtered[key] = trace_temp.int()

    #     mark_trace_history = ''
    #     mark_trace_future = ''
    #     steps_future = ''   

    #     future_steps_random = torch.randint(1, step_to_predict, (1,)).item()
    #     valid_marks = {}
    #     for key, trace in pos_traces_to_mark_filtered.items():
    #         # random select a frame position but not the last frame
    #         # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()

    #         trace_history = trace[:frame_pos]
    #         val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #         mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
    #         valid_marks[key] = trace[frame_pos]

    #         trace_future = trace[frame_pos+1:frame_pos+1+future_steps_random]
    #         val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_future.tolist()]) + ']'
    #         mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
    #         steps_future += f'Mark {key} in the next {trace_future.size(0)} steps\n'
        
    #     if len(mark_trace_history) > 0 and len(mark_trace_future) > 0:
    #         """
    #         visualize visual traces for debugging
    #         """
    #         # traces_to_vis = torch.stack([val for key, val in valid_traces.items()], 1).unsqueeze(0)
    #         # images = [image] * traces_to_vis.size(1)
    #         # trace_visibility = traces_to_vis.new(traces_to_vis.shape[0], traces_to_vis.shape[1], traces_to_vis.shape[2]).fill_(True)
    #         # video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #         # self.trace.visualize(video, traces_to_vis, trace_visibility)
    #         # speed = round(sum(speeds) / len(speeds))
    #         # mark_ids = sorted([key for key in valid_marks.keys()] + [key for key in neg_traces_to_mark.keys()])            
    #         conv_user = (
    #             f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with a number of numeric marks {mark_ids}.\n"
    #             # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #             f"Given what the person is doing next, please further tell me which marks will move and their {future_steps_random} future positions\n"
    #         )
    #         conversations.append({'from': 'human', 'value': conv_user})
    #         formmated_val = ', '.join([f"mark {key} at location [{val[0].item()},{val[1].item()}]" for key, val in valid_marks.items()])       
    #         conv_gpt = f"{formmated_val} will move, and their maximally {future_steps_random} future positions are:\n\n{mark_trace_future}"
    #         conversations.append({'from': 'gpt', 'value': conv_gpt})
    #     item['conversations'] = conversations
    #     return item

    # def _construct_epic3(self, item, image, frame_pos):
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
        
    #     pos_traces_to_mark = item['pos_traces_to_mark']
    #     neg_traces_to_mark = item['neg_traces_to_mark']
    #     mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])

    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(task_description) > 0:       
    #         conv_user = f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please briefly describe what you are seeing and what the person should do next.\n'     
    #     else:
    #         conv_user = f'<image>\nYou are an expert on world physics. Please briefly describe what you are seeing in the image and what happen next.\n'     
    #     # conv_user = f'<image>\nI have labeled a few numeric marks on the image. Please briefly describe what you are seeing in the image.\n'
    #     conv_gpt = gpt_response + '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model to generate a response to the given task description - language action for the future
    #     # if len(task_description) > 0:            
    #     #     conv_user = f'\nNow the task is {task_description}. Tell me what I should do next.\n'           
    #     # else:        
    #     #     conv_user = f'\nTell me what happen next.\n'
    #     # conv_gpt = gpt_response.split('\n\n')[1] + '\n'
    #     # conversations.append({'from': 'human', 'value': conv_user})
    #     # conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
    #     step_to_predict = item['step_to_predict'].item()
    #     speeds = []
    #     pos_mark_ids = []
     
    #     pos_traces_to_mark_filtered = {}
    #     for key, val in pos_traces_to_mark.items():
    #         # perform exponential moving smoothing for val
    #         trace = val[0, 0]
    #         trace[:, 0] = self.spatial_quant_size * trace[:, 0].clamp_(0, width-1) / width
    #         trace[:, 1] = self.spatial_quant_size * trace[:, 1].clamp_(0, height-1) / height
    #         trace = trace.int()

    #         trace_temp = trace.float()
    #         # remove (almost) static points
    #         trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 0.3]], dim=0)
    #         # remove invisible points
    #         trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
    #         if trace_temp.size(0) <= item['step_to_predict'].item() // 2:
    #             continue
    #         # calulate motion speed
    #         if trace_temp.size(0) < step_to_predict:
    #             trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
    #         elif trace_temp.size(0) > step_to_predict:
    #             trace_temp = trace_temp[:step_to_predict]         
    #         # calculate the delta of the trace
    #         trace_temp = torch.cat([trace_temp[:1], trace_temp[1:] - trace_temp[:-1]], dim=0)
    #         pos_traces_to_mark_filtered[key] = trace_temp.int()

    #     mark_trace_history = ''
    #     mark_trace_future = ''
    #     steps_future = ''   

    #     future_steps_random = torch.randint(1, step_to_predict, (1,)).item()
    #     valid_marks = {}
    #     for key, trace in pos_traces_to_mark_filtered.items():
    #         # random select a frame position but not the last frame
    #         # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()

    #         trace_history = trace[:frame_pos]
    #         val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #         mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
    #         valid_marks[key] = trace[frame_pos]

    #         trace_future = trace[frame_pos+1:frame_pos+1+future_steps_random]
    #         val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_future.tolist()]) + ']'
    #         mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
    #         steps_future += f'Mark {key} in the next {trace_future.size(0)} steps\n'
        
    #     if len(mark_trace_history) > 0 and len(mark_trace_future) > 0:
    #         """
    #         visualize visual traces for debugging
    #         """
    #         # traces_to_vis = torch.stack([val for key, val in valid_traces.items()], 1).unsqueeze(0)
    #         # images = [image] * traces_to_vis.size(1)
    #         # trace_visibility = traces_to_vis.new(traces_to_vis.shape[0], traces_to_vis.shape[1], traces_to_vis.shape[2]).fill_(True)
    #         # video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #         # self.trace.visualize(video, traces_to_vis, trace_visibility)
    #         # speed = round(sum(speeds) / len(speeds))
    #         # mark_ids = sorted([key for key in valid_marks.keys()] + [key for key in neg_traces_to_mark.keys()])            
    #         conv_user = (
    #             f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with a number of numeric marks {mark_ids}.\n"
    #             # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #             f"Given what the person is doing next, please further tell me which marks will move and their {future_steps_random} future relative movements\n"
    #         )
    #         conversations.append({'from': 'human', 'value': conv_user})
    #         formmated_val = ', '.join([f"mark {key} at location [{val[0].item()},{val[1].item()}]" for key, val in valid_marks.items()])       
    #         conv_gpt = f"{formmated_val} will move, and their {future_steps_random} future relative movements are:\n\n{mark_trace_future}"
    #         conversations.append({'from': 'gpt', 'value': conv_gpt})
    #     item['conversations'] = conversations
    #     return item

    # def _construct_epic4(self, item, image, frame_pos, visual_traces=None):
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

    #     # step_to_predict = item['step_to_predict'].item()
    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(task_description) > 0:       
    #         conv_user = f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please look at the image and briefly describe what you are seeing and what the person should do next specifically.\n'     
    #     else:
    #         conv_user = f'<image>\nYou are an expert on world physics. Please look at the image and briefly describe what you are seeing in the image and what happen next.\n'     
    #     # conv_user = f'<image>\nI have labeled a few numeric marks on the image. Please briefly describe what you are seeing in the image.\n'
    #     conv_gpt = gpt_response + '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})
    #     if visual_traces is None:
    #         item['conversations'] = conversations
    #         return item
            
    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
    #     pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]

    #     step_to_predict = pred_tracks.size(1)

    #     pred_tracks_history = visual_traces['pred_tracks'][:, :max(1, frame_pos)]
    #     pred_visibility_history = visual_traces['pred_visibility'][:, :max(1, frame_pos)]

    #     # only keep points that are visible at at least half steps
    #     valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]

    #     if valid_idx.sum() == 1:
    #         item['conversations'] = conversations
    #         return item
        
    #     pred_tracks = pred_tracks[:, :, valid_idx]
    #     pred_visibility = pred_visibility[:, :, valid_idx]

    #     pred_tracks_history = pred_tracks_history[:, :, valid_idx]
    #     pred_visibility_history = pred_visibility_history[:, :, valid_idx]

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

    #             if pred_tracks_history.size(1) > 0:
    #                 history_pts_transformed = []
    #                 for k in range(0, pred_tracks_history.shape[1]):
    #                     history_pts = pred_tracks_history[:, k][0]
    #                     history_pts_np = history_pts.cpu().numpy().reshape(-1, 2)
    #                     try:
    #                         (H, status) = cv2.findHomography(history_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
    #                     except Exception as e:
    #                         continue
    #                     history_pts_np_transformed = cv2.perspectiveTransform(history_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
    #                     history_pts_transformed_k = torch.tensor(history_pts_np_transformed, dtype=torch.float32)
    #                     history_pts_transformed.append(history_pts_transformed_k)            
    #                 pred_tracks_history = torch.stack(history_pts_transformed, dim=0).unsqueeze(0)   

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
    #         pos_tracks_history = pred_tracks_history[:, :, track_length > threshold]
    #         pos_visibility_history = pred_visibility_history[:, :, track_length > threshold]

    #         neg_tracks = pred_tracks[:, :, track_length <= threshold]
    #         neg_tracks_history = pred_tracks_history[:, :, track_length <= threshold]

    #         # clustering for positive traces
    #         pos_sampled_ids = self.trace.cluster_traces(pos_tracks, n_clusters=self.num_clusters)
    #         if pos_sampled_ids is None:
    #             item['conversations'] = conversations
    #             return item                
    #         pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
    #         pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]
    #         pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
    #         pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

    #         # clustering for negative traces
    #         neg_sampled_ids = self.trace.cluster_traces(neg_tracks, n_clusters=2*self.num_clusters)
    #         if neg_sampled_ids is None:
    #             item['conversations'] = conversations
    #             return item                
    #         neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]
    #         neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

    #         image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)
    #         pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
    #             som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)

    #         # visualize the traces
    #         # images = [image] * pos_tracks.shape[1]
    #         # video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #         # self.trace.visualizer.save_dir = "./videos"
    #         # res_video = self.trace.visualize(video, pos_tracks, pos_visibility, filename="visual_trace_to_predict", mode="rainbow")

    #         mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])
    #         pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
        
    #         mark_trace_history = ''
    #         mark_trace_future = ''

    #         valid_marks = {}
    #         speeds = {}
    #         for key, val in pos_traces_to_mark.items():
    #             # random select a frame position but not the last frame
    #             # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
    #             trace = val[0]
    #             trace[:, 0] = self.spatial_quant_size * trace[:, 0] / width
    #             trace[:, 1] = self.spatial_quant_size * trace[:, 1] / height

    #             trace_temp = trace.clone()
    #             # remove (almost) static points
    #             trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 1]], dim=0)
    #             # remove invisible points
    #             trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
    #             if trace_temp.size(0) <= step_to_predict // 4:
    #                 continue
    #             # calulate motion speed
    #             if trace_temp.size(0) < step_to_predict:
    #                 trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
    #             elif trace_temp.size(0) > step_to_predict:
    #                 trace_temp = trace_temp[:step_to_predict]   

    #             # calcualte speed
    #             speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean().item()
    #             speeds[key] = speed
    #             if speed < 1.0:
    #                 continue                
    #             # trace_history = trace[0]
    #             # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #             # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
    #             # round trace_temp
    #             valid_marks[key] = trace_temp[0].int()

    #             val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_temp[1:].int().tolist()]) + ']'
    #             mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
            
    #         if len(mark_trace_future) > 0:
    #             """
    #             visualize visual traces for debugging
    #             """          
    #             conv_user = (
    #                 f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with a number of numeric marks {mark_ids}.\n"
    #                 # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #                 f"Given then specific task what the person is doing next, please further tell me which marks will move and their {step_to_predict-1} future movements\n"
    #             )
    #             conversations.append({'from': 'human', 'value': conv_user})
    #             formmated_val = ', '.join([f"mark {key} at [{val[0].item()},{val[1].item()}]" for key, val in valid_marks.items()])       
    #             conv_gpt = f"{formmated_val} will move, and their future movements are:\n\n{mark_trace_future}"
    #             conversations.append({'from': 'gpt', 'value': conv_gpt})
    #         item['conversations'] = conversations
    #         return item

    # def _construct_epic(self, item, image, frame_pos):
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
    #     gpt_response = gpt_response.replace('What the person should do next', 'What you should do next')
    #     gpt_response = gpt_response.replace('What you should do next', 'What you are doing')
    #     gpt_response = gpt_response.replace('personr', 'person\'s')
    #     # remove all str (marks) from the gpt_response
    #     gpt_response = re.sub(r' \([^)]*\)', '', gpt_response)
        
    #     pos_traces_to_mark = item['pos_traces_to_mark']
    #     neg_traces_to_mark = item['neg_traces_to_mark']
    #     mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])

    #     conversations = []
    #     # model task 1: ask model to briefly describe the current image - understand the present
    #     if len(task_description) > 0:       
    #         conv_user = f'<image>\nYou are an expert coach. I am trying to do the task of \"{task_description}\". I have labeled a number of numeric marks {mark_ids} on the image. Please briefly describe what you are seeing and what I should do.\n'     
    #     else:
    #         conv_user = f'<image>\nPlease briefly describe what you are seeing in the image and what happen next.\n'     
    #     # conv_user = f'<image>\nI have labeled a few numeric marks on the image. Please briefly describe what you are seeing in the image.\n'
    #     conv_gpt = gpt_response + '\n'
    #     conversations.append({'from': 'human', 'value': conv_user})
    #     conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model to generate a response to the given task description - language action for the future
    #     # if len(task_description) > 0:            
    #     #     conv_user = f'\nNow the task is {task_description}. Tell me what I should do next.\n'           
    #     # else:        
    #     #     conv_user = f'\nTell me what happen next.\n'
    #     # conv_gpt = gpt_response.split('\n\n')[1] + '\n'
    #     # conversations.append({'from': 'human', 'value': conv_user})
    #     # conversations.append({'from': 'gpt', 'value': conv_gpt})

    #     # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
    #     # sort pos_traces_to_mark by the key
    #     pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
    #     step_to_predict = item['step_to_predict'].item()
    #     speeds = []
    #     pos_mark_ids = []
     
    #     pos_traces_to_mark_filtered = {}
    #     for key, val in pos_traces_to_mark.items():
    #         # perform exponential moving smoothing for val
    #         trace = val[0, 0]
    #         trace[:, 0] = self.spatial_quant_size * trace[:, 0].clamp_(0, width-1) / width
    #         trace[:, 1] = self.spatial_quant_size * trace[:, 1].clamp_(0, height-1) / height
    #         trace = trace.int()

    #         trace_temp = trace.float()
    #         # remove (almost) static points
    #         trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 0.3]], dim=0)
    #         # remove invisible points
    #         trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
    #         if trace_temp.size(0) <= item['step_to_predict'].item() // 2:
    #             continue
    #         # calulate motion speed
    #         # speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean().item()
    #         # speeds.append(speed)
    #         pos_traces_to_mark_filtered[key] = trace
    #         # if trace_temp.size(0) < step_to_predict:
    #         #     trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
    #         # elif trace_temp.size(0) > step_to_predict:
    #         #     trace_temp = trace_temp[:step_to_predict]         

    #     mark_trace_history = ''
    #     mark_trace_future = ''
    #     steps_future = ''   

    #     for key, trace in pos_traces_to_mark_filtered.items():
    #         # random select a frame position but not the last frame
    #         # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()

    #         trace_history = trace[:frame_pos+1]
    #         val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
    #         mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    

    #         trace_future = trace[frame_pos+1:]
    #         val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_future.tolist()]) + ']'
    #         mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n'      
    #         steps_future += f'Mark {key} in the next {trace_future.size(0)} steps\n'
        
    #     if len(mark_trace_history) > 0 and len(mark_trace_future) > 0:
    #         """
    #         visualize visual traces for debugging
    #         """
    #         # traces_to_vis = torch.stack([val for key, val in valid_traces.items()], 1).unsqueeze(0)
    #         # images = [image] * traces_to_vis.size(1)
    #         # trace_visibility = traces_to_vis.new(traces_to_vis.shape[0], traces_to_vis.shape[1], traces_to_vis.shape[2]).fill_(True)
    #         # video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    #         # self.trace.visualize(video, traces_to_vis, trace_visibility)
    #         # speed = round(sum(speeds) / len(speeds))
    #         # mark_ids = sorted([key for key in valid_marks.keys()] + [key for key in neg_traces_to_mark.keys()])            
    #         conv_user = (
    #             f"In the image with {self.spatial_quant_size}x{self.spatial_quant_size} grids. "
    #             # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
    #             f"Please tell me which marks should move and the {trace.shape[0]-frame_pos-1} future positions\n"
    #         )
    #         conversations.append({'from': 'human', 'value': conv_user})
    #         formmated_val = ', '.join([f"mark {key} at location [{val[0].item()},{val[1].item()}]" for key, val in valid_marks.items()])            
    #         conv_gpt = f"Their future positions for the target marks are:\n\n{mark_trace_future}\n"
    #         conversations.append({'from': 'gpt', 'value': conv_gpt})
    #     item['conversations'] = conversations
    #     return item