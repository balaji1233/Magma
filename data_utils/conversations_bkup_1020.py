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
import torchvision.io as tv_io
import torchvision
import time
from decord import VideoReader, cpu

# import av

class Constructor():
    def __init__(self, **kwargs):
        self.trace = visual_trace()
        self.mm_use_trace_start_end = kwargs.get('mm_use_trace_start_end', False)
        self.mm_use_trace_speed = kwargs.get('mm_use_trace_speed', False)
        self.mm_use_image_start_end = kwargs.get('mm_use_image_start_end', False)
        self.remove_static_trace_pts = kwargs.get('remove_static_trace_pts', False)
        self.show_trace = kwargs.get('show_trace', False)
        self.video_reader = kwargs.get('video_reader', 'decord')

    def _get_frame(self, video_path, frame_start, frame_pos, size):   
        if self.video_reader == 'cv2':
            video_cap = cv2.VideoCapture(video_path)
            num_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_start + frame_pos >= num_frames:
                frame_pos = 0
            trials = 0
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_pos)
            while trials < 5:
                success, image = video_cap.read()
                if success:
                    break
                else:
                    time.sleep(0.1)
                    trials += 1
            if not success:
                print(f"Failed to read video {video_path} at frame {frame_start + frame_pos}")
                image = None      
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image).resize(size)  
            video_cap.release()
            return image 
        elif self.video_reader == 'decord':
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                num_frames = len(vr)
                if frame_start+frame_pos >= num_frames:
                    frame_pos = 0            
                frame_idx = [frame_start+frame_pos]
                image = vr.get_batch(frame_idx).asnumpy()[0]
                # https://github.com/dmlc/decord/issues/208
                vr.seek(0)
                # convert image to rgb format                
                image = Image.fromarray(image).resize(size)                                               
                return image     
            except Exception as e:
                print(f"Failed to read video {video_path} at frame {frame_start + frame_pos}")
                return None    
    def _process_gpt_response(self, gpt_response):
        """
        Process the gpt_response
        """
        gpt_response = gpt_response.replace('What you see', 'What I see')
        gpt_response = gpt_response.replace('you see ', '').replace('You see ', '')
        gpt_response = gpt_response.replace('you', 'the person')
        gpt_response = gpt_response.replace('your', '')
        gpt_response = gpt_response.replace('In the first image, ', '')
        # gpt_response = gpt_response.replace('What the person should do next', 'What you should do next')
        # gpt_response = gpt_response.replace('What you should do next', 'What you are doing')
        gpt_response = gpt_response.replace('personr', 'person\'s')
        # remove all str (marks) from the gpt_response
        gpt_response = re.sub(r' \([^)]*\)', '', gpt_response)
        gpt_response = gpt_response if len(gpt_response) > 0 else task_description
        gpt_response = gpt_response.replace('camera wearer', 'person')
        return gpt_response

    def _construct_conv_semantic(self, item, gpt_response):
        """
        Construct conversations for semantic (language) prediction
        """
        # model task 1: ask model to briefly describe the current image - understand the present
        if item['dataset_tag'] == 'ego4d':
            conv_user = (
                f'<image>\nWhat is the person doing.\n'
            )                 
            conv_gpt = gpt_response + '\n'
            
            gpt_response_todo = gpt_response

        elif item['dataset_tag'] == 'howto100m':
            # for howto100m, it is narration
            conv_user = (
                f'<image>\nWhat is the person saying?\n'
            )                 
            conv_gpt = gpt_response + '\n'

            gpt_response_todo = gpt_response

        elif item['dataset_tag'] in ['coin', 'epic']:
            gpt_response_see = gpt_response.split('What the person should do next')[0].replace('#','').replace('*','').replace('What I see:', '').strip()
            conv_user = (
                f'<image>\nWhat do you see in the image?\n'
            )
            conv_gpt = gpt_response_see + '\n'
            gpt_response_todo = gpt_response.split('What the person should do next')[1].replace('#','').replace('*', '').replace(':','').strip()

        return conv_user, conv_gpt, gpt_response_todo

    
    def _construct_conv_som(self, item, image, visual_traces):
        """
        Construct conversations for spatial prediction
        """
        image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)


    def _construct_conv_tom(self, item, video_path, visual_traces):
        """
        Construct conversations for spatial-temporal prediction
        """

    def _construct_conv(self, item, video_path, visual_traces):
        """
        v4->v5: add trace of mark
        """        
        if video_path is None and visual_traces is None:
            dummy_conversations = []
            dummy_conversations.append({'from': 'human', 'value': "<image>\nWhat is in this image?"})
            dummy_conversations.append({'from': 'gpt', 'value': "This is a blank image."})            
            item['conversations'] = dummy_conversations
            item['image'] = None  
            return item
        
        if 'image_size' not in item:
            assert '(height,width)' in item, f"image_size not in item and (height,width) not in item"
            item['image_size'] = item['(height,width)'][::-1]            
        
        if isinstance(item['image_size'][0], torch.Tensor):
            width, height = item['image_size'][0].item(), item['image_size'][1].item()
            frame_start, frame_end = item['frame_interval'][0].item(), item['frame_interval'][1].item()
            task_description = item['global_instructions'][0]
            gpt_response = item['gpt_response'][0]
        else:
            width, height = item['image_size']
            frame_start, frame_end = item['frame_interval']     
            task_description = item['global_instructions']
            gpt_response = item['gpt_response']
        
        gpt_response = self._process_gpt_response(gpt_response)
        # step_to_predict = item['step_to_predict'].item()
        conversations = []
        conv_user, conv_gpt, gpt_response_todo = self._construct_conv_semantic(item, gpt_response)        
        conversations.append({'from': 'human', 'value': conv_user})
        conversations.append({'from': 'gpt', 'value': conv_gpt})

        if visual_traces is None:
            item['conversations'] = conversations
            item['image'] = self._get_frame(video_path, frame_start, 0, (width, height))
            return item
        
        if len(visual_traces['pred_tracks'].shape) == 3:
            visual_traces['pred_tracks'] = visual_traces['pred_tracks'][None]
        if len(visual_traces['pred_visibility'].shape) == 2:
            visual_traces['pred_visibility'] = visual_traces['pred_visibility'][None]
 
        # model task 2: ask model the predict the movements of the person and/or the object - visual action for the future
        # sort pos_traces_to_mark by the key
        # calculate the trace length for each step
        track_length = torch.norm(visual_traces['pred_tracks'][:, 1:] - visual_traces['pred_tracks'][:, :-1], dim=3).mean(2)
        # accum_sum track_length
        accum_sum = torch.cumsum(track_length, dim=1) / (1e-5 + track_length.sum(1)[:, None])
        
        # find last position
        frame_rightmost = min(max(1, (accum_sum[0] < self.settings['trace_planner']['step_rightmost_ratio']).int().sum().item()), visual_traces['pred_tracks'].shape[1]-1)
        # random select a frame position but not the last frame
        frame_pos = torch.randint(0, frame_rightmost, (1,)).item()
        
        pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
        pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]
        step_to_predict = pred_tracks.size(1)

        if step_to_predict == 0:            
            item['conversations'] = conversations
            item['image'] = self._get_frame(video_path, frame_start, 0, (width, height))    
            return item
        
        pred_tracks_history = visual_traces['pred_tracks'][:, :max(1, frame_pos+1)]
        pred_visibility_history = visual_traces['pred_visibility'][:, :max(1, frame_pos+1)]

        # only keep points that are visible at at least half steps
        valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]

        if valid_idx.sum() <= 1:
            image = self._get_frame(video_path, frame_start, 0, (width, height))            
            conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces)
            conversations.append({'from': 'human', 'value': conv_user})
            conversations.append({'from': 'gpt', 'value': conv_gpt})
            item['conversations'] = conversations
            item['image'] = image
            return item

        pred_tracks = pred_tracks[:, :, valid_idx]
        pred_visibility = pred_visibility[:, :, valid_idx]
        pred_tracks_history = pred_tracks_history[:, :, valid_idx]
        pred_visibility_history = pred_visibility_history[:, :, valid_idx]
        
        # calculate the trajectory lenght for pred_tracks
        pred_tracks_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (1, 1)).squeeze(0)
        # if 80% of the pred_tracks_length is larger than 2, then there is camera motion
        camera_motion = (pred_tracks_length > 1).sum() > 0.8*pred_tracks_length.size(0)
        camera_motion = True if item['dataset_tag'] in ['ego4d', 'epic'] else camera_motion
        
        start_pos = pred_tracks[:, 0][0]
        reference_pts_np = start_pos.cpu().numpy().reshape(-1, 2)

        # end_pos = pred_tracks[:, -1][0]
        # future_pts_np = end_pos.cpu().numpy().reshape(-1, 2)
        # try:
        #     (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
        # except Exception as e:
        #     H = None
        
        # if H is not None:
        #     camera_motion = False
        #     H = torch.tensor(H, dtype=torch.float32)
        #     rt_magnitude = torch.norm(H[:2, :2] - torch.eye(2)).item()
        #     scale_magnitude = torch.norm(H[:2, 2] / torch.Tensor([width, height])).item()
        #     distortion_magnitude = torch.norm(H[2, :2]).item()
        #     if rt_magnitude > 0.1 or scale_magnitude > 0.1 or distortion_magnitude > 0.1:
        #         camera_motion = True
            
        if camera_motion:
            # remove camera motion using homography transformation
            try:
                future_pts_transformed = []
                for k in range(1, pred_tracks.shape[1]):
                    future_pts = pred_tracks[:, k][0]
                    future_pts_np = future_pts.cpu().numpy().reshape(-1, 2)
                    try:
                        (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
                    except Exception as e:
                        continue
                    future_pts_np_transformed = cv2.perspectiveTransform(future_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
                    future_pts_transformed_k = torch.tensor(future_pts_np_transformed, dtype=torch.float32)
                    future_pts_transformed.append(future_pts_transformed_k)            
                pred_tracks = torch.stack([start_pos] + future_pts_transformed, dim=0).unsqueeze(0)           
            except Exception as e:
                pass
            
            if pred_tracks_history.size(1) > 0:
                try:
                    history_pts_transformed = []
                    for k in range(0, pred_tracks_history.shape[1]):
                        history_pts = pred_tracks_history[:, k][0]
                        history_pts_np = history_pts.cpu().numpy().reshape(-1, 2)
                        try:
                            (H, status) = cv2.findHomography(history_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
                        except Exception as e:
                            continue
                        history_pts_np_transformed = cv2.perspectiveTransform(history_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
                        history_pts_transformed_k = torch.tensor(history_pts_np_transformed, dtype=torch.float32)
                        history_pts_transformed.append(history_pts_transformed_k)            
                    pred_tracks_history = torch.stack(history_pts_transformed, dim=0).unsqueeze(0)   
                except Exception as e:
                    pass
        
        # step 2: find positive traces and negative traces
        track_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (1, 1)).squeeze(0)
        threshold = 2 # max(track_length.max(), 2) * self.settings['trace_processor']['postive_factor_threshold']
        # video is almost static
        if (track_length > threshold).sum() <= 1:
            image = self._get_frame(video_path, frame_start, 0, (width, height))            
            conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces)
            conversations.append({'from': 'human', 'value': conv_user})
            conversations.append({'from': 'gpt', 'value': conv_gpt})                  
            item['conversations'] = conversations
            item['image'] = image
            return item
        else:
            # find the positive traces and negative traces
            pos_tracks = pred_tracks[:, :, track_length > threshold]
            pos_visibility = pred_visibility[:, :, track_length > threshold]
            pos_tracks_history = pred_tracks_history[:, :, track_length > threshold]
            pos_visibility_history = pred_visibility_history[:, :, track_length > threshold]

            neg_tracks = pred_tracks[:, :, track_length <= threshold]
            neg_tracks_history = pred_tracks_history[:, :, track_length <= threshold]

            # clustering for positive traces
            pos_sampled_ids = self.trace.cluster_traces(pos_tracks, n_clusters=self.num_clusters)
            if pos_sampled_ids is None:
                item['conversations'] = conversations
                item['image'] = self._get_frame(video_path, frame_start, 0, (width, height))     
                return item                
            pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
            pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]
            pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
            pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

            # clustering for negative traces
            neg_sampled_ids = self.trace.cluster_traces(neg_tracks, n_clusters=2*self.num_clusters)
            if neg_sampled_ids is None:
                item['conversations'] = conversations
                item['image'] = self._get_frame(video_path, frame_start, 0, (width, height))
                return item    
                        
            neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]
            neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

            image = self._get_frame(video_path, frame_start, frame_pos, (width, height))
            if image is None:
                item['conversations'] = conversations
                item['image'] = image
                return item
            
            mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])
            pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
        
            mark_trace_history = ''
            mark_trace_future = ''

            valid_marks = {}
            speeds = {}
            for key, val in pos_traces_to_mark.items():
                # random select a frame position but not the last frame
                # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
                trace = val[0]
                trace[:, 0] = self.spatial_quant_size * trace[:, 0] / width
                trace[:, 1] = self.spatial_quant_size * trace[:, 1] / height

                trace_temp = trace.clone()
                # remove (almost) static points
                trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 1]], dim=0)
                # remove invisible points
                trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
                if trace_temp.size(0) <= step_to_predict // 4:
                    continue
                # calulate motion speed
                # if trace_temp.size(0) < step_to_predict:
                #     trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
                # elif trace_temp.size(0) > step_to_predict:
                #     trace_temp = trace_temp[:step_to_predict]   

                # calcualte speed
                speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean()
                if torch.isnan(speed):
                    continue

                speeds[key] = speed.item()

                
                if speed < self.settings['trace_processor']['postive_speed_threshold']:
                    continue                
                # trace_history = trace[0]
                # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
                # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
                # round trace_temp
                if self.remove_static_trace_pts:
                    valid_marks[key] = trace_temp.int()
                else:
                    valid_marks[key] = trace.int()

                # NOTE: there was a bug here
                val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in valid_marks[key][1:].tolist()]) + ']'

                mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
            
            if len(mark_trace_future) > 0:
                """
                visualize visual traces for debugging
                """
                image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)
                image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
                    som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)

                # visualize the traces
                if self.show_trace:
                    images = [image] * pos_tracks.shape[1]
                    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
                    self.trace.visualizer.save_dir = "./videos"
                    _ = self.trace.visualize(video, pos_tracks, pos_visibility, filename="visual_trace_to_predict", mode="rainbow")

                # 0923-1am
                if item['dataset_tag'] != 'howto100m':
                    conv_user = (
                        f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks.\n"
                        f"The person is doing: {gpt_response_todo}. how to move the marks in the next {step_to_predict-1} steps?\n"
                    )
                else:
                    conv_user = (
                        f"The image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks.\n"
                        f"The person is saying: {gpt_response_todo}. how to move the marks in the next {step_to_predict-1} steps?\n"
                    )
                conversations.append({'from': 'human', 'value': conv_user})
                if self.mm_use_trace_speed:
                    # calculate speed
                    formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move {val.shape[0]-1} steps with speed {round(speeds[key])}" for key, val in valid_marks.items()])       
                else:
                    formmated_val = ', '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}]" for key, val in valid_marks.items()])       
                if self.mm_use_trace_start_end:
                    mark_trace_future = f'<trace_start>{mark_trace_future}<trace_end>'                      
                conv_gpt = f"{formmated_val} should be moved, and their future positions are:\n\n{mark_trace_future}"
                conversations.append({'from': 'gpt', 'value': conv_gpt})          
            else:
                conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces)
                conversations.append({'from': 'human', 'value': conv_user})
                conversations.append({'from': 'gpt', 'value': conv_gpt})     

            item['conversations'] = conversations
            item['image'] = image
            return item