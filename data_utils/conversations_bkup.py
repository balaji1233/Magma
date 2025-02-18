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
import torchvision.io as io
import torchvision
import time
import av

class Constructor():
    def __init__(self, **kwargs):
        self.trace = visual_trace()
        self.mm_use_trace_start_end = kwargs.get('mm_use_trace_start_end', False)
        self.mm_use_trace_speed = kwargs.get('mm_use_trace_speed', False)
        self.mm_use_image_start_end = kwargs.get('mm_use_image_start_end', False)
        self.remove_static_trace_pts = kwargs.get('remove_static_trace_pts', False)
        self.show_trace = kwargs.get('show_trace', False)

    def _get_frame(self, video_path, video, frame_pos, size):   
        return self._get_frame_av(video_path, video, frame_pos, size)     
        if isinstance(video, torch.Tensor):
            image = video[frame_pos].permute(2, 0, 1)
            # convert image tensor tp PIL image
            image = torchvision.transforms.ToPILImage()(image).resize(size)
            return image
        else:
            trials = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            while trials < 5:
                success, image = video.read()
                if success:
                    break
                else:
                    time.sleep(0.1)
                    trials += 1
            if not success:
                print(f"Failed to read video {video_path} at frame {frame_pos}")
                return None               
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image).resize(size)  
                return image 

    def _get_frame_av(self, video_path, container, frame_pos, size):   
        # Seek to the specific frame (frame_number starts at 0)
        import pdb; pdb.set_trace()
        container.seek(frame_pos, any_frame=True, backward=True)
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format='rgb24')
            image = Image.fromarray(image).resize(size)  
            return image         
        print(f"Failed to read video {video_path} at frame {frame_pos}")
        return None  # If the frame is not found

    def _get_frame_tv(self, video_path, reader, frame_pos, size):
        # Step 1: Get the list of timestamps for each frame and the frame rate (fps)
        pts, fps = io.read_video_timestamps(video_path, pts_unit='sec')

        # Step 2: Check if the frame_id is within the valid range
        if frame_pos < 0 or frame_pos >= len(pts):
            print(f"Frame {frame_pos} is out of range. The video has {len(pts)} frames.")
            return None

        # Step 3: Get the timestamp corresponding to the frame_id
        frame_timestamp = pts[frame_pos]

        # Step 4: Use read_video to extract the frame at the specific timestamp
        frame, _, _ = io.read_video(video_path, start_pts=frame_timestamp, end_pts=frame_timestamp + 0.001, pts_unit='sec')

        if len(frame) > 0:
            # Convert frame tensor to a PIL image (in RGB format)            
            frame_array = frame[0].numpy()
            # Optional: Convert from RGB to BGR for OpenCV display
            image = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_array).resize(size)  
            return image
        else:
            print(f"Failed to read video {video_path} at frame {frame_pos}")
            return None

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
        
        tic = time.time()
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

        # step_to_predict = item['step_to_predict'].item()
        conversations = []
        # model task 1: ask model to briefly describe the current image - understand the present
        if item['dataset_tag'] != 'howto100m':
            if self.mm_use_image_start_end:
                conv_user = (
                    f'<image_start><image><image_end>\nYou are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the moving traces for all marks thus far on the image. Tell me what you are seeing and what specific task the person is doing.\n'
                )       
            else:
                conv_user = (
                    f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the moving traces for all marks thus far on the image. Tell me what you are seeing and what specific task the person is doing.\n'
                )                 
            conv_gpt = gpt_response + '\n'
        else:
            if self.mm_use_image_start_end:
                conv_user = (
                    f'<image_start><image><image_end>\nYou are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the moving traces for all marks thus far on the image. Guess what the person is saying.\n'
                )       
            else:
                conv_user = (
                    f'<image>\nYou are an expert on daily tasks. The person is doing a specific task. Please carefully look at the image and probably the moving traces for all marks thus far on the image. Guess what the person is saying.\n'
                )                 
            conv_gpt = gpt_response + '\n'
        conversations.append({'from': 'human', 'value': conv_user})
        conversations.append({'from': 'gpt', 'value': conv_gpt})

        # read video
        # video_path_in_frames = video_path.replace('all_detected_split_video_segments_30fps_v2', 'extracted_frames_video_segments_30fps_v2').replace('.mp4', '.pth')
        # if os.path.exists(video_path_in_frames):
        #     try:
        #         video_cap = torch.load(video_path_in_frames, map_location='cpu') 
        #         num_frames = video_cap.shape[0]  
        #     except:
        #         video_cap = cv2.VideoCapture(video_path)
        #         video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # Set an appropriate timeout value
        #         video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        #         num_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)                
        # else:
        # video_cap = cv2.VideoCapture(video_path)
        # video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # Set an appropriate timeout value
        # video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # num_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # video_cap = av.open(video_path)
        # video_stream = video_cap.streams.video[0]
        # num_frames = video_stream.frames
        video_cap = None

        if visual_traces is None:
            item['conversations'] = conversations
            item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))
            # if not isinstance(video_cap, torch.Tensor):
            #     video_cap.close()
            return item
        
        if len(visual_traces['pred_tracks'].shape) == 3:
            visual_traces['pred_tracks'] = visual_traces['pred_tracks'][None]
        if len(visual_traces['pred_visibility'].shape) == 2:
            visual_traces['pred_visibility'] = visual_traces['pred_visibility'][None]

        # print(f"Time: {time.time()-tic}")
        tic = time.time()
                
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
        # if frame_start + frame_pos >= num_frames:
        #     frame_pos = 0
        
        pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
        pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]
        step_to_predict = pred_tracks.size(1)

        if step_to_predict == 0:
            item['conversations'] = conversations
            item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))    
            # if not isinstance(video_cap, torch.Tensor):
            #     video_cap.close()
            return item
        
        pred_tracks_history = visual_traces['pred_tracks'][:, :max(1, frame_pos)]
        pred_visibility_history = visual_traces['pred_visibility'][:, :max(1, frame_pos)]

        # only keep points that are visible at at least half steps
        valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]

        if valid_idx.sum() <= 1:
            item['conversations'] = conversations
            item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))
            # if not isinstance(video_cap, torch.Tensor):
            #     video_cap.close()
            return item
        
        pred_tracks = pred_tracks[:, :, valid_idx]
        pred_visibility = pred_visibility[:, :, valid_idx]

        pred_tracks_history = pred_tracks_history[:, :, valid_idx]
        pred_visibility_history = pred_visibility_history[:, :, valid_idx]

        # step 1: determine whether there are camera motions
        start_pos = pred_tracks[:, 0][0]
        end_pos = pred_tracks[:, -1][0]
        reference_pts_np = start_pos.cpu().numpy().reshape(-1, 2)
        future_pts_np = end_pos.cpu().numpy().reshape(-1, 2)
        try:
            (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
        except Exception as e:
            H = None
        
        if H is not None:
            camera_motion = False
            H = torch.tensor(H, dtype=torch.float32)
            rt_magnitude = torch.norm(H[:2, :2] - torch.eye(2)).item()
            scale_magnitude = torch.norm(H[:2, 2] / torch.Tensor([width, height])).item()
            distorsion_magnitude = torch.norm(H[2, :3]).item()
            if rt_magnitude > 0.1 or scale_magnitude > 0.1 or distorsion_magnitude > 0.1:
                camera_motion = True
            
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
        
        # print(f"Time: {time.time()-tic}")
        tic = time.time()
        # step 2: find positive traces and negative traces
        track_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (width, height)).squeeze(0)
        max_length = track_length.max()
        threshold = max_length * self.settings['trace_processor']['postive_factor_threshold']
        # video is almost static
        if (track_length > threshold).sum() <= 1:      
            item['conversations'] = conversations
            item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))  
            # if not isinstance(video_cap, torch.Tensor):
            #     video_cap.close()
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
                item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))     
                # if not isinstance(video_cap, torch.Tensor):
                #     video_cap.close()
                return item                
            pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
            pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]
            pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
            pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

            # clustering for negative traces
            neg_sampled_ids = self.trace.cluster_traces(neg_tracks, n_clusters=2*self.num_clusters)
            if neg_sampled_ids is None:
                item['conversations'] = conversations
                item['image'] = self._get_frame_tv(video_path, video_cap, frame_start, (width, height))
                # if not isinstance(video_cap, torch.Tensor):
                #     video_cap.close()
                return item                
            neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]
            neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

            image = self._get_frame_tv(video_path, video_cap, frame_start + frame_pos, (width, height))
            if image is None:
                item['conversations'] = conversations
                item['image'] = image
                # if not isinstance(video_cap, torch.Tensor):
                #     video_cap.close()
                return item
            
            image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)
            image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
                som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)

            # visualize the traces
            if self.show_trace:
                images = [image] * pos_tracks.shape[1]
                video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
                self.trace.visualizer.save_dir = "./videos"
                _ = self.trace.visualize(video, pos_tracks, pos_visibility, filename="visual_trace_to_predict", mode="rainbow")

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
                # 0923-1am
                conv_user = (
                    # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
                    f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
                    f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please tell me which marks will move and their movements in the next {step_to_predict-1} time window.\n"
                )
                conversations.append({'from': 'human', 'value': conv_user})
                if self.mm_use_trace_speed:
                    # calculate speed
                    formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move {val.shape[0]-1} steps with speed {round(speeds[key])}" for key, val in valid_marks.items()])       
                else:
                    formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move {val.shape[0]-1} steps" for key, val in valid_marks.items()])       
                if self.mm_use_trace_start_end:
                    mark_trace_future = f'<trace_start>{mark_trace_future}<trace_end>'                      
                conv_gpt = f"{formmated_val}. Their future movements are:\n\n{mark_trace_future}"
                conversations.append({'from': 'gpt', 'value': conv_gpt})          

                # 0923-4am
                # conv_user = (
                #     # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
                #     f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
                #     f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please tell me which marks will move and their movements in the next {step_to_predict-1} time window.\n"
                # )
                # conversations.append({'from': 'human', 'value': conv_user})
                # formmated_val = '. '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}] will move" for key, val in valid_marks.items()])   
                # if self.mm_use_trace_start_end:
                #     mark_trace_future = f'<trace_start>{mark_trace_future}<trace_end>'      
                # conv_gpt = f"{formmated_val}. Their future movements are:\n\n{mark_trace_future}"
                # conversations.append({'from': 'gpt', 'value': conv_gpt})      

                # 0922:
                # conv_user = (
                #     # f"The history positions in the last {frame_pos+1} steps thus far for some target marks are:\n\n{mark_trace_history}\n"
                #     f"For your understanding, the image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks {mark_ids}.\n"
                #     f"Given the specific task that the person is doing and moving traces in the last {frame_pos} steps for all marks, please further tell me which marks will move and their {step_to_predict-1} future movements\n"
                # )
            # print(f"Time: {time.time()-tic}")
            item['conversations'] = conversations
            item['image'] = image
            # video_cap.close()
            # import pdb; pdb.set_trace()
            # if not isinstance(video_cap, torch.Tensor):
            #     video_cap.close()
            
            return item