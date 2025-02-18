"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import collections
import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from magma_utils.som import som_prompting, tom_prompting

# from prismatic.models.backbones.llm.prompting import PromptBuilder
# from prismatic.models.backbones.vision import ImageTransform
from ..action_tokenizer import ActionTokenizer
from .rlds import make_interleaved_dataset, make_single_dataset
from .rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from .rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
from typing import Callable, Dict, Sequence, Tuple

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: None # ImageTransform
    prompt_builder_fn: None # Type[PromptBuilder]
    visual_tracker: None
    dataset_settings: None
    data_root_dir: str = "/mnt/vlpdatasets"
    predict_stop_token: bool = True
    trace_folder: str = "open-x-traces-v2"
    image_folder: str = "open-x-images-v2"
    local_run: bool = False

    def _construct_conv_for_vpt(self, image, lang, pred_tracks, pred_visibility, frame_pos=1):
        """
        Construct visual planning training
        """        
        import pdb; pdb.set_trace()
        width, height = image.size
        pred_tracks[:,:,:,0] = pred_tracks[:,:,:,0] * width / 256
        pred_tracks[:,:,:,1] = pred_tracks[:,:,:,1] * height / 256
        # only keep points that are visible at at least half steps
        valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]
        if valid_idx.sum() <= 1:
            return None

        pred_tracks_future = pred_tracks[:, frame_pos:, valid_idx]
        pred_visibility_future = pred_visibility[:, frame_pos:, valid_idx]

        pred_tracks_history = pred_tracks[:, :frame_pos, valid_idx]
        pred_visibility_history = pred_visibility[:, :frame_pos, valid_idx]

        # step 1: determine whether there are camera motions
        start_pos = pred_tracks_future[:, 0][0]
        end_pos = pred_tracks_future[:, -1][0]
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
            scale_magnitude = torch.norm(H[:2, 2] / torch.Tensor([256, 256])).item()
            distortion_magnitude = torch.norm(H[2, :2]).item()
            if rt_magnitude > 0.1 or scale_magnitude > 0.1 or distortion_magnitude > 0.1:
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
        
        # step 2: find positive traces and negative traces
        track_length = self.visual_tracker.visual_trace_length(pred_tracks_future, pred_visibility_future, (256, 256)).squeeze(0)
        max_length = max(track_length.max(), 0.01)
        threshold = max_length * self.dataset_settings['trace_processor']['postive_factor_threshold']
        # video is almost static
        if (track_length > threshold).sum() <= 1:      
            return None
        else:
            # find the positive traces and negative traces
            pos_tracks_future = pred_tracks_future[:, :, track_length > threshold]
            pos_visibility_future = pred_visibility_future[:, :, track_length > threshold]
            pos_tracks_history = pred_tracks_history[:, :, track_length > threshold]
            pos_visibility_history = pred_visibility_history[:, :, track_length > threshold]

            neg_tracks_future = pred_tracks_future[:, :, track_length <= threshold]
            neg_tracks_history = pred_tracks_history[:, :, track_length <= threshold]

            # clustering for positive traces
            pos_sampled_ids = self.visual_tracker.cluster_traces(pos_tracks_future, n_clusters=self.dataset_settings['trace_processor']['num_clusters'])
            if pos_sampled_ids is None:
                return None             
            pos_tracks_future = pos_tracks_future[:, :, pos_sampled_ids.bool()]
            pos_visibility_future = pos_visibility_future[:, :, pos_sampled_ids.bool()]
            pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
            pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

            # clustering for negative traces
            neg_sampled_ids = self.visual_tracker.cluster_traces(neg_tracks_future, n_clusters=2*self.dataset_settings['trace_processor']['num_clusters'])
            if neg_sampled_ids is None:
                return None
                        
            neg_tracks_future = neg_tracks_future[:, :, neg_sampled_ids.bool()]
            neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

            image = tom_prompting(self.visual_tracker, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)
            image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
                som_prompting(image, pos_tracks_future, neg_tracks_future, draw_som_positive=True, draw_som_negative=True)

            import pdb; pdb.set_trace()
            # visualize the traces
            if self.local_run:
                images = [image] * pos_tracks_future.shape[1]
                video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
                self.visual_tracker.visualizer.save_dir = "./videos"
                _ = self.visual_tracker.visualize(video, pos_tracks_future, pos_visibility_future, filename="visual_trace_to_predict", mode="rainbow")

            # mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])
            # pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
        
            # mark_trace_history = ''
            # mark_trace_future = ''

            # valid_marks = {}
            # speeds = {}
            # for key, val in pos_traces_to_mark.items():
            #     # random select a frame position but not the last frame
            #     # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
            #     trace = val[0]
            #     trace[:, 0] = self.dataset_settings['trace_processor']['spatial_quant_size'] * trace[:, 0] / width
            #     trace[:, 1] = self.dataset_settings['trace_processor']['spatial_quant_size'] * trace[:, 1] / height

            #     trace_temp = trace.clone()
            #     # remove (almost) static points
            #     trace_temp = torch.cat([trace_temp[:1], trace_temp[1:][torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1) > 1]], dim=0)
            #     # remove invisible points
            #     trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
            #     if trace_temp.size(0) <= step_to_predict // 4:
            #         continue
            #     # calulate motion speed
            #     # if trace_temp.size(0) < step_to_predict:
            #     #     trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
            #     # elif trace_temp.size(0) > step_to_predict:
            #     #     trace_temp = trace_temp[:step_to_predict]   

            #     # calcualte speed
            #     speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean()
            #     if torch.isnan(speed):
            #         continue

            #     speeds[key] = speed.item()

                
            #     if speed < self.dataset_settings['trace_processor']['postive_speed_threshold']:
            #         continue                
            #     # trace_history = trace[0]
            #     # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
            #     # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
            #     # round trace_temp
            #     if self.remove_static_trace_pts:
            #         valid_marks[key] = trace_temp.int()
            #     else:
            #         valid_marks[key] = trace.int()

            #     # NOTE: there was a bug here
            #     val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in valid_marks[key][1:].tolist()]) + ']'

            #     mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        imgs_future = [Image.fromarray(img) for img in rlds_batch["observation_future"]["image_primary"]]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()        
        traj_index = rlds_batch['_traj_index']
        frame_index = rlds_batch['_frame_index']

        import pdb; pdb.set_trace()

        trace_folder = os.path.join(self.data_root_dir.replace('baolinpeng/open-x', 'vlpdatasets'), self.trace_folder)
        trace_file = f"{dataset_name}/{traj_index}/{frame_index}.pth"
        trace_path = os.path.join(trace_folder, trace_file)
        if os.path.exists(trace_path):
            import pdb; pdb.set_trace()
            image_path = trace_path.replace(self.trace_folder, self.image_folder).replace(".pth", ".jpg")
            if os.path.exists(image_path):
                img_from_disk = Image.open(image_path)
            trace_info = torch.load(trace_path)
            pred_traces, pred_visibility = trace_info['trace_info']
            conv_vpt = self._construct_conv_for_vpt(img_from_disk, lang, pred_traces, pred_visibility)
        
        # action_token_ids = self.action_tokenizer.encode_actions_to_token_ids(action)
        action_token_ids = self.action_tokenizer.encode_actions_to_discrete_ids(action)

        action_str = f"{action_token_ids.tolist()}"

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        lang = lang.replace(".", "")
        conversation = [
            {"from": "human", "value": f"<image>\nThe robot is to {lang}. The robot is taking action in the space of [x,y,z,roll,pitch,yall,gripper state] which is split into 256 bins. What is the action for next step?"},
            {"from": "gpt", "value": action_str}, # placeholder for action tokens
            # self.action_tokenizer(action)
        ]
        prompt_builder = self.prompt_builder_fn.default_conversation.copy()
        roles = {
            "human": prompt_builder.roles[0], "user": prompt_builder.roles[0], "gpt": prompt_builder.roles[1], "assistant": prompt_builder.roles[1], "agent": prompt_builder.roles[1]
        }
        # Apply prompt templates
        prompt_builder.messages = []
        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == prompt_builder.roles[j % 2], f"{i}"
            prompt_builder.append_message(role, sentence["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        action_token_len = len(self.base_tokenizer(action_str).input_ids) - 1
        # # replace the action_placeholder_token_id with action_token_ids in input_ids
        # input_ids = list(input_ids)
        # input_ids_filled = []
        # for i, token_id in enumerate(input_ids):
        #     if token_id == action_placeholder_token_id:
        #         input_ids_filled.extend(action_token_ids.tolist())
        #     else:
        #         input_ids_filled.append(token_id)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(input_ids)

        pixel_values = transforms.Compose([transforms.ToTensor()])(img)
        image_pt = self.image_transform(img, return_tensors='pt')
        images = collections.defaultdict(list)
        for key, val in image_pt.items():
            images[key].append(val)

        pixel_values_future = torch.stack([transforms.Compose([transforms.ToTensor()])(item) for item in imgs_future], dim=0)

        # extract visual traces
        # trace_folder = self.trace_folder
        # if not os.path.exists(trace_folder):
        #     trace_folder = "./open-x-traces-v2"
        # trace_file = f"{dataset_name}/{traj_index}/{frame_index}.pth"
        # trace_path = os.path.join(trace_folder, trace_file)
        # if not os.path.exists(trace_path):        
        #     pixel_values_seq = torch.cat([pixel_values.unsqueeze(0), pixel_values_future], dim=0).unsqueeze(0)
        #     out = self.visual_tracker.extract_visual_trace(pixel_values_seq*255)
        #     # self.visual_tracker.visualize(*out)
        #     # save the visual trace to disk
        #     trace_info = {
        #         'dataset_name': dataset_name,
        #         'traj_index': traj_index,
        #         'frame_index': frame_index,
        #         'lang': lang,
        #         'action': action,
        #         'trace_info': out[1:]
        #     }
        #     os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        #     torch.save(trace_info, trace_path)
        
        # save image
        # image_folder = self.image_folder
        # if not os.path.exists(image_folder):
        #     image_folder = "./open-x-images-v2"
        # image_file = f"{dataset_name}/{traj_index}/{frame_index}.jpg"
        # image_path = os.path.join(image_folder, image_file)      
        # if not os.path.exists(image_path):
        #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
        #     img.save(image_path)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # NOTE: we add 2 to the length of the action to account for the \n\n and <|eot_id|> tokens!
        labels[: -(action_token_len + 2)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        return dict(pixel_values=images['pixel_values'], image_sizes=images['image_sizes'], pixel_values_future=pixel_values_future, input_ids=input_ids, labels=labels, dataset_name=dataset_name) 
        # return dict(pixel_values=pixel_values, pixel_values_future=pixel_values_future, action=action, conversation=conversation, dataset_name=dataset_name)

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_action_window_size: int = 0,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
        
        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: None, # ImageTransform,
        prompt_builder_fn: None # Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
