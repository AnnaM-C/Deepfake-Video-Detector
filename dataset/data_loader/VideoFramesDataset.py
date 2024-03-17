import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import bisect
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random

# # class VideoFramesDataset(Dataset):
# #     def __init__(self, root_path, splits_file, transform=None, chunk_size=32):
# #         self.root_path = root_path
# #         self.transform = transform
# #         self.chunk_size = chunk_size
# #         self.video_labels = self._load_splits(splits_file)
# #         self.index_mapping = self._create_index_mapping()

# #     def _load_splits(self, splits_file):
# #         video_labels = {}
# #         with open(splits_file, 'r') as f:
# #             for line in f:
# #                 parts = line.split(" ")
# #                 if len(parts) == 2:
# #                     video_path, label = parts
# #                     # print('Video Path, ', video_path)
# #                     # print("Labels, ", label)
# #                     video_labels[video_path] = int(label)
# #         return video_labels

# #     def _create_index_mapping(self):
# #         mapping = []
# #         for video_path, label in self.video_labels.items():
# #             all_frame_names = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0]))
# #             num_frames = len(all_frame_names)
# #             num_full_chunks = num_frames // self.chunk_size
# #             for chunk in range(num_full_chunks):
# #                 mapping.append((video_path, chunk, label))
# #         return mapping

# #     def __len__(self):
# #         return len(self.index_mapping)

# #     def __getitem__(self, idx):
# #         video_path, chunk_idx, label = self.index_mapping[idx]
# #         print("Video path, ", video_path)
# #         print("Chunk index, ", chunk_idx)
# #         print("Label, ", label)
# #         sorted_frame_names = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0]))
# #         start_frame = chunk_idx * self.chunk_size
# #         selected_frame_names = sorted_frame_names[start_frame:start_frame + self.chunk_size]
# #         frames = [Image.open(os.path.join(video_path, f)) for f in selected_frame_names]
# #         if self.transform:
# #             frames = [self.transform(frame) for frame in frames]
# #         # use pytorches stack library to stack frames into a single new tensor [N, C, H, W]
# #         frames_stack = torch.stack(frames)
# #         print("Frames stack shape, ", frames_stack.shape)
# #         # return the frame chunk and label
# #         return frames_stack, torch.tensor(label, dtype=torch.long)

## Class for nonoverlapping chunks
# class VideoFramesDataset(Dataset):
#     """Dataset class for FaceForensics++, FaceShifter, and DeeperForensics. Supports returning only a subset of forgery
#     methods in dataset"""
#     def __init__(
#             self,
#             real_videos,
#             fake_videos_file_paths,
#             config,
#             dataset_names,
#             frames_per_clip,
#             transform,
#             max_frames_per_video,
#             compression='c23',
#     ):
#         self.frames_per_clip = frames_per_clip
#         self.videos_per_type = {}
#         self.paths = []
#         self.transform = transform
#         self.clips_per_video = []
#         self.config = config
#         self.dataset_names = dataset_names
#         self.fake_videos_file_paths = fake_videos_file_paths

#         ds_methods = ['youtube'] + self.dataset_names  # Since we compute AUC, we need to include the Real dataset as well
#         print("Ds methods, ", ds_methods)
#         for ds_method in ds_methods:

#             # get list of video names
#             if ds_method == 'youtube':
#                 ds_type = 'original_sequences'
#                 print("Real videos, ", len(real_videos))
#                 videos = sorted(real_videos)
#                 # print("Length of original videos, ", len(videos))
#             # elif ds_type == 'DeeperForensics':  # Extra processing for DeeperForensics videos due to naming differences
#             #     videos = []
#             #     for f in fake_videos:
#             #         for el in os.listdir(video_paths):
#             #             if el.startswith(f.split('_')[0]):
#             #                 videos.append(el)
#             #     videos = sorted(videos)
#             else:
#                 ds_type = 'manipulated_sequences'
#                 fake_videos = self.fake_videos_file_paths[ds_method]
#                 print("Fake videos, ", len(fake_videos))
#                 videos = sorted(fake_videos)
#                 print("Ds method, ", ds_method)

#             video_paths = os.path.join('/vol/research/DeepFakeDet/notebooks/FaceForensics++', ds_type, ds_method, compression, 'frames')
#             # print("Video paths", video_paths)
#             self.videos_per_type[ds_method] = len(videos)
#             # print("ds_method, ", len(videos))
#             for video in videos:
#                 # print("Video, ", video)
#                 path = os.path.join(video_paths, video)
#                 # print("Path, ", path)
#                 num_frames = min(len(os.listdir(path)), max_frames_per_video)
#                 # print("Num frames per video (270), ", num_frames)
#                 # print("Frames per clip (5) === ", frames_per_clip)
#                 # print("Frames per video (270), ", max_frames_per_video)
#                 num_clips = num_frames // frames_per_clip
#                 # print("Num clips, ", num_clips)
#                 self.clips_per_video.append(num_clips)
#                 self.paths.append(path)
#             print("Num clips per video real/fake, ", len(self.clips_per_video))
#             print("Num frames real+fake, ", sum([x * 5 for x in self.clips_per_video]))

#         print("Num clips per video real+fake, ", len(self.clips_per_video))
#         print("Num frames real+fake, ", sum([x * 5 for x in self.clips_per_video]))

#         print("Total clips per video, ", len(self.clips_per_video))
#         print("Total real+fake videos, ", len(self.paths))

#         clip_lengths = torch.as_tensor(self.clips_per_video)
#         self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
#         print('Cumulative list real+fake video indexes, ', len(self.cumulative_sizes))
#         # print("Paths, ", self.paths)

#     def __len__(self):
#         return self.cumulative_sizes[-1]
    
#     def crop_frame(self, pil_img):
#         width, height = pil_img.size
#         left = width * 0.15
#         top = height * 0.15 
#         right = width * 0.85  
#         bottom = height  
#         pil_img_cropped = pil_img.crop((left, top, right, bottom))
#         return pil_img_cropped
    

#     def get_clip(self, idx):
#         video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#         if video_idx == 0:
#             clip_idx = idx
#         else:
#             clip_idx = idx - self.cumulative_sizes[video_idx - 1]
#         # print("Clip index, ", clip_idx)

#         path = self.paths[video_idx]
#         # frames = sorted(os.listdir(path))
#         # print("Path, ", path)

#         # filter out files starting with '._'
#         frames = [f for f in sorted(os.listdir(path)) if not f.startswith('._')]

#         # print("Frames, ", len(frames))

#         start_idx = clip_idx * self.frames_per_clip

#         end_idx = start_idx + self.frames_per_clip

#         sample = []
#         for idx in range(start_idx, end_idx, 1):
#             with Image.open(os.path.join(path, frames[idx])) as pil_img:
#                 # print("Pil image type, ", type(pil_img))
#                 # img = np.array(pil_img)
#                 # print("Image shape, ", img.shape)
#                 pil_img_cropped = self.crop_frame(pil_img)

#                 # apply transform to each frame
#                 if self.transform is not None:
#                     # print("Transforming data..")
#                     img = self.transform(pil_img_cropped)
#                     # print("Image shape, ", img.shape)
#                 else:
#                     img = np.array(pil_img_cropped)  # convert to np array if transform is None
#                     # print("Image shape, ", img.shape)
#                 sample.append(img)
#             # sample.append(img)

#         sample = np.stack(sample)
#         # if transform is not applied stack and convert sample list to tensor
#         if self.transform is None:
#             sample = np.stack(sample)
#             sample = torch.from_numpy(sample)

#         return sample, video_idx

#     def save_clip_plots(self, idx, save_dir, frames_to_show=5):
#         """
#         Saves a set number of consecutive frames for the video clip represented by the given index.

#         Args:
#         idx (int): Index of the video clip to plot.
#         frames_to_show (int): Number of frames to plot from the clip.
#         save_dir (str): Save directory for plots.
#         """
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         sample, video_idx = self.get_clip(idx)
#         path = self.paths[video_idx]
#         frames = sorted(os.listdir(path))

#         label = '0' if video_idx < self.videos_per_type['youtube'] else '1'

#         # Ensure we are showing consecutive frames starting from the beginning of the chunk
#         for frame_idx in range(frames_to_show):
#             if frame_idx >= len(sample):
#                 break  # avoid going above the available number of frames

#             # image_np = sample[frame_idx].cpu().numpy()  #convert to np
#             image_np = sample[frame_idx]

#             image_np = np.transpose(image_np, (1, 2, 0)) #rearrange from (C, H, W) to (H, W, C)

#             # filepath = os.path.join(path, frames[frame_idx])
#             # plt.figure(figsize=(10, 8))
#             # plt.imshow(sample[frame_idx])
#             # plt.title(f'Clip {idx} - Frame {frame_idx}\nPath: {filepath}\nLabel: {label}', fontsize=8)
#             # plt.axis('off')
#             # safe_filename = filepath.replace('/', '_').replace('\\', '_')
#             # plt.savefig(os.path.join(save_dir, f'Label_{label}_Clip_{idx}_Frame_{frame_idx}_{safe_filename}.png'))
#             # plt.close()
#             plt.figure(figsize=(10, 8))
#             plt.imshow(image_np)
#             plt.title(f'Clip {idx} - Frame {frame_idx}\nPath: {path}\nLabel: {label}\nCrop boarders 15%', fontsize=8)
#             plt.axis('off')
#             safe_filename = path.replace('/', '_').replace('\\', '_') + f'_Frame_{frame_idx}.png'
#             plt.savefig(os.path.join(save_dir, f'Label_{label}_Clip_{idx}_{safe_filename}'))
#             plt.close()

#     def __getitem__(self, idx):
#         sample, video_idx = self.get_clip(idx)
#         label = 0 if video_idx < self.videos_per_type['youtube'] else 1
#         label = torch.from_numpy(np.array(label))
#         # print("Sample shape, ", sample.shape)
#         # sample = torch.from_numpy(sample).unsqueeze(-1)
#         sample = torch.from_numpy(sample)
#         # if self.transform is not None:
#         #     sample = self.transform(sample)
#         if self.config.MODEL.NAME == "Physnet":
#             #NOTE: for PhysNet only
#             sample = sample.permute(1, 0, 2, 3)
#         return sample, label, video_idx

import os
import torch
import numpy as np
from PIL import Image
import bisect
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Class for overlapping chunks - added diff normalisation
class VideoFramesDataset(Dataset):
    # Every 2
    def __init__(
        self,
        real_videos,
        fake_videos_file_paths,
        config,
        dataset_names,
        frames_per_clip,
        transform,
        max_frames_per_video,
        compression='c23',
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.dataset_names = dataset_names
        self.fake_videos_file_paths = fake_videos_file_paths
        self.transform = transform
        self.config = config
        self.paths = []
        self.clips_per_video = []
        print("real videos, ", real_videos)
        ds_methods = ['youtube'] + dataset_names
        for ds_method in ds_methods:
            ds_type = 'original_sequences' if ds_method == 'youtube' else 'manipulated_sequences'
            videos = sorted(real_videos if ds_method == 'youtube' else fake_videos_file_paths[ds_method])
            video_paths = os.path.join('/vol/research/DeepFakeDet/notebooks/FaceForensics++', ds_type, ds_method, compression, 'frames')
            self.videos_per_type[ds_method] = len(videos)

            for video in videos:
                path = os.path.join(video_paths, video)
                num_frames = min(len(os.listdir(path)), max_frames_per_video)
                # interval of 2 frames
                num_clips = ((num_frames - frames_per_clip) // 2) + 1 if num_frames >= frames_per_clip else 0
                self.clips_per_video.append(num_clips)
                self.paths.append(path)
        print("clips per video, ", self.clips_per_video)

        self.cumulative_sizes = [sum(self.clips_per_video[:i+1]) for i in range(len(self.clips_per_video))]
        print("From videoloader print [-1] cummulative: ", self.cumulative_sizes[-1])
        print("From videoloader print length cummulative: ", len(self.cumulative_sizes))
        print("Cummulative list, ", self.cumulative_sizes)
    # Every 4
    # def __init__(
    #     self,
    #     real_videos,
    #     fake_videos_file_paths,
    #     config,
    #     dataset_names,
    #     frames_per_clip,
    #     transform,
    #     max_frames_per_video,
    #     compression='c23',
    # ):
    #     self.frames_per_clip = frames_per_clip
    #     self.videos_per_type = {}
    #     self.dataset_names = dataset_names
    #     self.fake_videos_file_paths = fake_videos_file_paths
    #     self.transform = transform
    #     self.config = config
    #     self.paths = []
    #     self.clips_per_video = []

    #     ds_methods = ['youtube'] + dataset_names
    #     for ds_method in ds_methods:
    #         ds_type = 'original_sequences' if ds_method == 'youtube' else 'manipulated_sequences'
    #         videos = sorted(real_videos if ds_method == 'youtube' else fake_videos_file_paths[ds_method])

    #         video_paths = os.path.join('/vol/research/DeepFakeDet/notebooks/FaceForensics++', ds_type, ds_method, compression, 'frames')
    #         self.videos_per_type[ds_method] = len(videos)
    #         for video in videos:
    #             path = os.path.join(video_paths, video)
    #             num_frames = min(len(os.listdir(path)), max_frames_per_video)
    #             # Calculate the number of clips that can be made from the video, considering the start of each new clip every 4 frames
    #             num_clips = ((num_frames - frames_per_clip) // 4) + 1 if num_frames >= frames_per_clip else 0
    #             self.clips_per_video.append(num_clips)
    #             self.paths.append(path)

    #     self.cumulative_sizes = [sum(self.clips_per_video[:i+1]) for i in range(len(self.clips_per_video))]
    #     # print('Paths length in Video loader initialiser, ', len(self.paths))
    #     # print("Clips per video, ", len(self.clips_per_video))
    #     # print("Cumulative list, ", self.cumulative_sizes)
    #     # print("Dataset length, ", len(self.cumulative_sizes))

        
    def __len__(self):
        print("From videoloader print [-1] cummulative: ", self.cumulative_sizes[-1])
        print("From videoloader print length cummulative: ", len(self.cumulative_sizes))
        print("Cummulative list, ", self.cumulative_sizes)

        return self.cumulative_sizes[-1]
    
    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    def crop_frame(self, pil_img, left, top, right, bottom):
            width, height = pil_img.size
            # left = width * 0.20
            # top = height * 0.20
            # right = width * 0.80
            # bottom = height
            left = width * left
            top = height * top
            right = width - (width * right)
            bottom = height
            pil_img_cropped = pil_img.crop((left, top, right, bottom))
            return pil_img_cropped

    # # Consecutive frames
    # def get_clip(self, idx):
    #     video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    #     clip_idx = idx - self.cumulative_sizes[video_idx - 1] if video_idx > 0 else idx
    #     path = self.paths[video_idx]
    #     frames = [f for f in sorted(os.listdir(path)) if not f.startswith('._')]
    #     start_idx = clip_idx
    #     end_idx = start_idx + self.frames_per_clip
    #     sample = []
    #     for idx in range(start_idx, end_idx):
    #         with Image.open(os.path.join(path, frames[idx])) as pil_img:
    #             pil_img_cropped = self.crop_frame(pil_img)
    #             img = self.transform(pil_img_cropped) if self.transform is not None else np.array(pil_img_cropped)
    #             sample.append(img)
    #     sample = np.stack(sample)
    #     if self.transform is None:
    #         sample = torch.from_numpy(sample)
    #     return sample, video_idx

    # Every 4
    # def get_clip(self, idx):
    #     video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    #     clip_idx = idx - self.cumulative_sizes[video_idx - 1] if video_idx > 0 else idx
    #     path = self.paths[video_idx]
    #     frames = [f for f in sorted(os.listdir(path)) if not f.startswith('._')]

    #     # Calculate the starting index of the frames for this clip
    #     start_idx = clip_idx * 4
    #     end_idx = start_idx + self.frames_per_clip  # Now we just add the number of frames per clip

    #     # Check to ensure we have enough frames for this clip
    #     if end_idx > len(frames):
    #         return None, video_idx  # Skip this clip if not enough frames

    #     sample = []
    #     for idx in range(start_idx, end_idx):
    #         with Image.open(os.path.join(path, frames[idx])) as pil_img:
    #             pil_img_cropped = self.crop_frame(pil_img)
    #             img = self.transform(pil_img_cropped) if self.transform is not None else np.array(pil_img_cropped)
    #             sample.append(img)

    #     sample = np.stack(sample)
    #     if self.transform is None:
    #         sample = torch.from_numpy(sample)
    #     return sample, video_idx

    # Every 2
    # adapt to return frame paths
    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        print("Video index, ", video_idx)
        clip_idx = idx - self.cumulative_sizes[video_idx - 1] if video_idx > 0 else idx
        path = self.paths[video_idx]
        frames = [f for f in sorted(os.listdir(path)) if not f.startswith('._')]

        # TODO: rotation
        rotation_angle = random.uniform(-5, 5)
        crop_percents = {
            'left': random.uniform(0, 0.20),
            'top': random.uniform(0, 0.20),
            'right': random.uniform(0, 0.20),
            'bottom': random.uniform(0, 0.20)
        }

        start_idx = clip_idx * 2
        end_idx = start_idx + self.frames_per_clip

        if end_idx > len(frames):
            return None, video_idx

        sample = []
        frame_filepaths = []
        for idx in range(start_idx, end_idx):
            frame_path = os.path.join(path, frames[idx])
            frame_filepaths.append(frame_path)
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                #NOTE: rotation
                pil_img_rotated = pil_img.rotate(rotation_angle)
                pil_img_cropped = self.crop_frame(pil_img_rotated, **crop_percents)
                img = self.transform(pil_img_cropped) if self.transform is not None else np.array(pil_img_cropped)
                sample.append(img)

        sample = np.stack(sample)
        if self.config.TEST.DATA.PREPROCESS.DATA_TYPE[0] == 'DiffNormalized':
            # print("Diff Normalise")
            # print(sample.shape)
            sample = self.diff_normalize_data(sample)
        if self.transform is None:
            sample = torch.from_numpy(sample)
        return sample, video_idx, frame_filepaths
    
    def save_clip_plots(self, idx, save_dir, frames_to_show=5):
        """
        Saves a set number of consecutive frames for the video clip represented by the given index.

        Args:
        idx (int): Index of the video clip to plot.
        frames_to_show (int): Number of frames to plot from the clip.
        save_dir (str): Save directory for plots.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Video index as argument, ", idx)
        sample, video_idx, _ = self.get_clip(idx)
        path = self.paths[video_idx]
        frames = sorted(os.listdir(path))

        label = '0' if video_idx < self.videos_per_type['youtube'] else '1'

        # Ensure we are showing consecutive frames starting from the beginning of the chunk
        for frame_idx in range(frames_to_show):
            if frame_idx >= len(sample):
                break  # avoid going above the available number of frames

            # image_np = sample[frame_idx].cpu().numpy()  #convert to np
            image_np = sample[frame_idx]

            image_np = np.transpose(image_np, (1, 2, 0)) #rearrange from (C, H, W) to (H, W, C)

            # filepath = os.path.join(path, frames[frame_idx])
            # plt.figure(figsize=(10, 8))
            # plt.imshow(sample[frame_idx])
            # plt.title(f'Clip {idx} - Frame {frame_idx}\nPath: {filepath}\nLabel: {label}', fontsize=8)
            # plt.axis('off')
            # safe_filename = filepath.replace('/', '_').replace('\\', '_')
            # plt.savefig(os.path.join(save_dir, f'Label_{label}_Clip_{idx}_Frame_{frame_idx}_{safe_filename}.png'))
            # plt.close()
            plt.figure(figsize=(10, 8))
            plt.imshow(image_np)
            plt.title(f'Clip {idx} - Frame {frame_idx}\nPath: {path}\nLabel: {label}\nCrop boarders 15%', fontsize=8)
            plt.axis('off')
            safe_filename = path.replace('/', '_').replace('\\', '_') + f'_Frame_{frame_idx}.png'
            plt.savefig(os.path.join(save_dir, f'Label_{label}_Clip_{idx}_{safe_filename}'))
            plt.close()

    def __getitem__(self, idx):
        sample, video_idx, frame_filepaths = self.get_clip(idx)
        label = 0 if video_idx < self.videos_per_type['youtube'] else 1
        sample = torch.from_numpy(sample)
        label = torch.tensor(label, dtype=torch.long)
        if self.config.MODEL.NAME == "Physnet":
            #NOTE: for PhysNet only
            sample = sample.permute(1, 0, 2, 3)
        return sample, label, video_idx, frame_filepaths
    