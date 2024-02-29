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

# class VideoFramesDataset(Dataset):
#     def __init__(self, root_path, splits_file, transform=None, chunk_size=32):
#         self.root_path = root_path
#         self.transform = transform
#         self.chunk_size = chunk_size
#         self.video_labels = self._load_splits(splits_file)
#         self.index_mapping = self._create_index_mapping()

#     def _load_splits(self, splits_file):
#         video_labels = {}
#         with open(splits_file, 'r') as f:
#             for line in f:
#                 parts = line.split(" ")
#                 if len(parts) == 2:
#                     video_path, label = parts
#                     # print('Video Path, ', video_path)
#                     # print("Labels, ", label)
#                     video_labels[video_path] = int(label)
#         return video_labels

#     def _create_index_mapping(self):
#         mapping = []
#         for video_path, label in self.video_labels.items():
#             all_frame_names = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0]))
#             num_frames = len(all_frame_names)
#             num_full_chunks = num_frames // self.chunk_size
#             for chunk in range(num_full_chunks):
#                 mapping.append((video_path, chunk, label))
#         return mapping

#     def __len__(self):
#         return len(self.index_mapping)

#     def __getitem__(self, idx):
#         video_path, chunk_idx, label = self.index_mapping[idx]
#         print("Video path, ", video_path)
#         print("Chunk index, ", chunk_idx)
#         print("Label, ", label)
#         sorted_frame_names = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0]))
#         start_frame = chunk_idx * self.chunk_size
#         selected_frame_names = sorted_frame_names[start_frame:start_frame + self.chunk_size]
#         frames = [Image.open(os.path.join(video_path, f)) for f in selected_frame_names]
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]
#         # use pytorches stack library to stack frames into a single new tensor [N, C, H, W]
#         frames_stack = torch.stack(frames)
#         print("Frames stack shape, ", frames_stack.shape)
#         # return the frame chunk and label
#         return frames_stack, torch.tensor(label, dtype=torch.long)

class VideoFramesDataset(Dataset):
    """Dataset class for FaceForensics++, FaceShifter, and DeeperForensics. Supports returning only a subset of forgery
    methods in dataset"""
    def __init__(
            self,
            real_videos,
            fake_videos_file_paths,
            config,
            dataset_names,
            frames_per_clip,
            transform,
            compression='c23',
            max_frames_per_video=270,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.transform = transform
        self.clips_per_video = []
        self.config = config
        self.dataset_names = dataset_names
        self.fake_videos_file_paths = fake_videos_file_paths

        ds_methods = ['youtube'] + self.dataset_names  # Since we compute AUC, we need to include the Real dataset as well
        print("Ds methods, ", ds_methods)
        for ds_method in ds_methods:

            # get list of video names
            if ds_method == 'youtube':
                ds_type = 'original_sequences'
                # print("Real videos, ", real_videos)
                videos = sorted(real_videos)
                # print("Length of original videos, ", len(videos))
            # elif ds_type == 'DeeperForensics':  # Extra processing for DeeperForensics videos due to naming differences
            #     videos = []
            #     for f in fake_videos:
            #         for el in os.listdir(video_paths):
            #             if el.startswith(f.split('_')[0]):
            #                 videos.append(el)
            #     videos = sorted(videos)
            else:
                ds_type = 'manipulated_sequences'
                videos = sorted(self.fake_videos_file_paths[ds_method])
                print("Ds method, ", ds_method)
                print(videos)

                print("Length of fake videos, ", len(videos))

            video_paths = os.path.join('/vol/research/DeepFakeDet/notebooks/FaceForensics++', ds_type, ds_method, compression, 'frames')
            # print("Video paths", video_paths)
            self.videos_per_type[ds_method] = len(videos)
            # print("ds_method, ", len(videos))
            for video in videos:
                # print("Video, ", video)
                path = os.path.join(video_paths, video)
                # print("Path, ", path)
                num_frames = min(len(os.listdir(path)), max_frames_per_video)
                num_clips = num_frames // frames_per_clip
                self.clips_per_video.append(num_clips)
                self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        # print('Cumulative list, ', self.cumulative_sizes)
        # print("Paths, ", self.paths)
        print("Length of paths, ", len(self.paths))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        # print("Clip index, ", clip_idx)

        path = self.paths[video_idx]
        # frames = sorted(os.listdir(path))
        # print("Path, ", path)

        # filter out files starting with '._'
        frames = [f for f in sorted(os.listdir(path)) if not f.startswith('._')]

        # print("Frames, ", len(frames))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                # print("Pil image type, ", type(pil_img))
                # img = np.array(pil_img)
                # print("Image shape, ", img.shape)

                # apply transform to each frame
                if self.transform is not None:
                    # print("Transforming data..")
                    img = self.transform(pil_img)
                    # print("Image shape, ", img.shape)
                else:
                    img = np.array(pil_img)  # convert to np array if transform is None
                    # print("Image shape, ", img.shape)
                sample.append(img)
            # sample.append(img)

        sample = np.stack(sample)
        # if transform is not applied stack and convert sample list to tensor
        if self.transform is None:
            sample = np.stack(sample)
            sample = torch.from_numpy(sample)

        return sample, video_idx

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

        sample, video_idx = self.get_clip(idx)
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
            plt.title(f'Clip {idx} - Frame {frame_idx}\nPath: {path}\nLabel: {label}', fontsize=8)
            plt.axis('off')
            safe_filename = path.replace('/', '_').replace('\\', '_') + f'_Frame_{frame_idx}.png'
            plt.savefig(os.path.join(save_dir, f'Label_{label}_Clip_{idx}_{safe_filename}'))
            plt.close()

    def __getitem__(self, idx):
        sample, video_idx = self.get_clip(idx)
        label = 0 if video_idx < self.videos_per_type['youtube'] else 1
        label = torch.from_numpy(np.array(label))
        # print("Sample shape, ", sample.shape)
        # sample = torch.from_numpy(sample).unsqueeze(-1)
        sample = torch.from_numpy(sample)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        if self.config.MODEL.NAME == "Physnet":
            #NOTE: for PhysNet only
            sample = sample.permute(1, 0, 2, 3)
        return sample, label, video_idx
