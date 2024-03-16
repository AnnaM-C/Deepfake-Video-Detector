import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageDataLoader(Dataset):
    """Dataset class for loading individual frames from directories."""
    def __init__(
            self,
            real_images_folders,
            fake_images_folders,
            transform,
            max_frames_per_video,
    ):
        self.image_paths = []  
        self.labels = []
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        # for folder in real_images_folders:
        #     for frame in sorted(os.listdir(folder)):
        #         frame_path = os.path.join(folder, frame)
        #         if os.path.isfile(frame_path):
        #             self.image_paths.append(frame_path)
        #             self.labels.append(0)
        # real_len=(len(self.image_paths))
        # print("Real images folder, ", real_len)
        # for dataset_name, folders in fake_images_folders.items():
        #     for folder in folders:
        #         for frame in sorted(os.listdir(folder)):
        #             frame_path = os.path.join(folder, frame)
        #             if os.path.isfile(frame_path):
        #                 self.image_paths.append(frame_path)
        #                 self.labels.append(1)
        # print("Fake images len == 405490 , ", len(self.image_paths)-real_len)
        # real_folder = []
        # fake_folder = []
        for folder in real_images_folders:
            # real_folder.append(folder)
            for i, frame in enumerate(sorted(os.listdir(folder))):
                if frame.startswith('._'):
                    continue
                frame_path = os.path.join(folder, frame)
                if i >= self.max_frames_per_video:
                    break
                if os.path.isfile(frame_path):
                    self.image_paths.append(frame_path)
                    self.labels.append(0)
                    
        real_len=(len(self.image_paths))
        # print("Number of real folders (719, 140, 140), ", len(real_folder))
        print("Real images (194130 or 15400) ===, ", real_len)

        for dataset_name, folders in fake_images_folders.items():
            for folder in folders:
                # fake_folder.append(folder)
                for i, frame in enumerate(sorted(os.listdir(folder))):
                    if frame.startswith('._'):
                        continue
                    frame_path = os.path.join(folder, frame)
                    if i >= self.max_frames_per_video:
                        break
                    if os.path.isfile(frame_path):
                        self.image_paths.append(frame_path)
                        self.labels.append(1)
        # print("Number of fake folders (719, 140, 140) ===, ", len(fake_folder))
        print("Total, ", len(self.image_paths))
        print("Fake images (194130 or 15400) ===, ", len(self.image_paths)-real_len)
        print("Total frames (388260 or 30800) ===, ", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def plot_image(self, idx, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # get the image to plot
        img, label=self.__getitem__(idx)
        img = np.transpose(img, (1,2,0))
        plt.figure(figsize=(10,8))
        plt.imshow(img)
        plt.title(f'Image {idx} \n Label: {label}', fontsize=8)
        plt.axis('off')
        save_filename=f'Image_{idx}.png'
        plt.savefig(os.path.join(save_dir, f'Label_{label}_Frame_{idx}'))
        plt.close()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img)
                img = torch.from_numpy(img).permute(2, 0, 1)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return img, label
