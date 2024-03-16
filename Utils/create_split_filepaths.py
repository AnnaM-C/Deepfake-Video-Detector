#!/usr/bin/env python3
# Code adapted from IBM https://github.com/IBM/action-recognition-pytorch to create split paths for FF++ dataset, rewritten into a class
# FF++ protocol provided by the original FF++ paper https://github.com/ondyari/FaceForensics

import os
import json
import argparse


class Splitter:
    def __init__(self, train_file, val_file, test_file, train_img_folder, val_img_folder, test_img_folder, train_file_list, val_file_list, test_file_list, label, video_folder):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.train_img_folder = train_img_folder
        self.val_img_folder = val_img_folder
        self.test_img_folder = test_img_folder
        self.train_file_list = train_file_list
        self.val_file_list = val_file_list
        self.test_file_list = test_file_list
        self.label = label
        self.video_folder = video_folder

        self.train_videos = self.load_video_list(self.train_file)
        self.val_videos = self.load_video_list(self.val_file)
        self.test_videos = self.load_test_video_list(self.test_file)

        self.create_train_video()
        self.create_val_video()
        self.create_test_video()

    def load_video_list(self, file_path):
        videos = []
        with open(file_path) as f:
            file_list = json.load(f)
            for temp in file_list:
                # Assumes data in json is a dictionary with id
                # For Fakes
                if self.label == 1:
                    videos.append((temp[0] + "_" + temp[1], 1))
                    videos.append((temp[1] + "_" + temp[0], 1))
                # For Real
                elif self.label == 0:
                    videos.append((temp[0], 0))
                    videos.append((temp[1], 0))

        return videos

    def load_test_video_list(self, file_path):
        videos = []
        with open(file_path) as f:
            file_list = json.load(f)
            for temp in file_list:
                # For Fakes
                if self.label == 1:
                    videos.append((temp[0] + "_" + temp[1], 1))
                    videos.append((temp[1] + "_" + temp[0], 1))
                # For Real
                elif self.label == 0:
                    videos.append((temp[0], 0))
                    videos.append((temp[1], 0))
        return videos


    def create_train_video(self):
        with open(self.train_file_list, 'w') as f:
            for video, label in self.train_videos:
                print(f"{os.path.join(self.video_folder, video)} {label}", file=f)
        print("Train split done!")

    def create_val_video(self):
        with open(self.val_file_list, 'w') as f:
            for video, label in self.val_videos:
                print(f"{os.path.join(self.video_folder, video)} {label}", file=f)
        print("Val split done!")

    def create_test_video(self):
        with open(self.test_file_list, 'w') as f:
            for video, label in self.test_videos:
                print(f"{os.path.join(self.video_folder, video)} {label}", file=f)
        print("Test split done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset using FF++ protocol")
    
    parser.add_argument("--root", type=str, help="Path to the dataset root")

    args = parser.parse_args()
    folder_root = args.root

    if folder_root == "":
        raise ValueError("Please set folder_root")


    intra_datasets = [{'Deepfakes':1}, {'NeuralTextures':1}, {'Face2Face':1}, {'FaceSwap':1}, {'FaceShifter':1}, {'youtube': 0}]


    for dataset in intra_datasets:

        for name, label in dataset.items():
            if label == 0:
                source = "original_sequences"
            elif label == 1:
                source = "manipulated_sequences"
            else:
                raise ValueError("Label not recognised")
            
            video_folder = "{0}/{1}/{2}/c23/frames".format(folder_root, source, name)
            if not os.path.exists(video_folder):  # check if the dataset directory exists
                print(f"Dataset for {name} does not exist, skipping...")
                continue  # skip the rest of the loop for this dataset

            # input
            train_file = "{}/train.json".format(folder_root)
            val_file = "{}/val.json".format(folder_root)
            test_file = "{}/test.json".format(folder_root)

            # output
            train_img_folder = "{0}/{1}/{2}/c23/train".format(folder_root, source, name)
            val_img_folder = "{0}/{1}/{2}/c23/val".format(folder_root, source, name)
            test_img_folder = "{0}/{1}/{2}/c23/test".format(folder_root, source, name)

            train_file_list = "{0}/{1}/{2}/c23/train.txt".format(folder_root, source, name)
            val_file_list = "{0}/{1}/{2}/c23/val.txt".format(folder_root, source, name)
            test_file_list = "{0}/{1}/{2}/c23/test.txt".format(folder_root, source, name)

            label = label

            print(f"Splitting {name} ..")
            splitter = Splitter(train_file, val_file, test_file, train_img_folder, val_img_folder, test_img_folder, train_file_list, val_file_list, test_file_list, label, video_folder)