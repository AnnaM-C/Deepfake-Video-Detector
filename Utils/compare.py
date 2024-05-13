import os

def compare_video_folders(root_dir_1, root_dir_2):
    videos_in_dir1 = set()
    videos_in_dir2 = set()

    def add_video_dirs(root_dir, video_set):
        for entry in os.listdir(root_dir):
            path = os.path.join(root_dir, entry)
            if os.path.isdir(path):
                video_set.add(entry)

    add_video_dirs(root_dir_1, videos_in_dir1)
    add_video_dirs(root_dir_2, videos_in_dir2)

    only_in_dir1 = videos_in_dir1 - videos_in_dir2
    only_in_dir2 = videos_in_dir2 - videos_in_dir1

    print(f"Videos in '{root_dir_1}' not in '{root_dir_2}':")
    for video in only_in_dir1:
        print(os.path.join(root_dir_1, video))

    print(f"\nVideos in '{root_dir_2}' not in '{root_dir_1}':")
    for video in only_in_dir2:
        print(os.path.join(root_dir_2, video))

# Specify the root directories to compare
# root_dir_1 = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_4_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata'
# root_dir_2 = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_8_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata'

root_dir_1 = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_4_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_latest'
root_dir_2 = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_ADAM_LR=0.0001_LRReducer_16_frames_multi_model_testNT'

compare_video_folders(root_dir_1, root_dir_2)
