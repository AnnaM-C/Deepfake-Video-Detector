# import os
# from collections import defaultdict

# def analyze_missclassified_videos(root_dir):
#     paired_videos = defaultdict(lambda: {'real': [], 'fake': []})
#     unpaired_videos = {'real': [], 'fake': []}
    
#     for root, dirs, files in os.walk(root_dir):
#         for dir_name in dirs:
#             if '_' in dir_name and len(dir_name.split('_')[0]) == 3:
#                 real_name = dir_name.split('_')[0]
#                 if os.path.exists(os.path.join(root, real_name)):
#                     paired_videos[real_name]['fake'].append(dir_name)
#                     paired_videos[real_name]['real'].append(real_name)
#                 else:
#                     unpaired_videos['fake'].append(dir_name)
#             elif len(dir_name) == 3:
#                 if dir_name not in paired_videos or not paired_videos[dir_name]['fake']:
#                     unpaired_videos['real'].append(dir_name)

#     paired_count = sum(len(v['real']) + len(v['fake']) for v in paired_videos.values())
#     unpaired_count = len(unpaired_videos['real']) + len(unpaired_videos['fake'])

#     return paired_videos, unpaired_videos, paired_count, unpaired_count

# # root_dir = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_8_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata'
# # root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_4_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata_F2F"
# # root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_chunk_size_16_testdata"
# root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_chunk_size_32_testdata"

# paired_videos, unpaired_videos, paired_count, unpaired_count = analyze_missclassified_videos(root_dir)

# print(f"Paired Misclassified Videos (Both Real and Fake): {paired_count}")
# print(f"Unpaired Misclassified Videos: {unpaired_count}")
# print("Paired Videos:", paired_videos)
# print("Unpaired Videos:", unpaired_videos)
import os
from collections import defaultdict

def analyze_missclassified_videos(root_dir):
    video_counts = {'real': [], 'fake': []}
    
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if '_' in dir_name and len(dir_name.split('_')[0]) == 3:
                real_name = dir_name.split('_')[0]
                if os.path.exists(os.path.join(root, real_name)):
                    video_counts['fake'].append(dir_name)
                    video_counts['real'].append(real_name)
            elif len(dir_name) == 3:
                video_counts['real'].append(dir_name)

    total_fake_missed = len(video_counts['fake'])
    total_real_missed = len(video_counts['real'])
    total_missed = total_fake_missed + total_real_missed

    return total_fake_missed, total_real_missed, total_missed

# F2F
# root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_4_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata_F2F"
# root_dir = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_8_frames_new_preprocessed_ds_real_paths_overlap_skip2_frames_testdata_F2F'
# # root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_chunk_size_16_testdata"
# root_dir = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_chunk_size_32_testdata"

# NT
# root_dir="/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_chunk_size_4_testdata"
# root_dir="/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_chunk_size_8_testdata"
# root_dir="/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_chunk_size_16_testdata"
root_dir="/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTextures_chunk_size_32_testdata"


total_fake_missed, total_real_missed, total_missed = analyze_missclassified_videos(root_dir)

print(f"Total Fake Misclassified Videos: {total_fake_missed}")
print(f"Total Real Misclassified Videos: {total_real_missed}")
print(f"Total Misclassified Videos: {total_missed}")


