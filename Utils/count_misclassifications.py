import os

def count_items_in_subdirectories(directory):
    item_counts = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        # Count only files here, since we're interested in file counts.
        item_counts[dirpath] = len(filenames)
    return item_counts

def compare_directories(root1, root2):
    items1 = count_items_in_subdirectories(root1)
    items2 = count_items_in_subdirectories(root2)
    
    # Extract the set of directories relative to the root
    set1 = set(os.path.relpath(path, root1) for path in items1.keys())
    set2 = set(os.path.relpath(path, root2) for path in items2.keys())
    
    # Find directories that are in root1 but not in root2
    unique_to_root1 = set1 - set2
    
    # Prepare the report dictionary for unique directories and their file counts
    unique_directory_counts = {}
    for relative_path in unique_to_root1:
        absolute_path = os.path.join(root1, relative_path)
        unique_directory_counts[relative_path] = items1[absolute_path]
    
    unique_directory_counts = dict(sorted(unique_directory_counts.items(), key=lambda item: item[1], reverse=True))

    return unique_directory_counts

root_path1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_chunk_size_16_testdata"

root_path2 = "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_0.001_0.0001_LR_multi_model_epoch26"

unique_items = compare_directories(root_path1, root_path2)

for path, count in unique_items.items():
    print(f"{path}: {count} files")