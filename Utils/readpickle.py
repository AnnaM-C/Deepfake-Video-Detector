import pickle

pickle_file_path="/vol/research/DeepFakeDet/rPPG-Toolbox/runs/exp/logs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_continuation_from_epoch_15_full_ds_testset/NeuralTextures/saved_test_outputs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_continuation_from_epoch_15_full_ds_Epoch15_NeuralTextures_outputs.pickle"

def load_pickle_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

data = load_pickle_data(pickle_file_path)

print(type(data))

if isinstance(data, dict):
    print("Keys in the loaded dictionary:", data.keys())
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")
elif isinstance(data, list):
    print("Length of the loaded list:", len(data))
    print("First few elements of the list:", data[:5])