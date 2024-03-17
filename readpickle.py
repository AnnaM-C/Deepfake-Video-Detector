import pickle

pickle_file_path = 'runs/exp/logs/Xception_Face2Face_nopretrain_ADAM_LR=0.0002_batch_6_real_paths_full_dataset/Face2Face/saved_test_outputs/Xception_Face2Face_nopretrain_ADAM_LR=0.0002_batch_6_real_paths_full_dataset_Epoch18_Face2Face_outputs.pickle'

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