import pickle

# pickle_file_path = '/vol/research/DeepFakeDet/rPPG-Toolbox/runs/exp/logs/PhysNet_Face2Face_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_0.001_0.0001_LR_multi_model_epoch11_test/Face2Face/saved_test_outputs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_0.001_0.0001_LR_multi_model_Epoch11_Face2Face_outputs.pickle'
pickle_file_path="/vol/research/DeepFakeDet/rPPG-Toolbox/runs/exp/logs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.3binary_0.7rPPG_scaling_multi_model/NeuralTextures/saved_test_outputs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.3binary_0.7rPPG_scaling_multi_model_outputs.pickle"

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