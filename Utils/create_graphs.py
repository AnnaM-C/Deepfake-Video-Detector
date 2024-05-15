import tensorflow as tf
import os
import matplotlib.pyplot as plt

def plot_individual_graphs(logdir1,logdir2,logdir3a, logdir3b):
    train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32 = [], [], [], []
    train_accuracies_nonoverlapping, valid_accuracies_nonoverlapping, train_loss_nonoverlapping, valid_loss_nonoverlapping = [], [], [], []
    train_accuracies_overlapping, valid_accuracies_overlapping, train_loss_overlapping, valid_loss_overlapping = [], [], [], []


    training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32 = [], [], [], []
    training_accuracies_epochs_list_nonoverlapping, validation_accuracies_epochs_list_nonoverlapping, training_loss_epochs_list_nonoverlapping, validation_loss_epochs_list_nonoverlapping = [], [], [], []
    training_accuracies_epochs_list_overlapping, validation_accuracies_epochs_list_overlapping, training_loss_epochs_list_overlapping, validation_loss_epochs_list_overlapping = [], [], [], []

    # function to process directories and update epochs and accuracies lists
    def process_directory(logdir, training_accuracies_epochs_list, validation_accuracies_epochs_list, training_loss_epochs_list, validation_loss_epochs_list, train_accuracies_list, valid_accuracies_list, train_loss_list, valid_loss_list, initial_epoch=0):
        for file in os.listdir(logdir):
            if file.startswith("events.out.tfevents"):
                filepath = os.path.join(logdir, file)
                for e in tf.compat.v1.train.summary_iterator(filepath):
                    for v in e.summary.value:
                        print("Value, ", v)
                        if v.tag == 'Accuracy/train':
                            adjusted_epoch = e.step + initial_epoch
                            training_accuracies_epochs_list.append(adjusted_epoch)
                            train_accuracies_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Accuracy: {v.simple_value}")
                        if v.tag == 'Accuracy/validation':
                            adjusted_epoch = e.step + initial_epoch
                            validation_accuracies_epochs_list.append(adjusted_epoch)
                            valid_accuracies_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Accuracy: {v.simple_value}")
                        if v.tag == 'Loss/train':
                            adjusted_epoch = e.step + initial_epoch
                            training_loss_epochs_list.append(adjusted_epoch)
                            train_loss_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Loss: {v.simple_value}")
                        if v.tag == 'Loss/validation':
                            adjusted_epoch = e.step + initial_epoch
                            validation_loss_epochs_list.append(adjusted_epoch)
                            valid_loss_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Loss: {v.simple_value}")

    # process the nonoverlapping frame directories
    process_directory(logdir1, training_accuracies_epochs_list_nonoverlapping, validation_accuracies_epochs_list_nonoverlapping, training_loss_epochs_list_nonoverlapping, validation_loss_epochs_list_nonoverlapping, train_accuracies_nonoverlapping, valid_accuracies_nonoverlapping, train_loss_nonoverlapping, valid_loss_nonoverlapping)

    # process the overlapping frame directories
    process_directory(logdir2, training_accuracies_epochs_list_overlapping, validation_accuracies_epochs_list_overlapping, training_loss_epochs_list_overlapping, validation_loss_epochs_list_overlapping, train_accuracies_overlapping, valid_accuracies_overlapping, train_loss_overlapping, valid_loss_overlapping)

    # process the 32 frame directories
    process_directory(logdir3a, training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32, train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32)
    last_epoch_32 = max(training_accuracies_epochs_list_32) if training_accuracies_epochs_list_32 else 0
    process_directory(logdir3b, training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32, train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32, initial_epoch=last_epoch_32 + 1)

    # plot training accuracy
    plt.figure(figsize=[8,6])
    plt.plot(training_accuracies_epochs_list_nonoverlapping, train_accuracies_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(training_accuracies_epochs_list_overlapping, train_accuracies_overlapping, label='Overlapping Chunks')
    plt.plot(training_accuracies_epochs_list_32, train_accuracies_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Training Accuracy over Epochs for non-overlapping vs overlapping chunks and augmentation')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_overlapping_and_augmentations/PhysNet_Training_Accuracy_augment.png")

    # plot validation accuracy
    plt.figure(figsize=[8,6])
    plt.plot(validation_accuracies_epochs_list_nonoverlapping, valid_accuracies_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(validation_accuracies_epochs_list_overlapping, valid_accuracies_overlapping, label='Overlapping Chunks')
    plt.plot(validation_accuracies_epochs_list_32, valid_accuracies_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Validation Accuracy over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_overlapping_and_augmentations/PhysNet_Valid_Accuracy_augment.png")

    # plot training loss
    plt.figure(figsize=[8,6])
    plt.plot(training_loss_epochs_list_nonoverlapping, train_loss_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(training_loss_epochs_list_overlapping, train_loss_overlapping, label='Overlapping Chunks')
    plt.plot(training_loss_epochs_list_32, train_loss_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Training Loss over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_overlapping_and_augmentations/PhysNet_Training_Loss_augment.png")

    # plot validation loss
    plt.figure(figsize=[8,6])
    plt.plot(validation_loss_epochs_list_nonoverlapping, valid_loss_nonoverlapping, label='Non-overlapping Chunks')
    plt.plot(validation_loss_epochs_list_overlapping, valid_loss_overlapping, label='Overlapping Chunks')
    plt.plot(validation_loss_epochs_list_32, valid_loss_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Validation Loss over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_overlapping_and_augmentations/PhysNet_Validation_Loss_augment.png")


def plot_graphs_and_join_epochs(logdir_16_frames_1, logdir_16_frames_2, logdir_32_frames_1, logdir_32_frames_2, logdir_8_frames_1, logdir_4_frames_1):
    # Initialize lists for 32, 16, 8, and 4 frames
    accuracies_loss_dicts = {
        "32": {"train_acc": [], "valid_acc": [], "train_loss": [], "valid_loss": [], "train_acc_epochs": [], "valid_acc_epochs": [], "train_loss_epochs": [], "valid_loss_epochs": []},
        "16": {"train_acc": [], "valid_acc": [], "train_loss": [], "valid_loss": [], "train_acc_epochs": [], "valid_acc_epochs": [], "train_loss_epochs": [], "valid_loss_epochs": []},
        "8": {"train_acc": [], "valid_acc": [], "train_loss": [], "valid_loss": [], "train_acc_epochs": [], "valid_acc_epochs": [], "train_loss_epochs": [], "valid_loss_epochs": []},
        "4": {"train_acc": [], "valid_acc": [], "train_loss": [], "valid_loss": [], "train_acc_epochs": [], "valid_acc_epochs": [], "train_loss_epochs": [], "valid_loss_epochs": []}
    }

    def process_directory(logdir, frame_len, initial_epoch=0):
        dict_refs = accuracies_loss_dicts[frame_len]
        for file in os.listdir(logdir):
            if file.startswith("events.out.tfevents"):
                filepath = os.path.join(logdir, file)
                for e in tf.compat.v1.train.summary_iterator(filepath):
                    for v in e.summary.value:
                        if v.tag == 'Accuracy/train':
                            adjusted_epoch = e.step + initial_epoch
                            dict_refs["train_acc_epochs"].append(adjusted_epoch)
                            dict_refs["train_acc"].append(v.simple_value)
                        elif v.tag == 'Accuracy/validation':
                            adjusted_epoch = e.step + initial_epoch
                            dict_refs["valid_acc_epochs"].append(adjusted_epoch)
                            dict_refs["valid_acc"].append(v.simple_value)
                        elif v.tag == 'Loss/train':
                            adjusted_epoch = e.step + initial_epoch
                            dict_refs["train_loss_epochs"].append(adjusted_epoch)
                            dict_refs["train_loss"].append(v.simple_value)
                        elif v.tag == 'Loss/validation':
                            adjusted_epoch = e.step + initial_epoch
                            dict_refs["valid_loss_epochs"].append(adjusted_epoch)
                            dict_refs["valid_loss"].append(v.simple_value)

    # Process directories for each frame length
    process_directory(logdir_16_frames_1, "16")
    process_directory(logdir_16_frames_2, "16", initial_epoch=max(accuracies_loss_dicts["16"]["train_acc_epochs"]) + 1)
    process_directory(logdir_32_frames_1, "32")
    process_directory(logdir_32_frames_2, "32", initial_epoch=max(accuracies_loss_dicts["32"]["train_acc_epochs"]) + 1)
    process_directory(logdir_8_frames_1, "8")
    process_directory(logdir_4_frames_1, "4")

    # Plotting function for accuracy and loss
    def plot_metric(metric_name, title, y_label, save_path):
        plt.figure(figsize=[8,6])
        for frame_len, metrics in accuracies_loss_dicts.items():
            epochs = metrics[f"{metric_name}_epochs"]
            values = metrics[metric_name]
            plt.plot(epochs, values, label=f'{frame_len} Frames')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True)
        plt.savefig(save_path)

    # Plot graphs
    plot_metric("train_acc", "Training Accuracy over Epochs", "Accuracy", "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_size_experiments/Training_Accuracy_2.png")
    plot_metric("valid_acc", "Validation Accuracy over Epochs", "Accuracy", "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_size_experiments/Validation_Accuracy_2.png")
    plot_metric("train_loss", "Training Loss over Epochs", "Loss", "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_size_experiments/Training_Loss_2.png")
    plot_metric("valid_loss", "Validation Loss over Epochs", "Loss", "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/chunk_size_experiments/Validation_Loss_2.png")


logdir_16_frames_1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_16_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_2_2024-03-20_18-24-49"
logdir_16_frames_2 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_2_continuationfromepoch17_2024-03-27_15-33-53"

logdir_32_frames_1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_2024-03-18_00-31-11"
logdir_32_frames_2 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_continuation_from_epoch_15_full_ds_2024-03-22_13-39-29"

logdir_4_frames_1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_4_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_latest_2024-04-02_12-48-07"

logdir_8_frames_1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_8_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_3_2024-03-30_12-37-02"

# plot_graphs_and_join_epochs(logdir_16_frames_1, logdir_16_frames_2, logdir_32_frames_1, logdir_32_frames_2, logdir_8_frames_1, logdir_4_frames_1)

logdir1 = "/vol/research/DeepFakeDet/rPPG-Toolbox-ais/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_6_nooverlap_outputs_latest_2_2024-04-10_17-23-30"

logdir2 = "/vol/research/DeepFakeDet/rPPG-Toolbox-ais/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_overlap_skip_4_frames_2024-03-15_18-00-35"

# logdir3a = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_2024-03-18_00-31-11"
# logdir3b = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/PhysNet/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_continuation_from_epoch_15_full_ds_2024-03-22_13-39-29"

logdir3a="/vol/"
logdir3b=""


def plot_MMP_graphs(logdir1,logdir2,logdir3a, logdir3b):
    train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32 = [], [], [], []
    train_accuracies_nonoverlapping, valid_accuracies_nonoverlapping, train_loss_nonoverlapping, valid_loss_nonoverlapping = [], [], [], []
    train_accuracies_overlapping, valid_accuracies_overlapping, train_loss_overlapping, valid_loss_overlapping = [], [], [], []


    training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32 = [], [], [], []
    training_accuracies_epochs_list_nonoverlapping, validation_accuracies_epochs_list_nonoverlapping, training_loss_epochs_list_nonoverlapping, validation_loss_epochs_list_nonoverlapping = [], [], [], []
    training_accuracies_epochs_list_overlapping, validation_accuracies_epochs_list_overlapping, training_loss_epochs_list_overlapping, validation_loss_epochs_list_overlapping = [], [], [], []

    # function to process directories and update epochs and accuracies lists
    def process_directory(logdir, training_accuracies_epochs_list, validation_accuracies_epochs_list, training_loss_epochs_list, validation_loss_epochs_list, train_accuracies_list, valid_accuracies_list, train_loss_list, valid_loss_list, initial_epoch=0):
        for file in os.listdir(logdir):
            if file.startswith("events.out.tfevents"):
                filepath = os.path.join(logdir, file)
                for e in tf.compat.v1.train.summary_iterator(filepath):
                    for v in e.summary.value:
                        print("Value, ", v)
                        if v.tag == 'Binary Accuracy/train':
                            adjusted_epoch = e.step + initial_epoch
                            training_accuracies_epochs_list.append(adjusted_epoch)
                            train_accuracies_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Accuracy: {v.simple_value}")
                        if v.tag == 'Binary Accuracy/validation':
                            adjusted_epoch = e.step + initial_epoch
                            validation_accuracies_epochs_list.append(adjusted_epoch)
                            valid_accuracies_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Accuracy: {v.simple_value}")
                        if v.tag == 'Total Loss/train':
                            adjusted_epoch = e.step + initial_epoch
                            training_loss_epochs_list.append(adjusted_epoch)
                            train_loss_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Loss: {v.simple_value}")
                        if v.tag == 'Total Loss/validation':
                            adjusted_epoch = e.step + initial_epoch
                            validation_loss_epochs_list.append(adjusted_epoch)
                            valid_loss_list.append(v.simple_value)
                            print(f"Directory: {logdir}, Epoch: {adjusted_epoch}, Loss: {v.simple_value}")

    # process the nonoverlapping frame directories
    process_directory(logdir1, training_accuracies_epochs_list_nonoverlapping, validation_accuracies_epochs_list_nonoverlapping, training_loss_epochs_list_nonoverlapping, validation_loss_epochs_list_nonoverlapping, train_accuracies_nonoverlapping, valid_accuracies_nonoverlapping, train_loss_nonoverlapping, valid_loss_nonoverlapping)

    # process the overlapping frame directories
    process_directory(logdir2, training_accuracies_epochs_list_overlapping, validation_accuracies_epochs_list_overlapping, training_loss_epochs_list_overlapping, validation_loss_epochs_list_overlapping, train_accuracies_overlapping, valid_accuracies_overlapping, train_loss_overlapping, valid_loss_overlapping)

    # process the 32 frame directories
    process_directory(logdir3a, training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32, train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32)
    last_epoch_32 = max(training_accuracies_epochs_list_32) if training_accuracies_epochs_list_32 else 0

    process_directory(logdir3b, training_accuracies_epochs_list_32, validation_accuracies_epochs_list_32, training_loss_epochs_list_32, validation_loss_epochs_list_32, train_accuracies_32, valid_accuracies_32, train_loss_32, valid_loss_32, initial_epoch=last_epoch_32 + 1)

    # plot training accuracy
    plt.figure(figsize=[8,6])
    plt.plot(training_accuracies_epochs_list_nonoverlapping, train_accuracies_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(training_accuracies_epochs_list_overlapping, train_accuracies_overlapping, label='Overlapping Chunks')
    plt.plot(training_accuracies_epochs_list_32, train_accuracies_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Training Accuracy over Epochs for non-overlapping vs overlapping chunks and augmentation')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_Accuracy.png")

    # plot validation accuracy
    plt.figure(figsize=[8,6])
    plt.plot(validation_accuracies_epochs_list_nonoverlapping, valid_accuracies_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(validation_accuracies_epochs_list_overlapping, valid_accuracies_overlapping, label='Overlapping Chunks')
    plt.plot(validation_accuracies_epochs_list_32, valid_accuracies_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Validation Accuracy over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Valid_Accuracy.png")

    # plot training loss
    plt.figure(figsize=[8,6])
    plt.plot(training_loss_epochs_list_nonoverlapping, train_loss_nonoverlapping, label='Non-Overlapping Chunks (R&C)')
    plt.plot(training_loss_epochs_list_overlapping, train_loss_overlapping, label='Overlapping Chunks')
    plt.plot(training_loss_epochs_list_32, train_loss_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Training Loss over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_Loss.png")

    # plot validation loss
    plt.figure(figsize=[8,6])
    plt.plot(validation_loss_epochs_list_nonoverlapping, valid_loss_nonoverlapping, label='Non-overlapping Chunks')
    plt.plot(validation_loss_epochs_list_overlapping, valid_loss_overlapping, label='Overlapping Chunks')
    plt.plot(validation_loss_epochs_list_32, valid_loss_32, label='Overlapping Chunks (R&C)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Validation Loss over Epochs for 16 and 32 Frames')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig("/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Validation_Loss.png")


def plot_singular_graphs(logdirtransformer):
    acc_epoch_train=[]
    train_accuracies_list=[]
    acc_epoch_val=[]
    valid_accuracies_list=[]
    loss_epoch_train=[]
    train_loss_list=[]
    loss_epoch_val=[]
    valid_loss_list=[]

    for file in os.listdir(logdirtransformer):
                if file.startswith("events.out.tfevents"):
                    filepath = os.path.join(logdirtransformer, file)
                    for e in tf.compat.v1.train.summary_iterator(filepath):
                        for v in e.summary.value:
                            print("Value, ", v)
                            if v.tag == 'Acc/train':
                                acc_epoch_train.append(e.step)
                                train_accuracies_list.append(v.simple_value)
                                print(f"Directory: {logdirtransformer}, Epoch: {acc_epoch_train}, Accuracy: {v.simple_value}")
                            if v.tag == 'Acc/val':
                                acc_epoch_val.append(e.step)
                                valid_accuracies_list.append(v.simple_value)
                                print(f"Directory: {logdirtransformer}, Epoch: {acc_epoch_val}, Accuracy: {v.simple_value}")
                            if v.tag == 'Loss/train':
                                loss_epoch_train.append(e.step)
                                train_loss_list.append(v.simple_value)
                                print(f"Directory: {logdirtransformer}, Epoch: {loss_epoch_train}, Loss: {v.simple_value}")
                            if v.tag == 'Loss/val':
                                loss_epoch_val.append(e.step)
                                valid_loss_list.append(v.simple_value)
                                print(f"Directory: {logdirtransformer}, Epoch: {loss_epoch_val}, Loss: {v.simple_value}")
    return acc_epoch_train, train_accuracies_list, acc_epoch_val, valid_accuracies_list, loss_epoch_train, train_loss_list, loss_epoch_val, valid_loss_list


def plot_transformer_graph(epoch, list, directory, label):
    plt.figure(figsize=[8,6])
    plt.plot(epoch, list)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(label, fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig(directory)



logdir0703 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001&0.01_LRReducer_16_frames_0.7binary_0.3rPPG_scaling_multi_model_2024-04-04_23-26-28"

logdir0505LR = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_0.001_0.0001_LR_multi_model_2024-04-07_23-37-57"

logdir0505a = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_2024-04-05_00-03-34"
logdir0505b = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_continue_from_epoch_16_2024-04-11_17-36-17"

logdirtransformer="/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/best_trained_DeepFakeDet_CardioVT_NeuralTextures_3dcnn"

t_acc_epoch_train, t_train_accuracies_list, t_acc_epoch_val, t_valid_accuracies_list, t_loss_epoch_train, t_train_loss_list, t_loss_epoch_val, t_valid_loss_list = plot_singular_graphs(logdirtransformer)

plot_transformer_graph(t_acc_epoch_train, t_train_accuracies_list, "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/Transformer/Transformer_Training_Acc.png", "Training Accuracy")
plot_transformer_graph(t_acc_epoch_val, t_valid_accuracies_list, "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/Transformer/Transformer_Validation_Acc.png", "Validation Accuracy")
plot_transformer_graph(t_loss_epoch_train, t_train_loss_list, "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/Transformer/Transformer_Training_Loss.png", "Training Loss")
plot_transformer_graph(t_loss_epoch_val, t_valid_loss_list, "/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/Transformer/Transformer_Validation_Loss.png", "Validation Loss")

# plot_individual_graphs(logdir1, logdir2, logdir3a, logdir3b)

# plot_MMP_graphs(logdir0703, logdir0505LR, logdir0505a, logdir0505b)


def extract_events_data(logdir, start_epoch=0, stop_epoch=float('inf')):
    val_accuracies = []
    val_losses = []
    val_epochs_acc = []
    val_epochs_loss = []
    val_rPPG_loss = []
    val_epochs_rPPG_loss = []
    val_epochs_binary_loss = []
    val_binary_loss = []

    train_rPPG_loss=[]
    train_epochs_rPPG_loss = []
    train_binary_loss = []
    train_epochs_binary_loss = []
    train_accuracies = []
    train_losses = []
    train_epochs_acc = []
    train_epochs_loss = []

    for file in sorted(os.listdir(logdir)):
        if file.startswith("events.out.tfevents"):
            filepath = os.path.join(logdir, file)
            for e in tf.compat.v1.train.summary_iterator(filepath):
                if e.step < start_epoch or e.step > stop_epoch:
                    continue
                for v in e.summary.value:
                    if v.tag == 'Binary Accuracy/validation':
                        val_accuracies.append(v.simple_value)
                        val_epochs_acc.append(e.step)
                    elif v.tag == 'Total Loss/validation':
                        val_losses.append(v.simple_value)
                        val_epochs_loss.append(e.step)
                    elif v.tag == 'Binary Accuracy/train':
                        train_accuracies.append(v.simple_value)
                        train_epochs_acc.append(e.step)
                    elif v.tag == 'Total Loss/train':
                        train_losses.append(v.simple_value)
                        train_epochs_loss.append(e.step)
                    elif v.tag == 'rPPG Loss/train':
                        train_rPPG_loss.append(v.simple_value)
                        train_epochs_rPPG_loss.append(e.step)
                    elif v.tag == 'rPPG Loss/validation':
                        val_rPPG_loss.append(v.simple_value)
                        val_epochs_rPPG_loss.append(e.step)
                    elif v.tag == 'Binary Loss/train':
                        train_binary_loss.append(v.simple_value)
                        train_epochs_binary_loss.append(e.step)
                    elif v.tag == 'Binary Loss/validation':
                        val_binary_loss.append(v.simple_value)
                        val_epochs_binary_loss.append(e.step)
    return val_epochs_acc, val_epochs_loss, val_accuracies, val_losses, train_epochs_acc, train_epochs_loss, train_accuracies, train_losses, train_rPPG_loss, train_epochs_rPPG_loss, val_rPPG_loss, val_epochs_rPPG_loss, train_binary_loss, train_epochs_binary_loss, val_binary_loss, val_epochs_binary_loss

def extract_events_data_single(logdir):
    val_accuracies = []
    val_losses = []
    val_epochs_acc = []
    val_epochs_loss = []
    val_rPPG_loss=[]
    val_epochs_rPPG_loss=[]
    val_binary_loss=[]
    val_epochs_binary_loss=[]

    train_accuracies = []
    train_losses = []
    train_rPPG_loss=[]
    train_epochs_rPPG_loss=[]
    train_binary_loss=[]
    train_epochs_binary_loss=[]

    train_epochs_acc = []
    train_epochs_loss = []

    for file in sorted(os.listdir(logdir)):
        if file.startswith("events.out.tfevents"):
            filepath = os.path.join(logdir, file)
            for e in tf.compat.v1.train.summary_iterator(filepath):
                for v in e.summary.value:
                    if v.tag == 'Binary Accuracy/validation':
                        val_accuracies.append(v.simple_value)
                        val_epochs_acc.append(e.step)
                    elif v.tag == 'Total Loss/validation':
                        val_losses.append(v.simple_value)
                        val_epochs_loss.append(e.step)
                    elif v.tag == 'Binary Accuracy/train':
                        train_accuracies.append(v.simple_value)
                        train_epochs_acc.append(e.step)
                    elif v.tag == 'Total Loss/train':
                        train_losses.append(v.simple_value)
                        train_epochs_loss.append(e.step)
                    elif v.tag == 'rPPG Loss/train':
                        train_rPPG_loss.append(v.simple_value)
                        train_epochs_rPPG_loss.append(e.step)
                    elif v.tag == 'rPPG Loss/validation':
                        val_rPPG_loss.append(v.simple_value)
                        val_epochs_rPPG_loss.append(e.step)
                    elif v.tag == 'Binary Loss/train':
                        train_binary_loss.append(v.simple_value)
                        train_epochs_binary_loss.append(e.step)
                    elif v.tag == 'Binary Loss/validation':
                        val_binary_loss.append(v.simple_value)
                        val_epochs_binary_loss.append(e.step)
    return train_epochs_acc, train_epochs_loss, train_accuracies, train_losses, val_epochs_acc, val_epochs_loss, val_accuracies, val_losses, train_rPPG_loss, train_epochs_rPPG_loss, val_rPPG_loss, val_epochs_rPPG_loss, train_binary_loss, train_epochs_binary_loss, val_binary_loss, val_epochs_binary_loss

def plot_data(epochs_3, data_3, label_3, epochs_2, data_2, label_2, epochs_1, data_1, label_1, y_label, title, save_path):
    plt.figure(figsize=[8,6])
    plt.plot(epochs_3, data_3, label=label_3)
    plt.plot(epochs_2, data_2, label=label_2)
    plt.plot(epochs_1, data_1, label=label_1)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

# logdir1 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001&0.01_LRReducer_16_frames_0.7binary_0.3rPPG_scaling_multi_model_2024-04-04_23-26-28"
# logdir2 = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_0.001_0.0001_LR_multi_model_2024-04-07_23-37-57"
# logdir3a = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_2024-04-05_00-03-34"
# logdir3b = "/vol/research/DeepFakeDet/rPPG-Toolbox/Exp1/logs/output/MultiPhysNetModelTrainer/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_continue_from_epoch_16_2024-04-11_17-36-17"

val_epochs_acc_0505a, val_epochs_loss_0505a, val_accuracies_0505a, val_losses_0505a, train_epochs_acc_0505a, train_epochs_loss_0505a, train_accuracies_0505a, train_losses_0505a, train_rPPG_loss_0505a, train_epochs_rPPG_loss_0505a, val_rPPG_loss_0505a, val_epochs_rPPG_loss_0505a, train_binary_loss_0505a, train_epochs_binary_loss_0505a, val_binary_loss_0505a, val_epochs_binary_loss_0505a = extract_events_data(logdir0505a, start_epoch=0)

val_epochs_acc_0505b, val_epochs_loss_0505b, val_accuracies_0505b, val_losses_0505b, train_epochs_acc_0505b, train_epochs_loss_0505b, train_accuracies_0505b, train_losses_0505b, train_rPPG_loss_0505b, train_epochs_rPPG_loss_0505b, val_rPPG_loss_0505b, val_epochs_rPPG_loss_0505b, train_binary_loss_0505b, train_epochs_binary_loss_0505b, val_binary_loss_0505b, val_epochs_binary_loss_0505b = extract_events_data(logdir0505b, start_epoch=8, stop_epoch=15)

train_epochs_0703_acc, train_epochs_0703_loss, train_accuracies_0703, train_losses_0703, val_epochs_0703_acc, val_epochs_0703_loss, val_accuracies_0703, val_losses_0703, train_rPPG_loss_0703, train_epochs_rPPG_loss_0703, val_rPPG_loss_0703, val_epochs_rPPG_loss_0703, train_binary_loss_0703, train_epochs_binary_loss_0703, val_binary_loss_0703, val_epochs_binary_loss_0703 = extract_events_data_single(logdir0703)

train_epochs_0505LR_acc, train_epochs_0505LR_loss, train_accuracies_0505LR, train_losses_0505LR, val_epochs_0505LR_acc, val_epochs_0505LR_loss, val_accuracies_0505LR, val_losses_0505LR, train_rPPG_loss_0505LR, train_epochs_rPPG_loss_0505LR, val_rPPG_loss_0505LR, val_epochs_rPPG_loss_0505LR, train_binary_loss_0505LR, train_epochs_binary_loss_0505LR, val_binary_loss_0505LR, val_epochs_binary_loss_0505LR = extract_events_data_single(logdir0505LR)



# Adjust the epochs for 2
# last_epoch_3a = train_epochs_acc_3a[-1] if train_epochs_acc_3a else 0
# train_epochs_0505b_adjusted = [epoch + last_epoch_3a + 1 for epoch in train_epochs_acc_0505b]
# val_epochs_0505b_adjusted = [epoch + last_epoch_3a + 1 for epoch in val_epochs_acc_0505b]


last_epoch_3a=16
offset = last_epoch_3a + 1 - 6
train_epochs_0505b_adjusted = [epoch+offset for epoch in train_epochs_acc_0505b]
val_epochs_0505b_adjusted = [epoch+offset for epoch in val_epochs_acc_0505b]

# Combine the data for 3
train_epochs_acc_combined = train_epochs_acc_0505a + train_epochs_0505b_adjusted
train_epochs_loss_combined = train_epochs_loss_0505a + train_epochs_0505b_adjusted

train_accuracies_combined = train_accuracies_0505a + train_accuracies_0505b
train_losses_combined = train_losses_0505a + train_losses_0505b

val_epochs_acc_combined = val_epochs_acc_0505a + val_epochs_0505b_adjusted
val_epochs_loss_combined = val_epochs_loss_0505a + val_epochs_0505b_adjusted

val_accuracies_combined = val_accuracies_0505a + val_accuracies_0505b
val_losses_combined = val_losses_0505a + val_losses_0505b

train_epochs_rPPG_loss_combined = train_epochs_rPPG_loss_0505a + train_epochs_0505b_adjusted
train_epochs_binary_loss_combined = train_epochs_binary_loss_0505a + train_epochs_0505b_adjusted

train_rPPG_loss_combined = train_rPPG_loss_0505a + train_rPPG_loss_0505b
train_binary_loss_combined = train_binary_loss_0505a + train_binary_loss_0505b


val_epochs_rPPG_loss_combined = val_epochs_rPPG_loss_0505a + val_epochs_0505b_adjusted
val_epochs_binary_loss_combined = val_epochs_binary_loss_0505a + val_epochs_0505b_adjusted

val_rPPG_loss_combined = val_rPPG_loss_0505a + val_rPPG_loss_0505b
val_binary_loss_combined = val_binary_loss_0505a + val_binary_loss_0505b

# Plot the combined data
plot_data(train_epochs_acc_combined, train_accuracies_combined, 'Experiment 1', train_epochs_0703_acc, train_accuracies_0703,'Experiment 2', train_epochs_0505LR_acc, train_accuracies_0505LR, 'Experiment 3',
          'Training Accuracy', 'Accuracy over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_Accuracy.png')

plot_data(val_epochs_acc_combined, val_accuracies_combined, 'Experiment 1', val_epochs_0703_acc, val_accuracies_0703, 'Experiment 2', val_epochs_0505LR_acc, val_accuracies_0505LR, 'Experiment 3',
          'Training Accuracy', 'Accuracy over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Validation_Accuracy.png')

plot_data(train_epochs_loss_combined, train_losses_combined, 'Experiment 1', train_epochs_0703_loss, train_losses_0703, 'Experiment 2', train_epochs_0505LR_loss, train_losses_0505LR, 'Experiment 3',
          'Training Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_Loss.png')

plot_data(val_epochs_loss_combined, val_losses_combined, 'Experiment 1', val_epochs_0703_loss, val_losses_0703, 'Experiment 2',val_epochs_0505LR_loss, val_losses_0505LR, 'Experiment 3',
          'Validation Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Validation_Loss.png')

print("train_epochs_rPPG_loss_combined", train_epochs_rPPG_loss_combined)
print("train_rPPG_loss_combined", train_rPPG_loss_combined)
# val and train rppg loss
plot_data(train_epochs_rPPG_loss_combined, train_rPPG_loss_combined, 'Experiment 1', train_epochs_rPPG_loss_0703, train_rPPG_loss_0703, 'Experiment 2',train_epochs_rPPG_loss_0505LR, train_rPPG_loss_0505LR, 'Experiment 3',
          'Validation Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_rPPG_Loss.png')

plot_data(val_epochs_rPPG_loss_combined, val_rPPG_loss_combined, 'Experiment 1', val_epochs_rPPG_loss_0703, val_rPPG_loss_0703, 'Experiment 2',val_epochs_rPPG_loss_0505LR, val_rPPG_loss_0505LR, 'Experiment 3',
          'Validation Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Validation_rPPG_Loss.png')

# binary loss
plot_data(train_epochs_binary_loss_combined, train_binary_loss_combined, 'Experiment 1', train_epochs_binary_loss_0703, train_binary_loss_0703, 'Experiment 2', train_epochs_binary_loss_0505LR, train_binary_loss_0505LR, 'Experiment 3', 'Validation Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Training_Binary_Loss.png')

plot_data(val_epochs_binary_loss_combined, val_binary_loss_combined, 'Experiment 1', val_epochs_binary_loss_0703, val_binary_loss_0703, 'Experiment 2',val_epochs_binary_loss_0505LR, val_binary_loss_0505LR, 'Experiment 3',
          'Validation Loss', 'Loss over Epochs',
          '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/MMP_Validation_Binary_Loss.png')



# # Extract data from log directories
# epochs_val_3_a, accuracies_val_3_a, losses_val_3_a, epochs_train_3_a, accuracies_train_3_a, losses_train_3_a = extract_events_data(logdir0505a)
# epochs_b, accuracies_b, losses_b = extract_events_data(logdir0505b)

# # Adjust epochs in logdir0505b to continue from logdir3a
# last_epoch_3a = epochs_val_3_a[-1] if epochs_val_3_a else 0

# epochs_b_adjusted = [epoch + last_epoch_3a + 1 for epoch in epochs_b]

# # Combine data from both directories
# combined_epochs = epochs_a + epochs_b_adjusted
# combined_accuracies = accuracies_a + accuracies_b
# combined_losses = losses_a + losses_b

# Plot combined data
# plot_data(combined_epochs, combined_accuracies, combined_losses, label='Combined Logdir3', save_path='/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/Training_Accuracy_2.png')
# plot_data(combined_epochs, combined_accuracies, combined_losses, label='Combined Logdir3', save_path='/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/Validation_Accuracy_2.png')
# plot_data(combined_epochs, combined_accuracies, combined_losses, label='Combined Logdir3', save_path='/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/Training_Loss_2.png')
# plot_data(combined_epochs, combined_accuracies, combined_losses, label='Combined Logdir3', save_path='/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/MMP/Validation_Loss_2.png')
