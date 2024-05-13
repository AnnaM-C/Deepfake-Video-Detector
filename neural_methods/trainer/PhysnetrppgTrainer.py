"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.Physnetrppg import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import csv

class PhysnetrppgTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "get_rPPG":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def get_rPPG(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "get_rPPG":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        csv_data = [["Batch Index", "frame_path", "rPPG_values"]]
        '''
        with torch.no_grad():
            for batch_index, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                file_paths=test_batch[3]
                print("Filepath, ", file_paths)
                pred_rppg_test, _, _, _, _ = self.model(data)
                pred_rppg_test = pred_rppg_test.cpu().numpy()
                print("rPPG, ", pred_rppg_test)

                for i in range(batch_size):
                    row = [
                        batch_index + 1,
                        file_paths[i],
                        " ".join(map(str, pred_rppg_test[i]))
                    ]
                    csv_data.append(row)
        
        csv_file_path = 'rppgpredictions_testset2.csv'
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print(f"Saved all batch predictions to {csv_file_path}.")
        print('')
        '''
        txt_file_path = 'rppgpredictions_16frames_trainset2.txt'
        with torch.no_grad():
        # Open the text file for writing
            with open(txt_file_path, 'w') as file:
                for batch_index, test_batch in enumerate(tqdm(data_loader["train"], ncols=80)):
                    batch_size = test_batch[0].shape[0]
                    data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                    file_paths = test_batch[3]
                    # print("Filepath, ", file_paths)
                    pred_rppg_test, _, _, _, _ = self.model(data)
                    pred_rppg_test = pred_rppg_test.cpu().numpy()
                    # print("rPPG, ", pred_rppg_test)

                    for i in range(batch_size):
                        # Create the line to be written for each prediction
                        line = f"{batch_index + 1}, {file_paths[i]}, {' '.join(map(str, pred_rppg_test[i]))}\n"
                        # Write the line to the file
                        file.write(line)

        print(f"Saved all batch predictions to {txt_file_path}.")

    @staticmethod
    def save_rppg_to_file(matched_rppg, output_file):
        if not os.path.isdir(output_file):
            with open(output_file, 'w') as file:
                for batch in matched_rppg.values():
                    for frame, rppg_value in batch.items():
                        file.write(f"{frame} {rppg_value}\n")
        else:
            print(f"Cannot write to {output_file}: It is a directory")

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)