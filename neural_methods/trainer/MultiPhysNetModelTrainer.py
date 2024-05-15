# Adapted from Code Base by https://github.com/ubicomplab/rPPG-Toolbox
"""MultiModel Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.optim import SGD
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.Physnetrppg import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.MultiPhysNetModel import MultiPhysNetModel
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from Utils.utils import calculate_accuracy, get_missclassified_samples, save_misclassified_samples, save_predictions
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson

class MultiPhysNetModelTrainer(BaseTrainer):

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
        self.min_valid_loss_total = None
        self.best_epoch = 0

        self.physnet_model1 = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        
        self.physnet_model2 = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        
        self.multi_model = MultiPhysNetModel(physnet_model1=self.physnet_model1, physnet_model2=self.physnet_model2, mlp1_output_dim=config.MODEL.PHYSNET.FRAME_NUM, mlp2_output_dim=2).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])

            self.loss_rPPG = Neg_Pearson()
            self.loss_binary = torch.nn.CrossEntropyLoss()

            physnet_lr_1 = 1e-3
            physnet_lr_2 = 1e-4
            mlp_lr_1 = 1e-2
            mlp_lr_2 = 1e-3

            # create parameter groups
            param_groups = [
                {'params': self.multi_model.physnet1.parameters(), 'lr': physnet_lr_1},
                {'params': self.multi_model.physnet2.parameters(), 'lr': physnet_lr_2},
                {'params': self.multi_model.mlp_head1.parameters(), 'lr': mlp_lr_1},
                {'params': self.multi_model.mlp_head2.parameters(), 'lr': mlp_lr_2}
            ]

            self.optimizer = optim.Adam(param_groups)


            # # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
    
            # pretrained_weights = torch.load("runs/exp/logs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model/PreTrainedModels/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_Epoch16.pth")
            # if pretrained_weights:
            #     self.multi_model.load_state_dict(pretrained_weights, strict=False)
            #     print("Pretrained Loaded")
        
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader, writer):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses_total = []
        mean_training_losses_rPPG = []
        mean_training_losses_binary = []

        mean_valid_losses_total = []
        mean_valid_losses_rPPG = []
        mean_valid_losses_binary = []

        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_accuracy_binary=0.0

            running_loss_rPPG=0.0
            running_loss_binary=0.0
            running_loss_total=0.0
            train_loss_total=[]
            train_loss_rPPG=[]
            train_loss_binary=[]

            self.multi_model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                norm_frames=batch[0]
                diff_frames=batch[4]

                output_rPPG, output_binary = self.multi_model(diff_frames.to(torch.float32).to(self.device), norm_frames.to(torch.float32).to(self.device))

                labels_rPPG = batch[5].to(self.device)

                labels_binary = batch[1].to(self.device)

                loss_rPPG = self.loss_rPPG(output_rPPG, labels_rPPG)
                print(f"Index {idx} and loss_rPPG {loss_rPPG}")

                loss_binary = self.loss_binary(output_binary, labels_binary)

                # scale loss functions
                total_loss=loss_rPPG*0.7 + loss_binary*0.3
                total_loss.backward()

                running_loss_rPPG += loss_rPPG.item()
                running_loss_binary += loss_binary.item()
                running_loss_total += total_loss.item()

                proba = torch.softmax(output_binary, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)

                running_accuracy_binary += calculate_accuracy(pred_labels, labels_binary)

                if idx % 100 == 99:
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss_total / 100:.3f}')
                    running_loss_total = 0.0

                train_loss_total.append(total_loss.item())
                train_loss_rPPG.append(loss_rPPG.item())
                train_loss_binary.append(loss_binary.item())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                tbar.set_postfix(loss=loss_rPPG.item())
                tbar.set_postfix(loss=loss_binary.item())

            mean_training_losses_total.append(np.mean(train_loss_total))
            mean_training_losses_rPPG.append(np.mean(train_loss_rPPG))
            mean_training_losses_binary.append(np.mean(train_loss_binary))

            epoch_accuracy_binary = running_accuracy_binary / len(data_loader["train"])

            mean_epoch_loss_total=np.mean(train_loss_total)
            mean_epoch_loss_rPPG=np.mean(train_loss_rPPG)
            mean_epoch_loss_binary=np.mean(train_loss_binary)


            writer.add_scalar("Total Loss/train", mean_epoch_loss_total, epoch)
            writer.add_scalar("rPPG Loss/train", mean_epoch_loss_rPPG, epoch)
            writer.add_scalar("Binary Loss/train", mean_epoch_loss_binary, epoch)
            writer.add_scalar("Binary Accuracy/train", epoch_accuracy_binary, epoch)

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss_total, valid_accuracy_binary, valid_loss_rPPG, valid_loss_binary = self.valid(data_loader)
                writer.add_scalar("Total Loss/validation", valid_loss_total, epoch)
                writer.add_scalar("Binary Accuracy/validation", valid_accuracy_binary, epoch)
                writer.add_scalar("rPPG Loss/validation", valid_loss_rPPG, epoch)
                writer.add_scalar("Binary Loss/validation", valid_loss_binary, epoch)

                mean_valid_losses_total.append(valid_loss_total)
                print('validation loss: ', valid_loss_total)
                if self.min_valid_loss_total is None:
                    self.min_valid_loss_total = valid_loss_total
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss_total < self.min_valid_loss_total):
                    self.min_valid_loss_total = valid_loss_total
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss_total))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses_total, mean_valid_losses_total, lrs, self.config)
        writer.close()

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss_total = []
        valid_loss_rPPG = []
        valid_loss_binary = []

        valid_accuracy_binary = []
        self.multi_model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                norm_frames=valid_batch[0]
                diff_frames=valid_batch[4]
                output_rPPG, output_binary = self.multi_model(diff_frames.to(torch.float32).to(self.device), norm_frames.to(torch.float32).to(self.device))

                labels_rPPG = valid_batch[5].to(self.device)

                labels_binary = valid_batch[1].to(self.device)

                loss_rPPG = self.loss_rPPG(output_rPPG, labels_rPPG)
                loss_binary = self.loss_binary(output_binary, labels_binary)

                total_loss=loss_rPPG*0.8 + loss_binary*0.5


                proba = torch.softmax(output_binary, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)

                acc_binary = calculate_accuracy(pred_labels, labels_binary)
                valid_accuracy_binary.append(acc_binary)

                valid_loss_total.append(total_loss.item())
                valid_loss_rPPG.append(loss_rPPG.item())
                valid_loss_binary.append(loss_binary.item())

                valid_step += 1
                vbar.set_postfix(loss=total_loss.item())

            valid_loss_total = np.asarray(valid_loss_total)
            valid_loss_rPPG = np.asarray(valid_loss_rPPG)
            valid_loss_binary = np.asarray(valid_loss_binary)

            valid_accuracy_binary = np.asarray(valid_accuracy_binary)
        return np.mean(valid_loss_total), np.mean(valid_accuracy_binary), np.mean(valid_loss_rPPG), np.mean(valid_loss_binary)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            print("model path, ", self.config.INFERENCE.MODEL_PATH)
            self.multi_model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.multi_model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.multi_model.load_state_dict(torch.load(best_model_path))

        self.multi_model = self.multi_model.to(self.config.DEVICE)
        test_scores = []
        test_true = []
        test_preds = []
        test_acc = []
        test_filepaths=[]
        self.multi_model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels, file_path = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[3]
                labels = labels.long()

                norm_frames=test_batch[0]
                diff_frames=test_batch[4]

                _, prediction = self.multi_model(diff_frames.to(torch.float32).to(self.device), norm_frames.to(torch.float32).to(self.device))

                proba = torch.softmax(prediction, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)
                acc = calculate_accuracy(pred_labels, labels)
                test_acc.append(acc)

                scores = proba[:, 1].cpu().detach().numpy()
                test_true.extend(labels.cpu().numpy())
                test_scores.extend(scores)
                test_preds.extend(pred_labels)
                test_filepaths.extend(file_path)
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels = labels.cpu()

        test_acc = np.asarray(test_acc)
        test_acc = np.mean(test_acc)
        fpr, tpr, thresholds = metrics.roc_curve(test_true, test_scores)
        test_roc_auc = metrics.auc(fpr, tpr)
            
        # confusion matrix
        cm = confusion_matrix(test_true, test_preds)
        classes = ['real', 'fake']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
         # confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix for Test Set')
        output_dir=f'Exp1/cmatrix/test/{self.config.TRAIN.MODEL_FILE_NAME}_test_set_{self.config.TEST.DATA.DATASET}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the plot to the specified directory
        figure_path = f'{output_dir}/{self.config.TRAIN.MODEL_FILE_NAME}_{self.config.TEST.DATA.DATASET}_confusion_matrix_test_set.png'
        plt.savefig(figure_path)
        plt.close(fig) 
        print('')

        misclassified_samples=get_missclassified_samples(test_preds, test_true, test_filepaths)
        save_misclassified_samples(misclassified_samples, self.config)

       # save predictions
        save_predictions(test_preds, test_true, test_filepaths, self.config.TRAIN.MODEL_FILE_NAME)

        if self.config.TEST.OUTPUT_SAVE_DIR:
            print("Test accuracy, ", test_acc)
            print("Test roc, ", test_roc_auc)
            self.save_test_metrics(test_acc, test_roc_auc, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.multi_model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
