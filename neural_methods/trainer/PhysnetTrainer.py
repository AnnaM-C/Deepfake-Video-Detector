# Refactored from Code Base by https://github.com/ubicomplab/rPPG-Toolbox
"""PhysNet Trainer for deepfake classification"""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from Utils.utils import calculate_accuracy, get_missclassified_samples, save_misclassified_samples, save_predictions
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class PhysnetTrainer(BaseTrainer):

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
        self.patience = 10
        self.wait = 0
        self.early_stop = False
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = torch.nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        
            # pretrained_weights = torch.load("/vol/research/DeepFakeDet/rPPG-Toolbox-ais/runs/exp/logs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_overlap_skip_2_noaugmentation_latest_2/PreTrainedModels/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_overlap_skip_2_noaugmentation_latest_2_Epoch10.pth")
            # if pretrained_weights:
            #     self.model.load_state_dict(pretrained_weights, strict=False)
            #     print("Pretrained Loaded")
        
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")
    

    def train(self, data_loader, writer):
        print("Data loader train length in PhysNetTrainer inside train function, ", len(data_loader['train']))
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_accuracy=0.0
            running_loss = 0.0
            running_accuracy = 0.0
            train_loss = []
            train_true=[]
            train_scores=[]

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                output = self.model(batch[0].to(torch.float32).to(self.device))
                labels = batch[1].to(self.device)
                labels = labels.long()
                loss = self.loss_model(output, labels)
                loss.backward()
                running_loss += loss.item()

                proba = torch.softmax(output, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)

                running_accuracy += calculate_accuracy(pred_labels, labels)
                if idx % 100 == 99:
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                #TODO: amend this for StepLR
                # lrs.append(self.scheduler.get_last_lr())
                lrs.append(self.optimizer.param_groups[0]['lr'])

                proba = torch.softmax(output, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)
                running_accuracy += calculate_accuracy(pred_labels, labels)

                print("Running Acc, ", running_accuracy)
                print("Length dataloader, ", len(data_loader["train"]))


                self.optimizer.step()
                #TODO: uncomment for StepLR
                # self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())


                score = proba[:, 1].cpu().detach().numpy() 
                train_true.extend(labels.cpu().numpy())
                train_scores.extend(score)

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))
            epoch_accuracy = running_accuracy / len(data_loader["train"])
            mean_epoch_loss=np.mean(train_loss)
            writer.add_scalar("Loss/train", mean_epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss, valid_accuracy = self.valid(data_loader)
                writer.add_scalar("Loss/validation", valid_loss, epoch)
                writer.add_scalar("Accuracy/validation", valid_accuracy, epoch)
                mean_valid_losses.append(valid_loss)
                # this is logic to update the minimum valid loss if the valid loss for that epoch is less than the min valid loss
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    #NOTE: recently added for early stopping
                    self.wait = 0
                    #TODO: remove for LRStep
                    self.save_model(epoch)
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                #NOTE: recently added for early stopping
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print("Training stopped early, triggered by early stopping logic.")
                        self.early_stop = True
                        break
                #TODO: remove for LRStep
                self.scheduler.step(valid_loss)

            vwriter.add_scalar('Loss/val', valid_loss, epoch+1)
            vwriter.add_scalar('Acc/val', valid_accuracy, epoch+1)
            vwriter.add_scalar('AUC/val', val_roc_auc, epoch+1)
            print(f"Epoch: {epoch} Valid accuracy: {valid_accuracy}")

        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
        writer.close()


    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        valid_accuracy = []
        self.model.eval()
        valid_step = 0
        running_val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                output = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                label = valid_batch[1].to(self.device)
                label = label.long()

                loss_ecg = self.loss_model(output, label)
                proba = torch.softmax(output, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)

                acc = calculate_accuracy(pred_labels, label)
                valid_accuracy.append(acc)

                valid_loss.append(loss_ecg.item())
                valid_acc.append(acc)
                running_val_loss += loss_ecg.item()
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
                scores = proba[:, 1].cpu().detach().numpy()  # probability of positive class
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(scores)

            print("Losses list for all valid batches, ", valid_loss)
            print("running_val_loss/len(dataloader['valid']), ", running_val_loss/len(data_loader['valid']))
            valid_loss = np.asarray(valid_loss)
            valid_accuracy = np.asarray(valid_accuracy)
        return np.mean(valid_loss), np.mean(valid_accuracy)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        # labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
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
        test_scores = []
        test_true = []
        test_preds = []
        test_acc = []
        test_filepaths=[]
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels, file_path = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[3]
                labels = labels.long()
                # pred_ppg_test, _, _, _ = self.model(data)
                prediction = self.model(data)
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
        print("Misclassified samples, ", misclassified_samples)
        save_misclassified_samples(misclassified_samples, self.config)
        
        #Save predictions
        save_predictions(test_preds, test_true, test_filepaths, self.config.TRAIN.MODEL_FILE_NAME)

        # save for binary classification
        if self.config.TEST.OUTPUT_SAVE_DIR:
            # save
            print("Test accuracy, ", test_acc)
            print("Test roc, ", test_roc_auc)
            self.save_test_metrics(test_acc, test_roc_auc, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
