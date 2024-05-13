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

        #NOTE: define multimodel loss functions and optimiser, same for both
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])

            self.loss_rPPG = Neg_Pearson()
            self.loss_binary = torch.nn.CrossEntropyLoss()

            physnet_lr_1 = 1e-3
            physnet_lr_2 = 1e-4
            mlp_lr_1 = 1e-2
            mlp_lr_2 = 1e-3

            # Create parameter groups
            param_groups = [
                {'params': self.multi_model.physnet1.parameters(), 'lr': physnet_lr_1},
                {'params': self.multi_model.physnet2.parameters(), 'lr': physnet_lr_2},
                {'params': self.multi_model.mlp_head1.parameters(), 'lr': mlp_lr_1},
                {'params': self.multi_model.mlp_head2.parameters(), 'lr': mlp_lr_2}
            ]

            # self.optimizer = optim.Adam(
            #     self.multi_model.parameters(), lr=config.TRAIN.LR)
            momentum = 0.9
            weight_decay = 0.00001
            # self.optimizer = SGD(self.multi_model.parameters(), lr=config.TRAIN.LR, momentum=momentum, weight_decay=weight_decay)
            self.optimizer = optim.Adam(param_groups)


            # # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        
            # Define optimizer and scheduler
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, min_lr=0.0000000009, verbose=True)

            pretrained_weights = torch.load("runs/exp/logs/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model/PreTrainedModels/PhysNet_NeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_Epoch16.pth")
            if pretrained_weights:
                self.multi_model.load_state_dict(pretrained_weights, strict=False)
                print("Pretrained Loaded")
        
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader, writer):
        print("Data loader train length in PhysNetTrainer inside train function, ", len(data_loader['train']))

        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses_total = []
        mean_training_losses_rPPG = []
        mean_training_losses_binary = []

        # mean_valid_losses = []
        mean_valid_losses_total = []
        mean_valid_losses_rPPG = []
        mean_valid_losses_binary = []

        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_accuracy_binary=0.0
            # running_loss = 0.0
            # train_loss = []

            #TODO: define new running loss's, accuracies, total losses
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

                # epsilon = 1e-8
                # output_rPPG = (output_rPPG - torch.mean(output_rPPG)) / torch.std(output_rPPG)
                # output_rPPG_mean = torch.mean(output_rPPG)
                # output_rPPG_std = torch.std(output_rPPG) + epsilon
                # output_rPPG = (output_rPPG - output_rPPG_mean) / output_rPPG_std

                #TODO: normalise the rppg predictions to match the output
                labels_rPPG = batch[5].to(self.device)

                # labels_rPPG = (labels_rPPG - torch.mean(labels_rPPG)) / torch.std(labels_rPPG)
                # labels_rPPG_mean = torch.mean(labels_rPPG)
                # labels_rPPG_std = torch.std(labels_rPPG) + epsilon
                # labels_rPPG = (labels_rPPG - labels_rPPG_mean) / labels_rPPG_std

                labels_binary = batch[1].to(self.device)

                # print("rPPG pred, ", output_rPPG)
                # print("rPPG gt  , ", labels_rPPG)
                # print("binary pred, ", output_binary)
                # print("binary gt  , ", labels_binary)
                # print("fake/real pred, ", labels_binary.shape)

                loss_rPPG = self.loss_rPPG(output_rPPG, labels_rPPG)
                print(f"Index {idx} and loss_rPPG {loss_rPPG}")

                loss_binary = self.loss_binary(output_binary, labels_binary)

                # print("rPPG loss, ", loss_rPPG)
                # print("binary loss, ", loss_binary)

                total_loss=loss_rPPG*0.7 + loss_binary*0.3
                # print("Total loss, ", total_loss)
                # TODO: scale total_loss
                # record value of losses
                # scale the weights by makings ure the loss isnt too high. find out which one of the loss you can reduce
                # and by how much. if you reduce it too much it causes the model to not learn and causes an impact of the 
                # learning for that task. half each one? start with a 
                # stop at 5 epochs to see what losses you are getting, total, rPPG and binary.
                # reduce the bad loss and write the report base on this. Don't spend too much time on this
                # use preloaded model use original images
                # save the single model -> cant load pretrained 
                total_loss.backward()

                running_loss_rPPG += loss_rPPG.item()
                running_loss_binary += loss_binary.item()
                running_loss_total += total_loss.item()

                #TODO: rPPG output uncomment the below and fix
                proba = torch.softmax(output_binary, dim=1)
                pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)

                running_accuracy_binary += calculate_accuracy(pred_labels, labels_binary)

                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss_total / 100:.3f}')
                    running_loss_total = 0.0

                train_loss_total.append(total_loss.item())
                train_loss_rPPG.append(loss_rPPG.item())
                train_loss_binary.append(loss_binary.item())

                # # Append the current learning rate to the list
                # lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                tbar.set_postfix(loss=loss_rPPG.item())
                tbar.set_postfix(loss=loss_binary.item())

            # TODO: figure out losses
            # Append the mean training loss for the epoch
            mean_training_losses_total.append(np.mean(train_loss_total))
            mean_training_losses_rPPG.append(np.mean(train_loss_rPPG))
            mean_training_losses_binary.append(np.mean(train_loss_binary))

            epoch_accuracy_binary = running_accuracy_binary / len(data_loader["train"])

            mean_epoch_loss_total=np.mean(train_loss_total)
            mean_epoch_loss_rPPG=np.mean(train_loss_rPPG)
            mean_epoch_loss_binary=np.mean(train_loss_binary)


            writer.add_scalar("Total Loss/train", mean_epoch_loss_total, epoch)
            writer.add_scalar("rPPG Loss/train", mean_epoch_loss_rPPG, epoch)
            # writer.add_scalar("Classification Loss/train", mean_epoch_loss_binary, epoch)
            writer.add_scalar("Binary Loss/train", mean_epoch_loss_binary, epoch)
            # writer.add_scalar("Accuracy/train", epoch_accuracy_binary, epoch)
            writer.add_scalar("Binary Accuracy/train", epoch_accuracy_binary, epoch)

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss_total, valid_accuracy_binary, valid_loss_rPPG, valid_loss_binary = self.valid(data_loader)
                writer.add_scalar("Total Loss/validation", valid_loss_total, epoch)
                # writer.add_scalar("Classification Accuracy/validation", valid_accuracy_binary, epoch)
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

    #TODO: fix for multimodel
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
                # pred_ppg_test, _, _, _ = self.model(data)

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
                # print("File path, ",len(file_path))
                # print("gt, ", len(labels) )
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels = labels.cpu()

        test_acc = np.asarray(test_acc)
        test_acc = np.mean(test_acc)
        fpr, tpr, thresholds = metrics.roc_curve(test_true, test_scores)
        test_roc_auc = metrics.auc(fpr, tpr)

                # for idx in range(batch_size):
                #     subj_index = test_batch[2][idx]
                #     sort_index = int(test_batch[3][idx])
                #     if subj_index not in predictions.keys():
                #         predictions[subj_index] = dict()
                #         labels[subj_index] = dict()
                #     predictions[subj_index][sort_index] = pred_ppg_test[idx]
                #     labels[subj_index][sort_index] = label[idx]
            
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
        plt.close(fig)  # Close the figure to free memory
        print('')
        # calculate_metrics(predictions, labels, self.config)

        misclassified_samples=get_missclassified_samples(test_preds, test_true, test_filepaths)
        save_misclassified_samples(misclassified_samples, self.config)

       #Save predictions
        save_predictions(test_preds, test_true, test_filepaths, self.config.TRAIN.MODEL_FILE_NAME)

        # save for binary classification
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            # self.save_test_outputs(predictions, labels, self.config)
            # save
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
