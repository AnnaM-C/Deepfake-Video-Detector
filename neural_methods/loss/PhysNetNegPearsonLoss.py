from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn

class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, preds, labels):
        loss = torch.tensor(0.0, device=preds.device)
        epsilon = 1e-8 
        for i in range(preds.shape[0]):
            std_preds = torch.std(preds[i])
            std_labels = torch.std(labels[i])
            if std_preds <= epsilon or std_labels <= epsilon:
                sample_loss = torch.tensor(0.0, device=preds.device)
            else:
                sum_x = torch.sum(preds[i])
                sum_y = torch.sum(labels[i])
                sum_xy = torch.sum(preds[i] * labels[i])
                sum_x2 = torch.sum(torch.pow(preds[i], 2))
                sum_y2 = torch.sum(torch.pow(labels[i], 2))
                N = preds.shape[1]

                denominator = torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))) + epsilon
                pearson = (N * sum_xy - sum_x * sum_y) / denominator
                
                sample_loss = 1 - pearson

            loss += sample_loss

        loss = loss / preds.shape[0]
        return loss

