# https://github.com/dougbrion/pytorch-classification-uncertainty/blob/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(args):
    if args.task == "classification":
        if args.classifier_type.endswith("2"):
            criterion = nn.CrossEntropyLoss() #has already included a softmax layer inside.
                # target: Ground truth class indices or class probabilities;
        else:
            criterion = nn.BCEWithLogitsLoss()
            
    elif args.task == "regression":
        criterion = nn.MSELoss()
    return criterion

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
