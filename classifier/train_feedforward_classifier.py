import os
import argparse
import csv
import ast
import re
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers.activations import ACT2FN

cwd = os.getcwd()
cwd = "/".join(cwd.split("/")[:-1])
import sys
sys.path.append(cwd)
from losses import get_criterion
from src.utils import init_model, process_layers_to_process
sys.path.append(f"{cwd}/generate")
from save_hidden_states import get_batch_hidden_states

sys.path.append(f"{cwd}/internal_information")
from perceiver import Perceiver_LSTMClassifier2, P_LlamaMLPClassifier2, P_LlamaMLPClassifier

import pandas as pd
from tqdm import tqdm
import json
import wandb

import random
random.seed(42)
from classifier_models import *

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler
import deepspeed 

class HiddenLayerDataset(Dataset):
    def __init__(self, input_path, all_hidden_states, layer, label, hidden_state_dims=[], ignore_missing_hs=False) -> None:
        super().__init__()
        LABELMAP = {'hallucinated': 0, "ok": 1,
                    'unseen': 0, "seen": 1,
                    "AllSides":0, "ArchivalQA":1}
        layer = int(layer)
        dataset = pd.read_csv(input_path, encoding="utf8")
        self.data = []
        
        for i, row in dataset.iterrows():
            try:
                hidden_states = all_hidden_states[row["id"]][layer]
                if len(hidden_states.shape) == 1: # mean or last (4096,)
                    if hidden_state_dims:
                        hidden_states = hidden_states[hidden_state_dims]

                    if label in ["label", "hallucinated_source", "all_source"]: # classification
                        self.data.append({"id":row["id"], 
                                        "hidden_states": hidden_states, 
                                        "label":LABELMAP[row[label]]})
                    else: # regression
                        self.data.append({"id":row["id"], 
                                        "hidden_states": hidden_states, 
                                        "label":row[label]})
                elif len(hidden_states.shape) == 2: # each (seq_len, 4096)
                    if hidden_state_dims:
                        all_token_hidden_states = hidden_states[:, hidden_state_dims]
                    assert label in ["label", "hallucinated_source", "all_source"] # classification
                    for each_token_h in all_token_hidden_states:
                        if each_token_h.any():
                            self.data.append({"id":row["id"], 
                                            "hidden_states": each_token_h, 
                                            "label":LABELMAP[row[label]]})

                else:
                    raise ValueError(f"hidden_states shape {hidden_states.shape} error")
                
                
                
            except Exception as e:
                if ignore_missing_hs:
                    print(f"cannot find {row['id']}, {layer}")
                    continue
                else:
                    print(e)
                    print(row["id"], layer)
                    print(all_hidden_states[row["id"]])
                    print("input_path", input_path)
                    assert False
                
        print('total num of data', len(self.data))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def prepare_hidden_states(hidden_state_type, source_dirs, layers_to_process, hidden_state_model_name, batch_size, device, ignore_nan, split=['train','val','test']):
    # source_dirs is a list of dirs or pkl files
    assert type(source_dirs) == list
    print("prepare hidden_states source_dirs", source_dirs)
    time1 = time.time()
    
    all_hidden_states = {}
    if hidden_state_type != 'each':
        source_path_list = []
        for source_dir in source_dirs:
            if os.path.isfile(source_dir) and source_dir.endswith("pkl"):
                source_path_list.append(source_dir)
            else:
                for root, dirs, files in os.walk(source_dir, topdown=False):
                    for name in files:
                        if (hidden_state_type == 'last' and name == 'hidden_state.pkl') or \
                            (hidden_state_type == 'mean' and name == 'hidden_state_mean.pkl'):
                            items = root.split("/")
                            if any([f"{hidden_state_model_name}_{s}" in items for s in split]):
                                path = os.path.join(root, name)
                                source_path_list.append(path)
                                
        source_path_list = sorted(source_path_list)    
        print("source_path_list", source_path_list)
        for hidden_state_path in tqdm(source_path_list):
            # print('loading source_path', hidden_state_path)
            if os.path.isfile(hidden_state_path):
                # print('loading from existing', hidden_state_path)
                with open(hidden_state_path, 'rb') as f:
                    file_hidden_state = pickle.load(f)
                    for id, hidden_states in file_hidden_state.items():
                        layers = list(hidden_states.keys())
                        for layer in layers:
                            if layer not in layers_to_process:
                                hidden_states.pop(layer)
                    all_hidden_states.update(file_hidden_state)
            else:
                print("no exist", hidden_state_path)
                assert False
    else: # 当场生成
        assert False
        generate_model, generate_tokenizer = init_model(hidden_state_model_name, device, "left")
        generate_tokenizer.padding_side = "left"

        for source_dir in source_dirs: # csv files
            assert os.path.isfile(source_dir) and source_dir.endswith("csv")
            source_path = source_dir
            if any([s in source_path for s in split]):
                dataset = pd.read_csv(source_path, encoding="utf8")
                    
                for i in tqdm(range(0, len(dataset), batch_size)):
                    ids, questions = [], []
                    for j in range(i, min(i+batch_size, len(dataset))):
                        ids.append(dataset.loc[j, "id"])
                        questions.append(dataset.loc[j, 'question'])
                    batch_hidden_state = get_batch_hidden_states(ids, questions, generate_model, generate_tokenizer, 
                                                                layers_to_process, generate_model_name, 
                                                                hidden_state_type, save_attention=False, ignore_nan=ignore_nan)
                    all_hidden_states.update(batch_hidden_state)

    time2 = time.time()
    print("prepare hidden_states time", time2-time1)
    return all_hidden_states

def save_and_clear_checkpoint(save_dir, model, epoch):
    for file in os.listdir(save_dir):
        if file.endswith("model_weights.pth"):
            hist_e = int(file.split("_")[0])
            # if hist_e < epoch:
            os.remove(save_dir+f"/{file}")
    torch.save(model.state_dict(), save_dir+f"/{epoch}_model_weights.pth")


def load_classifier_model(args, rank=0):
    if args.classifier_type == "LlamaMLP2":
        model = LlamaMLPClassifier2(args.input_dim, args.hidden_dim)
    elif args.classifier_type == "ff":
        model = LinearClassifier(args.input_dim, args.hidden_dim)
    elif args.classifier_type == "ff1":
        model = LinearClassifier1(args.input_dim)
    elif args.classifier_type == "ff12":
        model = LinearClassifier12(args.input_dim)
    elif args.classifier_type == "LSTM2":
        model = LSTMClassifier2(args.input_dim, args.hidden_dim)
    elif args.classifier_type == "P_LSTM2":
        assert args.info_type == "each"
        model = Perceiver_LSTMClassifier2(args.input_dim, args.hidden_dim, 
                                          len(args.layers_to_process), 
                                          num_latents=args.p_num_latents,
                                          share_perceiver=args.share_perceiver)
    elif args.classifier_type == "P_LlamaMLP2":
        assert args.info_type == "each"
        model = P_LlamaMLPClassifier2(args.input_dim, args.hidden_dim, 
                                     len(args.layers_to_process), 
                                     num_latents=args.p_num_latents,
                                     share_perceiver=args.share_perceiver)
    elif args.classifier_type == "P_LlamaMLP":
        assert args.info_type == "each"
        model = P_LlamaMLPClassifier(args.input_dim, args.hidden_dim, 
                                     len(args.layers_to_process), 
                                     num_latents=args.p_num_latents,
                                     share_perceiver=args.share_perceiver)
    elif args.classifier_type == "ResNet2":
        model = ResNetClassifier2(len(args.layers_to_process))
    elif args.classifier_type == "postiveff1":
        model = postiveLinearClassifier1(args.input_dim)
    else:
        assert False
        
    if args.ddp and torch.cuda.device_count() > 1:
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)
    else:
        model = model.to(args.device)
    return model

def test_model(save_dir, best_epoch, dataset, args, INPUT):
    # evaluation best model on test set
    if not (args.ddp and torch.cuda.device_count() > 1):
        test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False)
    model = load_classifier_model(args)
    model.load_state_dict(torch.load(save_dir+f"/{best_epoch}_model_weights.pth"))
    model.eval()
    for param in model.parameters():
        DTYPE = param.dtype
        break
    device = model.get_device()
    criterion = get_criterion(args)

    t1 = time.time()
    loss_test, acc_test, f1_test = 0.0, None, None
    all_preds, all_pred_scores, all_labels = [], [], []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs = get_dataset_batch_input(batch, INPUT)
        inputs = inputs.to(DTYPE).to(device)
        outputs = model(inputs)
        labels = batch["label"].to(device)
        if args.classifier_type.endswith("2"):
            labels = labels.to(torch.long)
        else:
            labels = labels.to(DTYPE)
            outputs = outputs.view(-1)
        if args.uncertainty:
            loss = criterion(outputs, labels, 1)
        else:
            loss = criterion(outputs, labels)
        loss_test += loss
        if args.task == "classification":
            if args.classifier_type.endswith("2"):
                pred_score = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
            else:
                pred_score = torch.sigmoid(outputs)
                preds = torch.round(pred_score)

            all_pred_scores.append(pred_score)
            all_preds.append(preds)
            all_labels.append(labels)

    t2 = time.time()
    test_runtime = t2 - t1

    loss_test = loss_test.item()/(i+1)
    n_test_samples = len(dataset['test'])
    if args.task == "classification":
        all_preds = torch.cat(all_preds)
        all_pred_scores = torch.cat(all_pred_scores)
        all_labels = torch.cat(all_labels)
        acc_test = torch.eq(all_preds, all_labels).sum().item()
        acc_test /= n_test_samples
        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        f1_test = {"test_macro_f1": macro_f1, "test_micro_f1": micro_f1, "test_weighted_f1": weighted_f1}
        all_preds = all_preds.tolist()
        print(f"test set loss at epoch {best_epoch}: {loss_test}, acc: {acc_test}, f1: {f1_test}")
        with open(f"{save_dir}/{args.predict_result_file}", "w") as f:
            f1_test.update({"epoch": best_epoch,
                        "test_loss": loss_test, 
                        "test_accuracy": acc_test,
                        "test_runtime": test_runtime,
                        "test_samples": n_test_samples,
                        "all_preds": all_preds})
            json.dump(f1_test, f)
    else:
        print(f"test set loss at epoch {best_epoch}: {loss_test}")
        with open(f"{save_dir}/{args.predict_result_file}", "w") as f:
            json.dump({"epoch": best_epoch,
                        "test_loss": loss_test, 
                        "test_mse": loss_test,
                        "test_runtime": test_runtime,
                        "test_samples": n_test_samples,
                        "all_preds": all_preds,
                        "all_pred_scores": all_pred_scores}, f)
    return loss_test, acc_test, f1_test


def get_dataset(args, all_hidden_states, cached_dataset_path):
    if args.use_cached_dataset:
        if args.cached_dataset_path:
            cached_dataset_path = args.cached_dataset_path
        assert os.path.isfile(cached_dataset_path)
        with open(cached_dataset_path, "rb") as f:
            dataset = pickle.load(f)

        if dataset['train'][0]["hidden_states"].shape[0] == 4096 and args.hidden_state_dims:
            for split in ['train', 'valid', 'test']:
                data2 = []
                for example in dataset[split].data:
                    example["hidden_states"] = example["hidden_states"][args.hidden_state_dims]
                    data2.append(example)
                dataset[split].data = data2
        elif args.hidden_state_dims:
            assert False

        if args.except_source_dataset:
            print("delete source dataset from cached data", args.except_source_dataset)
            for split in ['train', 'valid', 'test']:
                data2 = []
                for example in dataset[split].data:
                    if args.except_source_dataset not in example["id"]:
                        data2.append(example)
                dataset[split].data = data2
    else:
        dataset = {
            "train": HiddenLayerDataset(args.input_path.format(split='train'), all_hidden_states, args.layer, args.label, args.hidden_state_dims, args.ignore_missing_hs),
            "valid": HiddenLayerDataset(args.input_path.format(split='val'), all_hidden_states, args.layer, args.label, args.hidden_state_dims, args.ignore_missing_hs),
            "test": HiddenLayerDataset(args.input_path.format(split='test'), all_hidden_states, args.layer, args.label, args.hidden_state_dims, args.ignore_missing_hs)
        }
        with open(cached_dataset_path, "wb") as f:
            pickle.dump(dataset, f)
    return dataset

def train_each_layer(args, all_hidden_states, the_last_round=False):
    save_dir = f"{args.save_dir}/{args.layer}/"
    os.makedirs(save_dir, exist_ok=True)
    cached_dataset_path = f"{save_dir}/cached_dataset.pkl"
    dataset = get_dataset(args, all_hidden_states, cached_dataset_path)
            
    if the_last_round:
        del all_hidden_states

    train_model(args, dataset, save_dir)

    # if args.clean_checkpoints or args.layer != 32 :
    #     for file in os.listdir(save_dir):
    #         if file.endswith("model_weights.pth"):
    #             os.remove(save_dir+f"/{file}")


def get_dataset_batch_input(batch, INPUT):
    if INPUT == "hidden_states_logits_js_div":
        hidden_states = batch["hidden_states"]
        logits_js_div = batch["logits_js_div"]
        inputs = torch.cat([hidden_states, logits_js_div], dim=1)
    elif INPUT == "max_activation_ratio_sparsity_correlation":
        max_activation_ratio = batch["max_activation_ratio"].squeeze(-1) # [batch_size, num_layers, 1]
        sparsity = batch["sparsity"].squeeze(-1) # [batch_size, num_layers, 1]
        activation_correlation = batch["activation_correlation"].squeeze(1) # [batch_size, 1, hidden_dim]
        inputs = torch.cat([max_activation_ratio, sparsity, activation_correlation], dim=1) # (batch_size, 2*num_layers +hidden_dim)

    elif INPUT == "max_activation_ratio_sparsity":
        max_activation_ratio = batch["max_activation_ratio"] # [batch_size, num_layers, 1]
        sparsity = batch["sparsity"] # [batch_size, num_layers, 1]
        inputs = torch.stack([max_activation_ratio, sparsity], dim=1) # (batch_size, 2, seq_len)
    else:
        inputs = batch[INPUT]

    return inputs

def ddpsetup(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    distributed.init_process_group("nccl", 
                                    rank=rank, 
                                    world_size=torch.cuda.device_count())

def get_dataloader(rank, dataset, batch_size, shuffle):
    sampler = DistributedSampler(dataset, shuffle=shuffle, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    return dataloader


def train_model(rank, args, dataset, save_dir, INPUT="hidden_states"):
    os.makedirs(save_dir, exist_ok=True)
    shuffle = not args.classifier_type.startswith("P_") # seq_len is different for each batch
    
    if args.ddp and torch.cuda.device_count() > 1:
        print("rank", rank)
        ddpsetup(rank)
        train_loader = get_dataloader(rank, dataset["train"], batch_size=args.batch_size, shuffle=shuffle)
        valid_loader = get_dataloader(rank, dataset["valid"], batch_size=args.batch_size, shuffle=False)
        model = load_classifier_model(args, rank)
    else:
        # if args.deepspeed and torch.cuda.device_count() > 1:
        #     deepspeed.init_distributed()
        train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=shuffle)
        valid_loader = DataLoader(dataset["valid"], batch_size=args.batch_size, shuffle=False)
        model = load_classifier_model(args)
    
    for param in model.parameters():
        DTYPE = param.dtype
        break
    device = model.get_device()

    criterion = get_criterion(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.deepspeed:
        ds_config = {  
            "train_batch_size": args.batch_size,  
            "optimizer": {  
                "type": "Adam",  
                "params": {  
                    "lr": args.lr,
                }  
            },  
            "fp16": {  
                "enabled": True  
            },
            "zero_optimization": {
                "stage": 3,
                "contiguous_gradients": True,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 1e7,
                "stage3_param_persistence_threshold": 1e5,
                "reduce_bucket_size": 1e7,
                "sub_group_size": 1e9,
                # "offload_optimizer": {
                #     "device": "cpu"
                # },
                # "offload_param": {
                #     "device": "cpu"
                # }
            }
        } 
        print(deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model, 
                                                                             num_gpus_per_node=2, 
                                                                             num_nodes=1, additional_buffer_factor=1.5))
        # Initialize DeepSpeed engine  
        model, optimizer, _, _ = deepspeed.initialize(  
            args=None,  
            model=model,  
            model_parameters=model.parameters(),  
            optimizer=optimizer,  
            config_params=ds_config,
            dist_init_required=True 
        )  
        

    best_loss, best_acc, best_epoch = float("inf"), float("-inf"), -1
    log_text = ''

    for file in os.listdir(save_dir):
        if file.endswith("model_weights.pth"):
            os.remove(save_dir+f"/{file}")
            
    for epoch in range(args.training_epoch):
        print(f"Epoch {epoch}")
        model.train()
        running_loss, acc_train = 0.0, 0.0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            inputs = get_dataset_batch_input(batch, INPUT)
            inputs = inputs.to(DTYPE).to(device)
            outputs = model(inputs)
            labels = batch["label"].to(device)
            if not args.classifier_type.endswith("2"):
                outputs = outputs.view(-1)

            if args.classifier_type.endswith("2"):
                assert args.task == "classification"
                labels = labels.to(torch.long)
                # print("outputs shape", outputs.shape, outputs)
                # print("labels shape", labels.shape, labels)
                if args.uncertainty:
                    loss = criterion(outputs, labels, epoch)
                else:
                    loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                acc = torch.eq(preds, labels).sum().item()
            elif args.task == "classification":
                labels = labels.to(DTYPE)
                loss = criterion(outputs, labels)
                preds = torch.round(torch.sigmoid(outputs)) #threshold = 0.5
                acc = torch.eq(preds, labels).sum().item()
            else:
                assert args.task == "regression"
                labels = labels.to(DTYPE)
                loss = criterion(outputs, labels)
                acc = -1
            
            # Backward pass  
            if args.deepspeed:
                model.backward(loss)  
                model.step()  
            else:
                loss.backward()
                optimizer.step()

            if torch.isnan(loss):
                print("--------------------------------- NAN training loss ---------------------------------")
                print(batch["id"])
                print("inputs", inputs.shape, inputs)
                print("outputs", outputs.shape, outputs)
                print(batch["label"])
                print("save_dir", save_dir)
                assert False 
            running_loss += loss.item()
            acc_train += acc
            wandb.log({"train_loss": loss.item(), "train_acc": acc_train/(i+1)})

        # evaluation on validation set
        model.eval()
        all_preds, all_labels = [], []
        loss_val = 0.0
        t1 = time.time()
        for i, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            inputs = get_dataset_batch_input(batch, INPUT)
            inputs = inputs.to(DTYPE).to(device)
            outputs = model(inputs)
            labels = batch["label"].to(device)
            if args.classifier_type.endswith("2"):
                labels = labels.to(torch.long)
            else:
                outputs = outputs.view(-1)
                labels = labels.to(DTYPE)
            if args.uncertainty:
                loss = criterion(outputs, labels, epoch)
            else:
                loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print("--------------------------------- NAN valid loss ---------------------------------")
                print(batch["id"])
                print("inputs", inputs.shape, inputs)
                print("outputs", outputs.shape, outputs)
                print(batch["label"])
                print("save_dir", save_dir)
                assert False
            # print(f"batch {i} val batch loss {loss}")
            if args.task == "classification":
                if args.classifier_type.endswith("2"):
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = torch.round(torch.sigmoid(outputs))
                all_preds.append(preds)
                all_labels.append(labels)
                
            loss_val += loss
            # print(f"batch {i} val loss_val {loss_val}")
        t2 = time.time()
        eval_runtime = t2 - t1
        loss_val = loss_val.item()/(i+1)
        n_eval_samples = len(dataset['valid'])
        if args.task == "classification":
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            acc_val = torch.eq(all_preds, all_labels).sum().item()
            acc_val /= n_eval_samples
            
            print(f"Epoch{epoch}: valid set loss: {loss_val}, acc: {acc_val}")
            log_text += f"Epoch{epoch}: valid set loss: {loss_val}, acc: {acc_val}\n"
            wandb.log({"eval_loss": loss_val, "eval_acc": acc_val})
            
            all_preds = all_preds.cpu().detach().numpy()
            all_labels = all_labels.cpu().detach().numpy()
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            micro_f1 = f1_score(all_labels, all_preds, average='micro')
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
            all_preds = all_preds.tolist()
            f1_val = {"eval_macro_f1": macro_f1, "eval_micro_f1": micro_f1, "eval_weighted_f1": weighted_f1}

            if acc_val > best_acc:
                best_acc = acc_val
                best_f1 = f1_val
                best_epoch = epoch
                with open(f"{save_dir}/eval_results.json", "w") as f:
                    best_f1.update({"epoch": best_epoch,
                                "eval_loss": loss_val, 
                                "eval_accuracy": best_acc,
                                "eval_runtime": eval_runtime,
                                "eval_samples": n_eval_samples,
                                "all_preds": all_preds})
                    json.dump(f1_val, f)

                save_and_clear_checkpoint(save_dir, model, best_epoch)
            else:
                args.patience -= 1
                if not args.patience:
                    break
        elif args.task == "regression":
            print(f"Epoch{epoch}: valid set loss: {loss_val}")
            log_text += f"Epoch{epoch}: valid set loss: {loss_val}\n"
            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                with open(f"{save_dir}/eval_results.json", "w") as f:
                    json.dump({"epoch": best_epoch,
                                "eval_loss": loss_val,
                                "eval_mse": loss_val,
                                "eval_runtime": eval_runtime,
                                "eval_samples": len(dataset['valid']),
                                "all_preds": all_preds}, f)
                save_and_clear_checkpoint(save_dir, model, best_epoch)
            else:
                args.patience -= 1
                if not args.patience:
                    break

    loss_test, acc_test, f1_test = test_model(save_dir, best_epoch, dataset, args, INPUT)
    log_text += f"test set loss at epoch {best_epoch}: {loss_test}, acc: {acc_test}, f1: {f1_test}\n"
    with open(save_dir+f"/log.txt", 'w') as f:
        f.write(log_text)

    if args.deepspeed:
        model.cleanup()
    if args.ddp and torch.cuda.device_count() > 1:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=11008)
    parser.add_argument("--classifier_type", type=str, default="LlamaMLP")
    parser.add_argument("--p_num_latents", type=int, default=1)
    parser.add_argument("--hidden_state_type", type=str, choices=["last", "mean", 'each'])
    parser.add_argument("--layers_to_process", nargs='*', type=str)
    parser.add_argument("--hidden_state_dims", type=str)
    parser.add_argument("--select_hidden_state_dims_method", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--uncertainty", type=str, default="")
    parser.add_argument("--annealing_step", type=int, default=10) #?????
    parser.add_argument("--training_epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_batch_size", type=int, default=64)
    parser.add_argument("--source_dirs", nargs='*', type=str)
    parser.add_argument("--input_path_o", type=str)
    parser.add_argument("--labels", nargs='*', type=str)
    parser.add_argument("--generate_model_name", type=str, default="")
    parser.add_argument("--hidden_state_model_name", type=str, default="")
    parser.add_argument("--save_dir_root", type=str)
    parser.add_argument("--less_mermory", action="store_true")
    parser.add_argument("--train_from_scratch", action="store_true")
    parser.add_argument("--clean_checkpoints", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--ignore_missing_hs", action="store_true")
    parser.add_argument("--use_cached_dataset", action="store_true")
    parser.add_argument("--cached_dataset_path", type=str, default="")
    parser.add_argument("--predict_result_file", type=str, default="test_results.json")
    parser.add_argument("--except_source_dataset", type=str, default="")
    parser.add_argument("--only_source_dataset", type=str, default="")
    args = parser.parse_args()

    batch_size = args.batch_size
    classifier_type = args.classifier_type
    hidden_state_type = args.hidden_state_type
    args.layers_to_process = process_layers_to_process(args.layers_to_process)
    save_dir_root = args.save_dir_root
    
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generate_model_name = args.generate_model_name
    if args.hidden_state_model_name:
        hidden_state_model_name = args.hidden_state_model_name
    else:
        hidden_state_model_name = args.generate_model_name
        
    save_attention = False
    all_hidden_states = None
    if not args.only_predict:
        save_dir_root += "/"+hidden_state_type
        if args.hidden_state_dims:
            args.hidden_state_dims = [int(i) for i in args.hidden_state_dims.split(",")]
            args.input_dim = len(args.hidden_state_dims)
            print("hidden_state_dims", args.hidden_state_dims)
            save_dir_root += f"/{args.select_hidden_state_dims_method}_top{args.input_dim}"
            os.makedirs(save_dir_root, exist_ok=True)
            with open(f"{save_dir_root}/hidden_state_dims.txt", "w") as f:
                f.write(str(args.hidden_state_dims))
        if not args.use_cached_dataset and not args.less_mermory:
            all_hidden_states = prepare_hidden_states(hidden_state_type, 
                                                      args.source_dirs,  
                                                      args.layers_to_process, hidden_state_model_name, args.hidden_batch_size, args.device, 
                                                      args.ignore_missing_hs) 
        
        if args.labels == ['all']:
            args.labels = ["label", 
                        "rouge_100", "standarded_rouge", "discrete_rouge", "rank_rouge", 
                            "entail_100", "standarded_entail", "discrete_entail", "rank_entail", 
                            "questeval", "standarded_questeval", "discrete_questeval", "rank_questeval"]

        for layer in args.layers_to_process:
            print("layer", layer)
            args.layer = layer
            if not args.use_cached_dataset and args.less_mermory:
                all_hidden_states = prepare_hidden_states(hidden_state_type,
                                                          args.source_dirs, 
                                                          [layer], hidden_state_model_name, args.hidden_batch_size, args.device,
                                                          args.ignore_missing_hs)
            # more time to load but save memory
            num_labels = len(args.labels)
            for l_idx, label in enumerate(args.labels):
                args.label = label
                if hidden_state_model_name == generate_model_name:
                    args.save_dir = f"{save_dir_root}/{classifier_type}_{label}_{generate_model_name}_model"
                else:
                    print("hidden_state_model_name != generate_model_name")
                    args.save_dir = f"{save_dir_root}/{classifier_type}_{label}_{generate_model_name}_{hidden_state_model_name}_model"
                ########################################################################
                if not args.train_from_scratch and os.path.isfile(f"{args.save_dir}/{layer}/test_results.json"):
                    with open(f"{args.save_dir}/{layer}/test_results.json") as f:
                        if "test_macro_f1" in json.load(f):
                            print(f"already have {layer} {label}")
                            continue # don't need to train
                # if os.path.isfile(f"{args.save_dir}/{layer}/eval_results.json"):
                #     with open(f"{args.save_dir}/{layer}/eval_results.json") as f:
                #         best_epoch = json.load(f)["epoch"]
                #     if os.path.isfile(f"{args.save_dir}/{layer}/{best_epoch}_model_weights.pth"):
                #         print(f"already have {layer} {label}")
                #         continue # don't need to train
                #     # 之前训练过 有eval_results.json 但删除了 model_weights.pth 重新train
                #     for s in ['test', 'eval']:
                #         if os.path.isfile(f"{args.save_dir}/{layer}/{s}_results.json"):
                #             if os.path.isfile(f"{args.save_dir}/{layer}/{s}_results_old.json"):
                #                 assert False
                #             else:
                #                 shutil.copyfile(f"{args.save_dir}/{layer}/{s}_results.json", f"{args.save_dir}/{layer}/{s}_results_old.json")
                ########################################################################
                os.makedirs(args.save_dir, exist_ok=True)
                if re.search("(discrete)|(label)|(source)", label):
                    args.task = "classification"
                else:
                    args.task = "regression"
                    
                if args.input_path_o:
                    args.input_path = args.input_path_o
                else:
                    t = args.source_dirs[0].split("/")[-1]
                    args.input_path = f"{cwd}/classifier/NI/{t}/dataset/{args.generate_model_name}_"
                    if label == 'label':
                        args.input_path += "{split}_comprehensive.csv"
                    else:
                        args.input_path += "{split}.csv"
                        
                train_each_layer(args, all_hidden_states, l_idx==num_labels-1)
    else: ################# only predict classification ##################
        assert len(args.labels) == 1
        label = args.labels[0]
        args.task = "classification"
        criterion = nn.BCEWithLogitsLoss()
        layers_to_process = args.layers_to_process
        predict_result_file = args.predict_result_file

        FLAG = False
        for layer in layers_to_process:
            best_epoch = -1
            if os.path.isfile(f"{save_dir_root}/{layer}/eval_results.json"):
                with open(f"{save_dir_root}/{layer}/eval_results.json") as f:
                    best_epoch = json.load(f)["epoch"]
            if os.path.isfile(f"{save_dir_root}/{layer}/{best_epoch}_model_weights.pth"):
                FLAG = True
        if FLAG:
            if not args.use_cached_dataset:
                if args.source_dirs:
                    all_hidden_states = prepare_hidden_states(hidden_state_type,
                                                            args.source_dirs, 
                                                            layers_to_process, 
                                                            hidden_state_model_name, args.hidden_batch_size, args.device,
                                                            args.ignore_missing_hs)
                else:
                    if hidden_state_type == 'last':
                        hidden_state_path = "/".join(args.input_path_o.split("/")[:-1]) + "/hidden_state.pkl"
                    elif hidden_state_type == 'mean':
                        hidden_state_path = "/".join(args.input_path_o.split("/")[:-1]) + "/hidden_state_mean.pkl"
                    all_hidden_states = prepare_hidden_states(hidden_state_type,
                                                            [hidden_state_path], 
                                                            layers_to_process, 
                                                            hidden_state_model_name, args.hidden_batch_size, args.device,
                                                            args.ignore_missing_hs)
                
        for layer in layers_to_process:
            best_epoch = -1
            if os.path.isfile(f"{save_dir_root}/{layer}/eval_results.json"):
                with open(f"{save_dir_root}/{layer}/eval_results.json") as f:
                    best_epoch = json.load(f)["epoch"]
            if os.path.isfile(f"{save_dir_root}/{layer}/{best_epoch}_model_weights.pth"):
                # if os.path.isfile(f"{save_dir_root}/{layer}/{predict_result_file}"):
                #     # print(f"already have predict result")
                #     # assert False
                #     if os.path.isfile(f"{save_dir_root}/{layer}/{predict_result_file[:-5]}_old.json"):
                #         assert False
                #     else:
                #         shutil.copyfile(f"{save_dir_root}/{layer}/{predict_result_file}", f"{save_dir_root}/{layer}/{predict_result_file[:-5]}_old.json")
                if args.use_cached_dataset and args.cached_dataset_path:
                    cached_dataset_path = args.cached_dataset_path
                    assert os.path.isfile(cached_dataset_path)
                    with open(cached_dataset_path, "rb") as f:
                        dataset = pickle.load(f)
                    dataset = {"test": dataset["test"]}
                    if args.only_source_dataset:
                        print("load source dataset from cached data", args.only_source_dataset)
                        for split in ['test']:
                            data2 = []
                            for example in dataset[split].data:
                                if args.only_source_dataset in example["id"]:
                                    data2.append(example)
                            dataset[split].data = data2
                            print("test_len", len(data2))
                else:
                    dataset = {"test": HiddenLayerDataset(args.input_path_o, all_hidden_states, layer, label)}
                loss_test, acc_test, f1_test = test_model(f"{save_dir_root}/{layer}", best_epoch, dataset, args)
            else:
                print(f"{save_dir_root}/{layer} no exist checkpoint")

        