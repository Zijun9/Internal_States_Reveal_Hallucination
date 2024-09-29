import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import csv
import json
from sklearn.metrics import f1_score, recall_score, precision_score
import argparse
import ast
# ast.literal_eval()

def get_train_ppls(dataset_root_path, task):
    task_path  = f"{dataset_root_path}/{task}"
    ID2PPL = {}
    all_q_ppls = []
    sources = list(os.listdir(task_path))
    for source in tqdm(sources, total=len(sources)):
        if not os.path.isdir(f"{task_path}/{source}"):
            continue
        for split in ['train', 'test']:
            data = pd.read_csv(f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv") 
            for i, row in data.iterrows():
                try:
                    q_ppl = float(row['ppl'].split("/")[0])
                    ID2PPL[row['id']] = q_ppl
                    ########################################
                    if split == 'train':
                        all_q_ppls.append(q_ppl)
                    ########################################
                except:
                    print(f"!!!! {task_path}/{source}")
                    if os.path.isfile(f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv_copy"):
                        shutil.copyfile(f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv_copy", f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv")
                    assert False
                              
    all_q_ppls = np.array(all_q_ppls)
    return ID2PPL, all_q_ppls


def scan_threshold(task, ID2PPL, min_q_ppl, max_q_ppl, plot_out=False):
    all_ppl_threshold, all_ppl_comprehensive_ACC, all_ppl_rouge_ACC, all_ppl_entail_ACC, all_ppl_questeval_ACC  = [], [], [], [], []
    best_ppl_comprehensive_ACC, best_ppl_comprehensive_threshold = 0, -1
    best_ppl_rouge_ACC, best_ppl_rouge_threshold = 0, -1
    best_ppl_entail_ACC, best_ppl_entail_threshold = 0, -1
    best_ppl_questeval_ACC, best_ppl_questeval_threshold = 0, -1

    for ppl_threshold in tqdm(range(int(min_q_ppl), min(1000, int(max_q_ppl)))):
        ppl_comprehensive_ACC, ppl_rouge_ACC, ppl_entail_ACC, ppl_questeval_ACC = [], [], [], []
        with open(f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_train_comprehensive.csv") as fc, \
            open(f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_train.csv") as f:
            reader = csv.DictReader(fc)
            for row in reader:
                ppl = ID2PPL[row['id']]
                discrete_ppl = ppl < ppl_threshold
                ppl_comprehensive_ACC.append(discrete_ppl == (row['label'] == "ok"))
            ppl_comprehensive_ACC = round(np.array(ppl_comprehensive_ACC).mean()*100, 2)
            all_ppl_threshold.append(ppl_threshold)
            all_ppl_comprehensive_ACC.append(ppl_comprehensive_ACC)

            if ppl_comprehensive_ACC > best_ppl_comprehensive_ACC:
                best_ppl_comprehensive_threshold = ppl_threshold
                best_ppl_comprehensive_ACC = ppl_comprehensive_ACC
                
            reader = csv.DictReader(f)
            for row in reader:
                ppl = ID2PPL[row['id']]
                discrete_ppl = ppl < ppl_threshold
                ppl_rouge_ACC.append(discrete_ppl == (row['discrete_rouge']=="True"))
                ppl_entail_ACC.append(discrete_ppl == (row['discrete_entail']=="True")) 
                ppl_questeval_ACC.append(discrete_ppl == (row['discrete_questeval']=="True"))

            ppl_rouge_ACC = round(np.array(ppl_rouge_ACC).mean()*100, 2)
            ppl_entail_ACC = round(np.array(ppl_entail_ACC).mean()*100, 2)
            ppl_questeval_ACC = round(np.array(ppl_questeval_ACC).mean()*100, 2)

            if ppl_rouge_ACC > best_ppl_rouge_ACC:
                best_ppl_rouge_threshold = ppl_threshold
                best_ppl_rouge_ACC = ppl_rouge_ACC
            if ppl_entail_ACC > best_ppl_entail_ACC:
                best_ppl_entail_threshold = ppl_threshold
                best_ppl_entail_ACC = ppl_entail_ACC
            if ppl_questeval_ACC > best_ppl_questeval_ACC:
                best_ppl_questeval_threshold = ppl_threshold
                best_ppl_questeval_ACC = ppl_questeval_ACC

            all_ppl_rouge_ACC.append(ppl_rouge_ACC)
            all_ppl_entail_ACC.append(ppl_entail_ACC)
            all_ppl_questeval_ACC.append(ppl_questeval_ACC)
    if plot_out:
        plt.plot(all_ppl_threshold,all_ppl_comprehensive_ACC, 'k', label="comprehensive")
        plt.plot(all_ppl_threshold,all_ppl_rouge_ACC, 'b', label="rouge")
        plt.plot(all_ppl_threshold,all_ppl_entail_ACC, 'g', label="entail")
        plt.plot(all_ppl_threshold,all_ppl_questeval_ACC, 'c', label="questeval")
        plt.xlabel("ppl_threshold")
        plt.ylabel("ACC")
        plt.legend(loc = "best")
        plt.title(task)
        plt.show()

    return all_ppl_threshold, all_ppl_comprehensive_ACC, all_ppl_rouge_ACC, all_ppl_entail_ACC, all_ppl_questeval_ACC, \
            best_ppl_comprehensive_threshold, best_ppl_comprehensive_ACC, \
            best_ppl_rouge_threshold, best_ppl_rouge_ACC, \
            best_ppl_entail_threshold, best_ppl_entail_ACC, \
            best_ppl_questeval_threshold, best_ppl_questeval_ACC

def test_acc_f1(test_comprehensive_file, test_single_file, ID2PPL, 
                best_ppl_comprehensive_threshold, best_ppl_rouge_threshold, best_ppl_entail_threshold, best_ppl_questeval_threshold, 
                force_pred_label=-1):
    preds, labels = [], []
    with open(test_comprehensive_file) as fc:
        reader = csv.DictReader(fc)
        for row in reader:
            if force_pred_label == -1:
                ppl = ID2PPL[row['id']]
                preds.append(ppl < best_ppl_comprehensive_threshold)
            else:
                preds.append(force_pred_label)
            labels.append(row['label'] == "ok")
    preds = np.array(preds)
    labels = np.array(labels)

    assert len(preds) == len(labels)
    acc = np.mean(preds == labels)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)

    macro_recall = recall_score(labels, preds, average='macro')
    micro_recall = recall_score(labels, preds, average='micro')
    weighted_recall = recall_score(labels, preds, average='weighted')
    macro_pre = precision_score(labels, preds, average='macro')
    micro_pre = precision_score(labels, preds, average='micro')
    weighted_pre = precision_score(labels, preds, average='weighted')

    comprehensive_res = {"accuracy": acc, 
                         "macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1,
                         "macro_recall": macro_recall, "micro_recall": micro_recall, "weighted_recall": weighted_recall,
                         "macro_pre": macro_pre, "micro_pre": micro_pre, "weighted_pre": weighted_pre,
                         "precision":precision, "recall":recall}

    if test_single_file:
        rouge_preds, entail_preds, questeval_preds = [], [], []
        rouge_labels, entail_labels, questeval_labels = [], [], []
        with open(test_single_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if force_pred_label == -1:
                    ppl = ID2PPL[row['id']]
                    rouge_preds.append(ppl < best_ppl_rouge_threshold)
                    entail_preds.append(ppl < best_ppl_entail_threshold)
                    questeval_preds.append(ppl < best_ppl_questeval_threshold)
                else:
                    rouge_preds.append(force_pred_label)
                    entail_preds.append(force_pred_label)
                    questeval_preds.append(force_pred_label)

                rouge_labels.append(row['discrete_rouge']=="True")
                entail_labels.append(row['discrete_entail']=="True")
                questeval_labels.append(row['discrete_questeval']=="True")
        rouge_preds = np.array(rouge_preds)
        entail_preds = np.array(entail_preds)
        questeval_preds = np.array(questeval_preds)
        rouge_labels = np.array(rouge_labels)
        entail_labels = np.array(entail_labels)
        questeval_labels = np.array(questeval_labels)
        
        rouge_acc = np.mean(rouge_preds == rouge_labels)
        entail_acc = np.mean(entail_preds == entail_labels)
        questeval_acc = np.mean(questeval_preds == questeval_labels)

        rouge_macro_f1 = f1_score(rouge_labels, rouge_preds, average='macro')
        entail_macro_f1 = f1_score(entail_labels, entail_preds, average='macro')
        questeval_macro_f1 = f1_score(questeval_labels, questeval_preds, average='macro')

        rouge_micro_f1 = f1_score(rouge_labels, rouge_preds, average='micro')
        entail_micro_f1 = f1_score(entail_labels, entail_preds, average='micro')
        questeval_micro_f1 = f1_score(questeval_labels, questeval_preds, average='micro')

        rouge_weighted_f1 = f1_score(rouge_labels, rouge_preds, average='weighted')
        entail_weighted_f1 = f1_score(entail_labels, entail_preds, average='weighted')
        questeval_weighted_f1 = f1_score(questeval_labels, questeval_preds, average='weighted')

        rouge_res = {"accuracy": rouge_acc, "macro_f1": rouge_macro_f1, "micro_f1": rouge_micro_f1, "weighted_f1": rouge_weighted_f1,}
        entail_res = {"accuracy": entail_acc, "macro_f1": entail_macro_f1, "micro_f1": entail_micro_f1, "weighted_f1": entail_weighted_f1,}
        questeval_res = {"accuracy": questeval_acc, "macro_f1": questeval_macro_f1, "micro_f1": questeval_micro_f1, "weighted_f1": questeval_weighted_f1,}
    else:
        rouge_res, entail_res, questeval_res = None, None, None
    return comprehensive_res, rouge_res, entail_res, questeval_res

def run(dataset_root_path, task, from_scratch=False, plot_out=False):
    ID2PPL, all_q_ppls = get_train_ppls(dataset_root_path, task)
    median_q_ppl = np.percentile(all_q_ppls, 50)
    min_q_ppl = np.min(all_q_ppls)
    max_q_ppl = np.max(all_q_ppls)
    print(f"Median PPL: {median_q_ppl}, Min PPL: {min_q_ppl}, Max PPL: {max_q_ppl}")

    out_path = f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_ppl.json"
    if not from_scratch and os.path.isfile(out_path):
        print("load from cache")
        with open(out_path) as f:
            data = json.load(f)
            assert len(data.keys()) == 17
        all_ppl_threshold = data["all_ppl_threshold"] 
        all_ppl_comprehensive_ACC = data["all_ppl_train_comprehensive_ACC"]
        all_ppl_rouge_ACC = data["all_ppl_train_rouge_ACC"]
        all_ppl_entail_ACC = data["all_ppl_train_entail_ACC"]
        all_ppl_questeval_ACC = data["all_ppl_train_questeval_ACC"]
        best_ppl_comprehensive_threshold = data["best_ppl_comprehensive_threshold"]
        best_ppl_comprehensive_ACC = data["best_ppl_train_comprehensive_ACC"]
        best_ppl_rouge_threshold = data["best_ppl_rouge_threshold"]
        best_ppl_rouge_ACC = data["best_ppl_train_rouge_ACC"]
        best_ppl_entail_threshold = data["best_ppl_entail_threshold"]
        best_ppl_entail_ACC = data["best_ppl_train_entail_ACC"]
        best_ppl_questeval_threshold = data["best_ppl_questeval_threshold"]
        best_ppl_questeval_ACC = data["best_ppl_train_questeval_ACC"]

    else: # scan threshold
        all_ppl_threshold, all_ppl_comprehensive_ACC, all_ppl_rouge_ACC, all_ppl_entail_ACC, all_ppl_questeval_ACC, \
        best_ppl_comprehensive_threshold, best_ppl_comprehensive_ACC, \
        best_ppl_rouge_threshold, best_ppl_rouge_ACC, \
        best_ppl_entail_threshold, best_ppl_entail_ACC, \
        best_ppl_questeval_threshold, best_ppl_questeval_ACC = scan_threshold(task, ID2PPL, min_q_ppl, max_q_ppl, plot_out)

    test_comprehensive_file = f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_test_comprehensive.csv"
    test_single_file = f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_test.csv"
    comprehensive_res, rouge_res, entail_res, questeval_res = test_acc_f1(test_comprehensive_file, test_single_file, ID2PPL, best_ppl_comprehensive_threshold, best_ppl_rouge_threshold, best_ppl_entail_threshold, best_ppl_questeval_threshold)
    
    with open(out_path, "w") as f:
        data = {"best_ppl_comprehensive_threshold": best_ppl_comprehensive_threshold, 
                "best_ppl_train_comprehensive_ACC": best_ppl_comprehensive_ACC, 
                "ppl_test_comprehensive_res": str(comprehensive_res),
                "best_ppl_rouge_threshold": best_ppl_rouge_threshold, 
                "best_ppl_train_rouge_ACC": best_ppl_rouge_ACC,
                "ppl_test_rouge_res": str(rouge_res),
                "best_ppl_entail_threshold": best_ppl_entail_threshold,
                "best_ppl_train_entail_ACC": best_ppl_entail_ACC,
                "ppl_test_entail_res": str(entail_res),
                "best_ppl_questeval_threshold": best_ppl_questeval_threshold,
                "best_ppl_train_questeval_ACC": best_ppl_questeval_ACC,
                "ppl_test_questeval_res": str(questeval_res),
                "all_ppl_threshold": all_ppl_threshold, 
                "all_ppl_train_comprehensive_ACC": all_ppl_comprehensive_ACC, 
                "all_ppl_train_rouge_ACC": all_ppl_rouge_ACC,
                "all_ppl_train_entail_ACC": all_ppl_entail_ACC,
                "all_ppl_train_questeval_ACC": all_ppl_questeval_ACC,}
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--plot_out", action="store_true")
    args = parser.parse_args()

    dataset_root_path = "/home/zjiad/Hallu_source/generate/NI_output"
    model_name = 'Llama2-7B-Chat'
    for task in ["Question_Answering", "Translation"]:
        print(task)
        run(dataset_root_path, task, args.from_scratch, args.plot_out)