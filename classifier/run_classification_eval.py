import os
path = os.getcwd()
root_path = "/".join(path.split("/")[:4])
from sklearn.metrics import f1_score, recall_score, precision_score
import csv
import argparse
import json
import numpy as np

LABELS = {"ok": 1, "hallucinated": 0, 
          "yes": 1, "no": 0, }

def run_classification_eval(generated_path, generated_label, label_path):
    with open(label_path) as f:
        reader = csv.DictReader(f)
        label_lines = list(reader)

    preds, labels = [], []
    if generated_path.endswith("csv"):
        with open(generated_path) as f:
            reader = csv.DictReader(f)
            lines = list(reader)
            assert len(lines) == len(label_lines)
            for line, ll in zip(lines, label_lines):
                p = line[generated_label]
                if p in LABELS:
                    preds.append(LABELS[p])
                    labels.append(LABELS[ll['label']])
                # else:
                #     label = LABELS[ll['label']]
                #     if label == 1:
                #         preds.append(0)
                #     else:
                #         preds.append(1)
                #     labels.append(label)
    else:
        with open(generated_path) as f:
            lines = f.readlines()[1:]
            assert len(lines) == len(label_lines)
            for line, ll in zip(lines, label_lines):
                p = line.strip().split("\t")[1]
                preds.append(LABELS[p])
                labels.append(LABELS[ll['label']])

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

    res = {"accuracy": acc, 
           "macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1,
           "precision": precision, "recall": recall,
           "macro_recall": macro_recall, "micro_recall": micro_recall, "weighted_recall": weighted_recall,
           "macro_precision": macro_pre, "micro_precision": micro_pre, "weighted_precision": weighted_pre,}
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path",type=str,)
    parser.add_argument("--generated_label",type=str,)
    parser.add_argument("--label_path",type=str,)
    parser.add_argument("--res_path",type=str,)
    args = parser.parse_args()
    
    res = run_classification_eval(args.generated_path, args.generated_label, args.label_path)

    if args.res_path:
        with open(args.res_path, 'w') as f:
            json.dump(res, f)
    else:
        print(res)
            
        