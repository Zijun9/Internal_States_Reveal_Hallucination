# find the specific dims of embedding (neurons) that are most important for the classification
import os
cwd = os.getcwd()
root_path = "/".join(cwd.split("/")[:-1])
import sys
sys.path.append(root_path)
import numpy as np
import torch
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_feedforward_classifier import prepare_hidden_states
import csv
import matplotlib
from tqdm import tqdm


def load_embedding(task, hidden_state_type, layers, hidden_state_model_name, split, data_paths=None, source_dirs=None, ignore_missing_hs=False):
    # different from training labels
    LABELS = {"ok": 0,
              "hallucinated": 1, 
              "seen": 0,
              "unseen": 1,
              "ArchivalQA": 0,
              "AllSides": 1}
    if not source_dirs:
        if task == "unseen":
            source_dirs = [f"/home/zjiad/Hallu_source/generate/AllSides_output", "/home/zjiad/Hallu_source/generate/ArchivalQA_output"]
        else:
            source_dirs = [f"/home/zjiad/Hallu_source/generate/NI_output/{task}"]
    # hidden_state_type, source_dirs, layers_to_process, hidden_state_model_name, batch_size, device, split=['train','val','test']):
    all_hidden_states = prepare_hidden_states(hidden_state_type, 
                                                source_dirs, 
                                                layers, 
                                                hidden_state_model_name, 1, None, 
                                                split=split,
                                                ignore_nan=ignore_missing_hs) 
    all_rows = []
    if data_paths:
        print(f"loading data from {data_paths}")
        for data_path in data_paths:
            with open(data_path) as f:
                reader = csv.DictReader(f)
                all_rows += list(reader)
    else:
        for s in split:
            data_path = f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/Llama2-7B-Chat_{s}_comprehensive.csv"
            with open(data_path) as f:
                reader = csv.DictReader(f)
                all_rows += list(reader)
    print(f"len(all_rows): {len(all_rows)}")
           
    all_X = [] # [layer1, layer2, ...]
    for target_layer in layers:
        X, y = [], [] # layer i
        for row in all_rows:
            id_ = row["id"]
            if task == "unseen" and ("label" not in row):
                print("label" in row, row)
                label = LABELS[row["all_source"]]
            else:
                label = LABELS[row["label"]]
            try:
                embeddings = all_hidden_states[id_][target_layer].numpy(force=True)
                X.append(embeddings)
                y.append(label)
            except:
                print(f"cannot find {id_}, {target_layer}")
                if ignore_missing_hs:
                    continue
                else:
                    raise ValueError

        X, y = np.array(X), np.array(y)
        all_X.append(X)
    return all_X, y

def text2list(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    return [int(i) for i in text.split()]

def get_most_important_indexs(X_list, y, K, method, out_weight=False):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    weights = None
    if method == "ANOVA":
        all_f_values = []
        for X in tqdm(X_list):
            f_values, p_values = f_classif(X, y)
            all_f_values.append(f_values)
        f_values = np.concatenate(all_f_values)
        values, most_important_indexs = torch.topk(torch.tensor(f_values), K)
        most_important_indexs = most_important_indexs.numpy()
        if out_weight:
            f_values_std = scaler.fit_transform(f_values.reshape(-1,1)).reshape(-1)
            weights = f_values_std[most_important_indexs]

    elif method == "mutual information":
        all_mutual_info = []
        for X in tqdm(X_list):
            mutual_info = mutual_info_classif(X, y)
            all_mutual_info.append(mutual_info)
        all_mutual_info = np.concatenate(all_mutual_info)
        values, most_important_indexs = torch.topk(torch.tensor(all_mutual_info), K)
        most_important_indexs = most_important_indexs.numpy()
        if out_weight:
            mutual_info_std = scaler.fit_transform(all_mutual_info.reshape(-1,1)).reshape(-1)
            weights = mutual_info_std[most_important_indexs]

    elif method == "cov":
        # https://zhuanlan.zhihu.com/p/37495710
        scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(0, 1))
        sample_num = y.shape[0]
        all_covMat_abs = []
        for X in tqdm(X_list):
            XY = np.concatenate([X, y.reshape(sample_num,1)], axis=1)
            XY_std = scaler.fit_transform(XY)
            covMat = np.cov(XY_std, rowvar=0) #rowvar = 0表示传入的数据一行代表一个样本
            covMat_abs = abs(covMat[:, -1][:-1]) # remove the y column
            all_covMat_abs.append(covMat_abs)
        all_covMat_abs = np.concatenate(all_covMat_abs)
        values, most_important_indexs = torch.topk(torch.tensor(all_covMat_abs), K)
        most_important_indexs = most_important_indexs.numpy()
        if out_weight:
            covMat_abs_std = scaler.fit_transform(all_covMat_abs.reshape(-1,1)).reshape(-1)
            weights = covMat_abs_std[most_important_indexs]

    elif method.endswith(".pth"): # linear_learable_model
        # save_dir = "/home/zjiad/Hallu_source/classifier/NI/Question_Answering/ff1_label_Llama2-7B-Chat_model/32/"
        # save_dir+f"/9_model_weights.pth"
        weights = torch.load(method)['score.weight']
        weights_abs = weights.abs()
        values, indices = torch.topk(weights_abs, K)
        most_important_indexs = indices.cpu().numpy().squeeze()
        if out_weight:
            values_std =  scaler.fit_transform(values.cpu().numpy().squeeze().reshape(-1,1)).reshape(-1)
            weights = values_std

    elif method == "DecisionTree": # not good slow
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)
        importances = clf.feature_importances_
        importances_std = scaler.fit_transform(importances.reshape(-1,1)).reshape(-1)
        values, most_important_indexs = torch.topk(torch.tensor(importances), K)
        most_important_indexs = most_important_indexs.numpy()
        weights =importances_std[most_important_indexs]
        
    return most_important_indexs, weights


def colorize(tokens, scores, tokenizer):
    token_ids = tokenizer.encode(tokens)[1:]
    assert len(tokens) == len(token_ids)
    # '<0xE5>'
    # process Chinese tokens
    words, color_array, subword, subword_score = [], [], [], []
    for token, token_id, score in zip(tokens, token_ids, scores):
        # print("token", token, "token_id", token_id)
        if token in ['<s>', '</s>', '[', 'INST', ']', '▁[', '/']:
            continue

        if "<0x" in token:
            subword.append(token_id)
            subword_score.append(score)
        else:
            if subword:
                # print("before decode subword", subword)
                subword = tokenizer.decode(subword)
                # print("after decode subword", subword)
                words.append(subword)
                color_array.append(np.mean(subword_score))
            subword, subword_score = [], []

            words.append(token)
            color_array.append(score)

    if subword:
        # print("before decode subword", subword)
        subword = tokenizer.decode(subword)
        # print("after decode subword", subword)
        words.append(subword)
        color_array.append(np.mean(subword_score))
        
    # words is a list of words
    # color_array # 0-1越大越深
    cmap = matplotlib.cm.get_cmap('Reds')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')

    # with open('colorize.html', 'w') as f:
    #     f.write(s)
    return colored_string

# # Linear Discriminant Analysis, LDA
# NOT WORK
# https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/



# Supervised PCA
# 原有的embedding会变 不是挑出idx 而且把dim size变小
# X, y = QA_X_32, QA_y_32
# y_color = [COLORS[i] for i in y]

# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)

# # Split the data based on labels
# X_class0 = X_std[y == 0]
# X_class1 = X_std[y == 1]

# # Compute class means
# mean_class0 = np.mean(X_class0, axis=0)
# mean_class1 = np.mean(X_class1, axis=0)

# # Center the data of each class by subtracting the respective class mean
# X_class0_centered = X_class0 - mean_class0
# X_class1_centered = X_class1 - mean_class1

# # Combine the centered data
# X_centered_combined = np.vstack((X_class0_centered, X_class1_centered))

# # Apply PCA to the centered combined data
# pca = PCA(random_state=0, n_components=2)
# principal_components = pca.fit_transform(X_centered_combined)
# plt.scatter(principal_components[:,0], principal_components[:,1], 
#             s=scatter_size, c=y_color, alpha=1)