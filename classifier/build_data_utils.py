import os
import pandas as pd
import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from scipy.stats import percentileofscore
from tqdm import tqdm
import json

def get_stats_for_task(task_path, model_name):
    # 综合同一个task下的所有source
    all_entails, all_questevals, all_rouges = [], [], []
    sources = list(os.listdir(task_path))
    for source in tqdm(sources, total=len(sources)):
        if not os.path.isdir(f"{task_path}/{source}/"):
            continue
        for split in ['train', 'val', 'test']:
            data = pd.read_csv(f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv") 
            for i, row in data.iterrows():
                # nli,questeval,rouge
                # "entailment", "neutral", "contradiction"
                # 0: 75.89/14.43/9.68,0.41780670038585005,16.67
                entail = float(row['nli'].split(": ")[1].split("/")[0])/ 100
                all_entails.append(entail)
                all_questevals.append(row['questeval'])
                all_rouges.append(row['rouge']/100)

    all_entails = np.array(all_entails)
    all_questevals = np.array(all_questevals)
    all_rouges = np.array(all_rouges)

    min_entail = min(all_entails)   
    max_entail = max(all_entails)  
    median_entail = np.percentile(all_entails, 50)

    min_questeval = min(all_questevals)
    max_questeval = max(all_questevals)
    median_questeval = np.percentile(all_questevals, 50)

    min_rouge = min(all_rouges)
    max_rouge = max(all_rouges)
    median_rouge = np.percentile(all_rouges, 50)

    with open(f"{task_path}/stats.json", "w") as f:
        json.dump({"entail": [min_entail, max_entail, median_entail], 
                   "questeval": [min_questeval, max_questeval, median_questeval], 
                   "rouge": [min_rouge, max_rouge, median_rouge]}, f)

    sources = list(os.listdir(task_path))
    for source in tqdm(sources, total=len(sources)):
        if not os.path.isdir(f"{task_path}/{source}/"):
            continue
        for split in ['train', 'val', 'test']:
            data = pd.read_csv(f"{task_path}/{source}/{model_name}_{split}/logits_with_metrics.csv")

            data['entail_100'] = data['nli'].apply(lambda x: float(x.split(": ")[1].split("/")[0]) / 100)
            data['standarded_entail'] = (data['entail_100'] - min_entail) / (max_entail - min_entail)
            data['discrete_entail'] = data['entail_100'] > median_entail
            data['rank_entail'] = percentileofscore(all_entails, data['entail_100'])/100

            data['standarded_questeval'] = (data['questeval'] - min_questeval) / (max_questeval - min_questeval)
            data['discrete_questeval'] = data['questeval'] > median_questeval
            data['rank_questeval'] = percentileofscore(all_questevals, data['questeval']) / 100

            data['rouge_100'] = data['rouge'] / 100
            data['standarded_rouge'] = (data['rouge_100'] - min_rouge) / (max_rouge - min_rouge)
            data['discrete_rouge'] = data['rouge_100'] > median_rouge
            data['rank_rouge'] = percentileofscore(all_rouges, data['rouge_100']) / 100
            
            data.to_csv(f"{task_path}/{source}/{model_name}_{split}/logits_metrics_stats.csv", index=False)




def label_data_comprehensive(file_path, hallucinated_out_path, ok_out_path, splits=['train', 'val', "test"]):
    hallucinated_ids, ok_ids = [], []
    nli_score_persource, questeval_score_persource, rouge_score_persource = [], [], []
    ids_persource = []
    for split in splits:
        # print(split)
        data = pd.read_csv(file_path.format(split=split))
        for index, row in data.iterrows():
            if (not pd.isna(row['generated_text'])) and str(row['generated_text']).strip():
                # id,question,answer,generated_text,nli,questeval,bleu,rouge
                nli = int(row['nli'].split(": ")[0])
                questeval = float(row['questeval'])
                # bleu = float(row['bleu'].split("/")[0])
                rouge = float(row['rouge'])
                if row['id'] in ids_persource:
                    print(file_path.format(split=split))
                    print(row['id'], len(ids_persource))
                assert row['id'] not in ids_persource
                ids_persource.append(row['id'])
                nli_score_persource.append(nli)
                questeval_score_persource.append(questeval)
                # bleu_score_persource.append(bleu)
                rouge_score_persource.append(rouge)
                               
    ####
    questeval_thre = np.percentile(questeval_score_persource, 50)
    # bleu_thre = np.percentile(bleu_score_persource, 50)
    rouge_thre = np.percentile(rouge_score_persource, 50)
    assert len(ids_persource) == len(nli_score_persource) == len(questeval_score_persource) == len(rouge_score_persource)
    for id, nli, questeval, rouge in zip(ids_persource, nli_score_persource, questeval_score_persource, rouge_score_persource):
        if nli in [2] and questeval < questeval_thre and rouge < rouge_thre: # and bleu < bleu_thre   
            if id in hallucinated_ids:
                print("repeat", id)
            assert id not in hallucinated_ids
            hallucinated_ids.append(id)
        elif nli in [0] and questeval > questeval_thre and rouge > rouge_thre: # and bleu > bleu_thre
            if id in ok_ids:
                print("repeat", id)
            assert id not in ok_ids
            ok_ids.append(id)
            
    with open(hallucinated_out_path, 'w') as f:
        for id in hallucinated_ids:
            f.write(f"{id}\n")
    with open(ok_out_path, 'w') as f:
        for id in ok_ids:
            f.write(f"{id}\n")

    print("hallucinated num", len(hallucinated_ids), "ok num",  len(ok_ids))
    # print("nli", np.mean(nli_score_persource), np.percentile(nli_score_persource, 10), np.percentile(nli_score_persource, 90)) 
    # print("bleu1", np.mean(bleu_score_persource), np.percentile(bleu_score_persource, 10), np.percentile(bleu_score_persource, 90))  
    # print("rouge", np.mean(rouge_score_persource), np.percentile(rouge_score_persource, 10), np.percentile(rouge_score_persource, 90))   
    # plt.hist(questeval_score_persource, bins=50)
    # plt.title("questeval")
    # plt.show()
    # plt.title("bleu1")
    # plt.hist(bleu_score_persource, bins=50)
    # plt.show()
    # plt.title("rouge")
    # plt.hist(rouge_score_persource, bins=50)
    # plt.show()

def write_data(task_path, out_dir, model_name, splits=['train', 'val', "test"]):
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        file_list = []
        sources = list(os.listdir(task_path))
        for source in tqdm(sources, total=len(sources)):
            if not os.path.isdir(f"{task_path}/{source}/"):
                continue
            data_per_source = pd.read_csv(f"{task_path}/{source}/{model_name}_{split}/logits_metrics_stats.csv")
            data_per_source['source'] = source
            data_per_source.drop(columns=['answer', "generated_text", "nli", "rouge"], inplace=True)
            file_list.append(data_per_source)

        data = pd.concat(file_list, ignore_index=True)
        data.to_csv(f"{out_dir}/{model_name}_{split}.csv", index=False)


def write_data_comprehensive(task_path, out_dir, model_name, id_list={}, sources=[], splits=['train', 'val', "test"]):
    # for one task include multiple sources
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        file_list = []
        if not sources:
            sources = list(os.listdir(task_path))
        for source in tqdm(sources, total=len(sources)):
            if not os.path.isdir(f"{task_path}/{source}/"):
                continue
            if id_list:
                id_path = id_list.format(source=source, label='hallucinated')
                if os.path.exists(id_path):
                    with open(id_path, 'r') as f:
                        hallucinated_ids = f.read().strip().split("\n")
                    id_path = id_list.format(source=source, label='ok')
                    with open(id_path, 'r') as f:
                        ok_ids = f.read().strip().split("\n")
                else:
                    print('no id list', id_path)
                    continue
            else:
                with open(f"{task_path}/{source}/{model_name}_train/hallucinated_ids.txt", 'r') as f:
                    hallucinated_ids = f.read().strip().split("\n")
                with open(f"{task_path}/{source}/{model_name}_train/ok_ids.txt", 'r') as f:
                    ok_ids = f.read().strip().split("\n")

            data_per_source = pd.read_csv(f"{task_path}/{source}/{model_name}_{split}/logits.csv")
            data_per_source.drop(columns=['answer', "generated_text"], inplace=True)
            data_per_source['source'] = source
            data_per_source['label'] = ""
            for index, row in data_per_source.iterrows():
                if row['id'] in hallucinated_ids:
                    data_per_source.at[index, 'label'] = "hallucinated"
                elif row['id'] in ok_ids:
                    data_per_source.at[index, 'label'] = "ok"
            data_per_source = data_per_source.loc[data_per_source['label'].isin(['hallucinated', 'ok']),:] 
            file_list.append(data_per_source)
        ########################################################################
        data = pd.concat(file_list, ignore_index=True)
        os.makedirs(out_dir, exist_ok=True)
        data.to_csv(f"{out_dir}/{model_name}_{split}_comprehensive.csv", index=False)
