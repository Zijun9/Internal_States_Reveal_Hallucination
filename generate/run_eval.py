import os
path = os.getcwd()
root_path = "/".join(path.split("/")[:-1])
import collections
from datasets import load_metric
import evaluate
from nltk.tokenize import word_tokenize
import argparse
import re
import csv
import pandas as pd
import ast
import argparse
import math
import numpy as np
from tqdm import tqdm
import string
import torch
import shutil
import sys
sys.path.append(root_path)
sys.path.append(root_path+"/QuestEval")
# try:
#     nlp = spacy.load("en_core_web_sm")
#     from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
# except:
#     print("lack package")

try:
    import spacy
    from questeval.questeval_metric import QuestEval
except:
    print("lack packages")

from src.utils import if_source_summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import warnings
warnings.filterwarnings('ignore')

def get_ngrams(tokens, n, language='zh'):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]  # list of str

def get_ngram_counter(tokens, n):
    ngrams = get_ngrams(tokens, n)
    counter = collections.Counter()
    counter.update(ngrams)
    return counter


def _prec_recall_f1_score(pred_items, gold_items, language, n=1):
    assert language in ["en", "zh"]
    if language == "en":
        pred_items = pred_items.split()
        gold_items = gold_items.split()
        
    pred_items = get_ngram_counter(pred_items, n)
    gold_items = get_ngram_counter(gold_items, n)
    common = gold_items & pred_items
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# https://github.com/baiyyang/BLEU    
def calculate_ngram(candidates, references, n, language):
    count_clip = 0
    count = 0
    for index, candidate in enumerate(candidates):
        references_list = lines2dic(references, index, n, language)
        if language == "en":
            words = candidate.split()
        else:
            words = candidate
        limit = len(words) - n + 1
        candidate_dic = {}
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in candidate_dic.keys():
                candidate_dic[key] += 1
            else:
                candidate_dic[key] = 1
        count_clip += clip(candidate_dic, references_list)
        count += limit
    if count_clip == 0:
        pr = 0
    else:
        pr = float(count_clip) / count
    return pr


def brevity_penalty(candidates, references, language):
    c = 0
    r = 0
    for index, candidate in enumerate(candidates):
        c_length = len(candidate.split()) if language == "en" else len(candidate)
        reference_index = [reference[index] for reference in references]
        r_lengths = [len(r.split()) if language == "en" else len(r) for r in reference_index]
        c += c_length
        r += match_reference(c_length, r_lengths)
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - float(r) / c)
    return bp


def match_reference(candidate_len, reference_lens):
    """
    计算当c<=r时，最佳匹配的r的长度
    :param candidate_len:
    :param reference_lens:
    :return:
    """
    best_len = abs(reference_lens[0] - candidate_len)
    best_ref = reference_lens[0]
    for length in reference_lens:
        if abs(length - candidate_len) < best_len:
            best_len = abs(length - candidate_len)
            best_ref = length
    return best_ref


def clip(candidate, references):
    count = 0
    for cand in candidate.keys():
        cand_value = candidate[cand]
        max_count = 0
        for reference in references:
            if cand in reference.keys():
                max_count = max(reference[cand], max_count)
        count += min(max_count, cand_value)
    return count

def lines2dic(references, index, n, language):
    reference_list = []
    for reference in references:
        reference_dic = {}
        line = reference[index]
        if language == "en":
            words = line.split()
        else:
            words = line
        limit = len(words) - n + 1
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in reference_dic.keys():
                reference_dic[key] += 1
            else:
                reference_dic[key] = 1
        reference_list.append(reference_dic)
    return reference_list

def remove_punctuation(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub("\s+", ' ', s).strip()
    return s
    

def get_pre(row):
    if type(row['answer']) == list:
        answer = row['answer'][0]
    else:
        answer = row['answer']
    ref = remove_punctuation(answer).lower()
    generated_text = str(row['generated_text']).strip()
    generated_text = re.sub("<pad> ?", "", generated_text)
    doc = remove_punctuation(generated_text).lower()
    precision1, _, _ = _prec_recall_f1_score(ref, doc, language, n=1)
    precision1 = round(precision1,2)
    precision2, _, _ = _prec_recall_f1_score(ref, doc, language, n=2)
    precision2 = round(precision2,2)
    precision3, _, _ = _prec_recall_f1_score(ref, doc, language, n=3)
    precision3 = round(precision3,2)
    return precision1, precision2, precision3


def get_selfchckeNLI(NLI_model, doc, ref):
    doc = str(doc)
    if type(ref) != list:
        ref = [ref]

    sentences = [sent.text.strip() for sent in nlp(doc).sents]
    sent_scores_nli = NLI_model.predict(
        sentences = sentences,
        sampled_passages = ref,
    )
    return sent_scores_nli

def process_generated_list(generated_list):
    generated_list2 = []
    for generated_text in generated_list:
        if pd.isna(generated_text):
            generated_text = ""
        generated_text = str(generated_text).strip()
        generated_text = re.sub("<pad> ?", "", generated_text)
        generated_list2.append(generated_text)
    return generated_list2

def get_NLI(NLI_model, premise, hypothesis, device, tokenizer):
    # time1 = time.time()
    premise2 = []
    for p in premise:
        if type(p) == list:
            if p:
                if type(p[0]) in [float, int]:
                    p[0] = str(p[0])
                premise2.append(p[0])
            else:
                premise2.append("")
        elif type(p) == str:
            premise2.append(p)
        else:
            assert False
    hypothesis2 = process_generated_list(hypothesis)
    # try:
    input = tokenizer(premise2, hypothesis2, truncation=True, padding=True, return_tensors="pt")
    # except:
    #     print("premise")
    #     print(premise)
    #     print("premise2")
    #     for p in premise2:
    #         print(type(p), p)
    #     print("hypothesis")
    #     for p in hypothesis2:
    #         print(type(p), p)
    #     assert False
    output = NLI_model(input["input_ids"].to(device))
    #["entailment", "neutral", "contradiction"]
    prediction = torch.argmax(output["logits"], dim=1).cpu().tolist()
    scores = torch.softmax(output["logits"], dim=1)
    scores = torch.mul(100, scores).cpu().tolist()
    # time2 = time.time()
    # print("NLI TIME", time2-time1)
    return prediction, scores 

def get_questeval_score(questeval, question_list, answer_list, generated_list):
    # time1 = time.time()
    generated_list = process_generated_list(generated_list)
    answer_list2 = []
    for a in answer_list:
        if type(a) == list:
            answer_list2.append(a)
        else:
            answer_list2.append([a])
    score = questeval.corpus_questeval(
                hypothesis=generated_list, 
                sources=question_list,
                list_references=answer_list2)
    # time2 = time.time()
    # print("questeval TIME", time2-time1)
    return score['ex_level_scores']

def get_ppl_score(ppl_scorer, question_list, generated_list):
    # time1 = time.time()
    generated_list = process_generated_list(generated_list)
    q_scores = ppl_scorer.compute(predictions=question_list)['perplexities']
    g_scores = ppl_scorer.compute(predictions=generated_list)['perplexities']
    assert len(question_list) == len(generated_list) == len(q_scores) == len(g_scores)
    # time2 = time.time()
    # print("ppl TIME", time2-time1)
    return q_scores, g_scores

def get_bleu_score(bleu_scorer, row):
    bleu_generated_txt = str(row['generated_text'])
    bleu_generated_txt = bleu_generated_txt.strip().lower()
    bleu_generated_txt = re.sub("<pad> ?", "", bleu_generated_txt)
    
    if not (bleu_generated_txt and row['answer']):
        return ["0", "0", "0"]
    
    bleu_generated_txt = [word_tokenize(bleu_generated_txt)]
    
    if type(row['answer']) == list:
        bleu_target_txt = []
        for a in row['answer']:
            bleu_target_txt.append(word_tokenize(a.strip().lower()))
    else:
        bleu_target_txt = [word_tokenize(row['answer'].strip().lower())]

    bleus = []
    for i in range(1, 4):
        bleu_dict = bleu_scorer.compute(predictions=bleu_generated_txt, references=bleu_target_txt, max_order=i)
        bleus.append(str(round(bleu_dict['bleu']*100, 2)))
    return bleus


def get_rouge_score(rouge_scorer, row):
    rouge_generated_txt = str(row['generated_text'])
    rouge_generated_txt = re.sub("<pad> ?", "", rouge_generated_txt).strip()
    rouge_generated_txt = [rouge_generated_txt.lower()]

    if type(row['answer']) == list:
        rouge_target_txt = []
        for a in row['answer']:
            rouge_target_txt.append(a.strip().lower())
    else:
        rouge_target_txt = [row['answer'].strip().lower()]
        
    rouge_dict = rouge_scorer.compute(predictions=rouge_generated_txt, references=rouge_target_txt)
    r = rouge_dict["rougeL"].mid.fmeasure*100
    return round(r, 2)



def process_answer(answer):
    if type(answer) in [int, float, str]:
        return str(answer)
    
    assert type(answer) == str
    if "[" in answer:
        answer = ast.literal_eval(answer)

    return answer

def get_history_data(args, output_file):
    history_data = {}
    if not args.from_scratch and os.path.isfile(output_file):
        copy_history_data = {}
        if os.path.isfile(output_file+"_copy"):
            with open(output_file+"_copy") as f:
                reader = csv.DictReader(f)
                for index, row in enumerate(reader):
                    copy_history_data[row['id']] = [index, row]
        else:
            shutil.copyfile(output_file, output_file+"_copy")

        with open(output_file) as f:
            reader = csv.DictReader(f)
            for index, row in enumerate(reader):
                history_data[row['id']] = [index, row]

        for key, value in copy_history_data.items():
            if key not in history_data: # output_file_copy 比  output_file 数据多
                history_data[key] = value #中途被打断 导致questeval丢失
    print("len history_data", len(history_data))
    return history_data

def get_batch_res(metric, batch_rows, use_input, NLI_model, questeval, ppl_scorer, device, tokenizer):
    assert metric in ["nli", "questeval", "ppl"]
    process_ids, question_list, answer_list, generated_list = [], [], [], []
    for row in batch_rows:
        if not justify_if_have_score(metric, row):
            process_ids.append(row['id'])
            question_list.append(str(row['question']).strip())
            if type(row['answer']) == list:
                a = row['answer'][0].strip()
            else:
                a = row['answer'].strip()
            answer_list.append(a)
            generated_list.append(str(row['generated_text']).strip())
    assert len(question_list) == len(answer_list) == len(generated_list) == len(process_ids)

    if question_list:
        if metric == "nli":
            # print("calculate nli")
            if use_input: # input is reference
                nli_preds, nli_scores = get_NLI(NLI_model, question_list, generated_list, device, tokenizer)
            else:
                nli_preds, nli_scores = get_NLI(NLI_model, answer_list, generated_list, device, tokenizer)
        elif metric == "questeval":
            # print("calculate questeval")
            questeval_scores = get_questeval_score(questeval, question_list, answer_list, generated_list)
        elif metric == "ppl":
            q_ppl_scores, g_ppl_scores = get_ppl_score(ppl_scorer, question_list, generated_list)
    else:
        # print("exsiting", metric)
        nli_preds, nli_scores = [], []
        questeval_scores = []
        q_ppl_scores, g_ppl_scores = [], []

    text_list = []
    j = 0
    for row in batch_rows:
        if justify_if_have_score(metric, row):
            text_list.append(row[metric])
        else:
            assert row['id'] == process_ids[j]
            if metric == "nli":
                nli_text = f"{nli_preds[j]}: {round(nli_scores[j][0],2)}/{round(nli_scores[j][1],2)}/{round(nli_scores[j][2],2)}"
                text_list.append(nli_text)
            elif metric == "questeval":
                text_list.append(questeval_scores[j])
            elif metric == "ppl":
                ppl_text = f"{round(q_ppl_scores[j],2)}/{round(g_ppl_scores[j],2)}"
                text_list.append(ppl_text)
            j += 1

    assert len(batch_rows) == len(text_list)
    return text_list

def justify_if_have_score(metric, row):
    if metric in row.keys():
        if row[metric] in ['nan', -1]:
            # print(metric, row[metric], 'in nan', -1)
            return False
        if type(row[metric]) == float and math.isnan(row[metric]):
            # print(metric, row[metric], 'math.isnan')
            return False
        if pd.isna(row[metric]):
            # print(metric, row[metric], 'pd.isna')
            return False
    else:
        return False
    
    return True

def get_overall_metrics(path, metrics=["nli", "questeval", "rouge", "ppl"]):
    all_nli, all_questeval, all_rouge, all_ppl = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 2: 37.81/16.13/46.07,0.31960134928425155,10.53,79.3/44.44
            if "nli" in metrics: # entailment score
                nli = float(row["nli"].split(": ")[1].split("/")[0])
            else:
                nli = 0
            if "questeval" in metrics:
                questeval = float(row["questeval"])
            else:
                questeval = 0

            if "rouge" in metrics:
                rouge = float(row["rouge"])
            else:
                rouge = 0

            if "ppl" in metrics:
                ppl = float(row["ppl"].split("/")[0])
            else:
                ppl = 0
            
            all_nli.append(nli)
            all_questeval.append(questeval)
            all_rouge.append(rouge)
            all_ppl.append(ppl)

    mean_nli = sum(all_nli)/len(all_nli)
    mean_questeval = sum(all_questeval)/len(all_questeval)
    mean_rouge = sum(all_rouge)/len(all_rouge)
    mean_ppl = sum(all_ppl)/len(all_ppl)
    return mean_nli, mean_questeval, mean_rouge, mean_ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path",type=str,)
    parser.add_argument("--generated_dir",type=str,)
    parser.add_argument("--generate_model_name", choices=[None, 'Llama2-7B-Chat', 'Llama2-7B', "Mistral-7B-Instruct", "T0-3B"])
    parser.add_argument("--output_suffix", type=str, default="_with_metrics")
    parser.add_argument("--metrics", nargs='*', type=str, help="metric to add")
    parser.add_argument("--batch_size",type=int,)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    generated_path = args.generated_path
    generated_dir = args.generated_dir

    if generated_path and not generated_dir:
        generated_path_list = [generated_path]
    elif generated_dir and not generated_path:
        generate_model_name = args.generate_model_name
        # task level
        generated_path_list = []
        for root, dirs, files in os.walk(generated_dir, topdown=False):
            for name in files:
                if name == 'logits.csv' and \
                not re.search("(hallucinated)|(ok\W)", root) and \
                generate_model_name in root:
                    path = os.path.join(root, name)
                    generated_path_list.append(path)
        generated_path_list = sorted(generated_path_list)
        print(len(generated_path_list))
        # generated_path_list = generated_path_list[80:100]
    else:
        assert False

    metrics = args.metrics
    batch_size = args.batch_size
    
    ### load model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "nli" in metrics:
        model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        NLI_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        NLI_model.to(device)
    else:
        tokenizer = None
        NLI_model = None

    if "questeval" in metrics:
        questeval = QuestEval(no_cuda=False)
    else:
        questeval = None
        
    # if "selfchecknli" in metrics:
    #     NLI_model = SelfCheckNLI(device=device)
    # else:
    #     NLI_model = None

    if "bleu" in metrics:
        bleu_scorer = load_metric("bleu")
    if "rouge" in metrics:
        rouge_scorer = load_metric("rouge")
    if "ppl" in metrics:
        ppl_scorer = evaluate.load(f"{root_path}/generate/perplexity.py", module_type="metric")
    else:
        ppl_scorer = None

    for generated_path in generated_path_list:
        print(generated_path)
        output_file = generated_path[:-4]+f"{args.output_suffix}.csv"
        use_input = if_source_summary(generated_path) # 只对nli有用
        history_data = get_history_data(args, output_file)
        
        language = 'en'
        data = pd.read_csv(generated_path)
        try:
            data.drop(columns=['logits'], inplace=True)
        except:
            print("no logits col")
        for metric in metrics:
            data[metric] = pd.Series(dtype='object')

        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "question", "answer", "generated_text"]+metrics)

        rows_to_process = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            if 'generated_text' not in row or pd.isna(row['generated_text']):
                row['generated_text'] = ""
            if 'answer' in row:
                row['answer'] = process_answer(row['answer'])
            else:
                row['answer'] = ""

            if row['id'] in history_data:
                whole_row_fine = True
                history_index, history_row = history_data[row['id']]
                if history_row["question"] != row['question']:
                    print(row['id'])
                    print("history_row", history_row["question"])
                    print("source", row['question'])
                # assert history_row["question"] == row['question']
                metric_socres = []
                for metric in metrics:
                    if justify_if_have_score(metric, history_row):
                        metric_socres.append(history_row[metric])
                    else:
                        whole_row_fine = False
                        break

                if whole_row_fine:
                    assert len(metric_socres) == len(metrics)
                    with open(output_file, "a", encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([row["id"], row["question"], row["answer"], row["generated_text"]]+metric_socres)
                else:
                    rows_to_process.append(history_row) #!!!!!!
            else:
                rows_to_process.append(row)
        print("rows_to_process", len(rows_to_process))
              
        for i in tqdm(range(0, len(rows_to_process), batch_size)):
            ############ batch ############
            batch_rows = rows_to_process[i:i+batch_size]
            if "nli" in metrics:
                nli_text_list = get_batch_res("nli", batch_rows, use_input, NLI_model, questeval, ppl_scorer, device, tokenizer)
                # print("nli_text_list", nli_text_list)

            if "questeval" in metrics:
                questeval_text_list = get_batch_res("questeval", batch_rows, use_input, NLI_model, questeval, ppl_scorer, device, tokenizer)
                # print("questeval_text_list", questeval_text_list)

            if "ppl" in metrics:
                ppl_text_list = get_batch_res("ppl", batch_rows, use_input, NLI_model, questeval, ppl_scorer, device, tokenizer)
            ############ batch ############
                    
            for j, row in enumerate(rows_to_process[i:i+batch_size]):
                metric_socres = []
                if "pre" in metrics:
                    if justify_if_have_score("pre", row):
                        metric_socres.append(row["pre"])
                    else:
                        precision1, precision2, precision3 = get_pre(row)
                        metric_socres.append(f"{precision1}/{precision2}/{precision3}")

                if "nli" in metrics:
                    metric_socres.append(nli_text_list[j])

                if "questeval" in metrics:
                    # q_scores = get_questeval_score(questeval, [row['question']], [row['answer']], [row['generated_text']])
                    # metric_socres.append(q_scores[0])
                    metric_socres.append(questeval_text_list[j])
                    
                # if "selfchecknli" in metrics:
                #     if justify_if_have_score("selfchecknli", row):
                #         metric_socres.append(row["selfchecknli"])
                #     else:
                #         if type(row['answer']) == list:
                #             a = row['answer'][0].strip()
                #         else:
                #             a = row['answer'].strip()

                #         if use_input:
                #             l1 = len(str(row['question']).split())
                #         else:
                #             l1 = len(a.split())
                #         l2 = len(str(row['generated_text']).split())
                #         if max(l1, l2) > 9000:
                #             print("skip", l1, l2)
                #             metric_socres.append(-1)
                #         else:
                #             if use_input:
                #                 sent_scores_nli = get_selfchckeNLI(NLI_model, row['generated_text'], row['question'])
                #             else:
                #                 sent_scores_nli = get_selfchckeNLI(NLI_model, row['generated_text'], a)
                #             sent_scores_nli = [str(round(i*100,2)) for i in sent_scores_nli]
                #             metric_socres.append("/".join(sent_scores_nli))
                    
                if "bleu" in metrics:
                    if justify_if_have_score("bleu", row):
                        b_score = row["bleu"]
                    else:
                        bleu_scores = get_bleu_score(bleu_scorer, row)
                        b_score = "/".join(bleu_scores)
                    metric_socres.append(b_score)
                    # except Exception as e: 
                    #     print(e)
                    #     print("bleu_scores", bleu_scores)
                    #     assert False

                if "rouge" in metrics:
                    if justify_if_have_score("rouge", row):
                        r_score = row["rouge"]
                    else:
                        # print("calculate rouge")
                        r_score = get_rouge_score(rouge_scorer, row)
                    # print("rouge", r_score)
                    metric_socres.append(r_score)

                if "ppl" in metrics:
                    metric_socres.append(ppl_text_list[j])

                with open(output_file, "a", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([row["id"], row["question"], row["answer"], row["generated_text"]]+metric_socres)

        # ######### filter #########
        # if metric == "pre":
        #     output_file2 = args.generated_path[:-4]+f"_with_{metric}_filtered.csv"
        #     with open(output_file) as f, \
        #         open(output_file2, 'w') as fout:
        #         lines = list(csv.reader(f))
        #         head = lines[0]
        #         writer = csv.writer(fout)
        #         writer.writerow(head)

        #         for line in lines[1:]:
        #             if line[-1] == '0/0/0' and not re.search("said", line[1]):
        #                 writer.writerow(line)
                        