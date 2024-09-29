import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import pandas as pd
import logging
from tqdm import tqdm
import argparse
import sys
import csv
csv.field_size_limit(sys.maxsize)
cwd = os.getcwd()
cwd = "/".join(cwd.split("/")[:-1])
sys.path.append(cwd)
from src.utils import init_model, get_batch_generate
from length import get_max_new_tokens
import re
import warnings
warnings.filterwarnings('ignore')

def init_out_files(args, output_dir):
    logits_output_path = f"{output_dir}/logits.csv"
    history_data = {}
    old_len = 0
    if os.path.isfile(logits_output_path) and not args.from_scratch:
        with open(logits_output_path, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                history_data[str(row[0])] = row
                
        old_len = len(history_data.keys())
                    
    with open(logits_output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'answer','generated_text'])

    print("old lens", old_len)
    return history_data, logits_output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['Llama2-7B-Chat', 'Llama2-7B', "Mistral-7B", "T0-3B"])
    parser.add_argument("--dataset_path", default='')
    parser.add_argument("--input_dir", default='')
    parser.add_argument("--output_dir", default='')
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model(model_name, device, "left")
    tokenizer.padding_side = "left" 

    max_sample = args.max_sample
    if max_sample == -1:
        max_sample = float("inf")
        
    batch_size = args.batch_size
    dataset_path = args.dataset_path
    input_dir = args.input_dir
    
    if dataset_path and not input_dir:
        output_dir_list = [args.output_dir] #具体到split
        dataset_path_list = [dataset_path]
    elif input_dir and not dataset_path:
        # task level
        dataset_path_list, output_dir_list = [], []
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                dataset_path_list.append(path)
                #args.output_dir 到根目录 具体到corpus, 不用提task
                items = path.split("/")
                split = items[-1].split(".")[0]
                source = items[-2]
                task = items[-3]
                path2 = f"{args.output_dir}/{task}/{source}/{model_name}_{split}"
                output_dir_list.append(path2)
    else:
        assert False

    print("dataset_path_list", dataset_path_list)
    print("output_dir_list", output_dir_list)
    # assert False
    for dataset_path, output_dir in tqdm(zip(dataset_path_list, output_dir_list)):
        print('dataset_path', dataset_path)
        print('output_dir', output_dir)
        if (args.max_sample == 8000) and re.search("\/((test)|(val))", dataset_path):
            max_sample = 1000
        if (args.max_sample == 8000) and re.search("\/train", dataset_path):
            max_sample = args.max_sample
            
        os.makedirs(output_dir, exist_ok=True)
        history_data, logits_output_path = init_out_files(args, output_dir)
        # print(history_data.keys())
        
        # max_new_tokens = get_max_new_tokens(dataset_path) # only for natural
        # if not max_new_tokens:
        max_new_tokens = args.max_new_tokens
        print('max_new_tokens', max_new_tokens)

        if dataset_path.endswith('csv'):
            dataset = pd.read_csv(dataset_path, encoding="utf8")
        elif dataset_path.endswith('jsonl'):
            dataset = pd.read_json(dataset_path, lines=True, encoding="utf8")
        else:
            print("wrong file type")
            continue

        rows_to_process = []
        for i, row in dataset.iterrows():
            row['id'] = str(row['id'])
            if row['id'] in history_data.keys():
                generated_row = history_data[row['id']]
                with open(logits_output_path, "a", encoding="utf8") as f:
                    writer = csv.writer(f)
                    writer.writerow(generated_row)
                    
            else:
                if i >= max_sample:
                    break
                # if "train" not in args.dataset_path:
                #     if 'source' in row and row['source'] != 'wiki':
                #         continue
                rows_to_process.append(row)
                
        print("rows_to_process", len(rows_to_process)) 
        print("batch generate")
        for i in tqdm(range(0, len(rows_to_process), batch_size)):
            ids, questions, answers = [], [], []
            for j in range(i, min(i+batch_size, len(rows_to_process))):
                row = rows_to_process[j]
                ids.append(row['id'])
                questions.append(row['question'])
                answers.append(row['answer'])
            # print("questions", questions)

            if model_name in ['Llama2-7B-Chat', "Mistral-7B"]:
                prompts = []
                for text in questions:
                    prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{text}'}], tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
            else:
                prompts = questions
            generated_texts = get_batch_generate(prompts, model, tokenizer, max_new_tokens)
            assert len(generated_texts) == len(questions)
            with open(logits_output_path, "a", encoding='utf8') as f:
                writer = csv.writer(f)
                for id, question, answer, generated_text in zip(ids, questions, answers, generated_texts):
                    generated_text = re.sub("<pad> ?", "", generated_text)
                    if generated_text.strip():
                        writer.writerow([id, question, answer, generated_text])
                    
if __name__ == "__main__":
    main()
