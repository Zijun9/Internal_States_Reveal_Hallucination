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
import re
from run_classification_eval import run_classification_eval
import json

ICL_PROMPT_TITLE = "Are you capable of providing an accurate response to the following instruction? Respond only to this question with 'yes' or 'no' and do not address the content of the instruction itself."
ICL_PROMPT_TAIL = """\n\nInstruction: {QUESTION}\nAnswer:"""

PROMPT_Q = """Instruction: {QUESTION}

Are you capable of providing an accurate response to the instruction given above? Respond only to this question with 'yes' or 'no' and do not address the content of the instruction itself."""
# PROMPT_QA = "Please verify the accuracy of your previous response. If it is accurate, respond with a simple 'yes'. If it is not, respond with a simple 'no'."

def init_out_files(args, predict_output_path):
    history_data = {}
    old_len = 0
    if os.path.isfile(predict_output_path) and not args.from_scratch:
        with open(predict_output_path, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                history_data[str(row[0])] = row
                
        old_len = len(history_data.keys())
                    
    with open(predict_output_path, "w") as f:
        writer = csv.writer(f)
        if args.ICL:
            head = ['id', 'question', 'ICL_prompt_eval_q']
        else:
            head = ['id', 'question', 'prompt_eval_q']
        writer.writerow(head) #, 'prompt_eval_qa'

    print("old lens", old_len)
    return history_data

def process_eval_text(eval_text):
    eval_text = re.sub("<pad> ?", "", eval_text).lower().strip()
    if "yes" in eval_text:
        eval_text = "yes"
    elif "no" in eval_text:
        eval_text = "no"
    else:
        print("eval_text", eval_text)
        eval_text = -1
    return eval_text


def get_prompt(model_name, task, tokenizer):
    MAX_LEN = 1500
    final_prompt = ICL_PROMPT_TITLE
    ok_examples = []
    hallucinated_examples = []
    with open(f"/home/zjiad/Hallu_source/classifier/NI/{task}/dataset/{model_name}_train_comprehensive.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_l = len(tokenizer.encode(row["question"]))
            if row["label"] == 'ok':
                ok_examples.append([q_l, row["question"]])
            elif row["label"] == 'hallucinated':
                hallucinated_examples.append([q_l, row["question"]])

    ok_examples.sort(key=lambda x: x[0]) 
    hallucinated_examples.sort(key=lambda x: x[0])     

    final_prompt += f"\n\nInstruction: {hallucinated_examples[0][1]}\nAnswer: no"

    next_line = f"\n\nInstruction: {ok_examples[0][1]}\nAnswer: yes"
    if len(tokenizer.encode(final_prompt+next_line+ICL_PROMPT_TAIL)) < MAX_LEN:
        final_prompt += next_line
        next_line = f"\n\nInstruction: {hallucinated_examples[1][1]}\nAnswer: no"
    else:
        return final_prompt+ICL_PROMPT_TAIL
    
    if len(tokenizer.encode(final_prompt+next_line+ICL_PROMPT_TAIL)) < MAX_LEN:
        final_prompt += next_line
        next_line = f"\n\nInstruction: {ok_examples[1][1]}\nAnswer: yes"
    else:
        return final_prompt+ICL_PROMPT_TAIL
    
    if len(tokenizer.encode(final_prompt+next_line+ICL_PROMPT_TAIL)) < MAX_LEN:
        final_prompt += next_line

    return final_prompt+ICL_PROMPT_TAIL

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Llama2-7B-Chat', choices=['Llama2-7B-Chat', 'Llama2-7B', "Mistral-7B", "T0-3B"])
    parser.add_argument("--task", default='')
    parser.add_argument("--dataset_path", default='')
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--ICL", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--predict_file", default="")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    print("args.ICL", args.ICL)
    
    if not args.predict_file:
        args.predict_file = f"{args.model_name}_{args.split}_prompt.csv"

    model_name = args.model_name
    task = args.task

    max_new_tokens = args.max_new_tokens
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model(model_name, device, "left", load_model=args.do_predict)
    tokenizer.padding_side = "left" 
    batch_size = args.batch_size
    
    max_sample = args.max_sample
    if max_sample == -1:
        max_sample = float("inf")
        
    
    dataset_path = args.dataset_path
    items = dataset_path.split("/")
    output_dir_list = ["/".join(items[:-2])+"/prompt_eval/"] # 同一个文件夹下
    dataset_path_list = [dataset_path]
    

    ICL_PROMPT_Q = get_prompt(model_name, task, tokenizer)
    print("dataset_path_list", dataset_path_list)
    print("output_dir_list", output_dir_list)
    # assert False
    for dataset_path, output_dir in tqdm(zip(dataset_path_list, output_dir_list)):
        print('dataset_path', dataset_path)
        print('output_dir', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        predict_output_path = f"{output_dir}/{args.predict_file}"

        if args.do_predict:
            history_data = init_out_files(args, predict_output_path)
            # print(history_data.keys())
            dataset = pd.read_csv(dataset_path, encoding="utf8")

            rows_to_process = []
            for i, row in dataset.iterrows():
                row['id'] = str(row['id'])
                if row['id'] in history_data.keys():
                    eval_row = history_data[row['id']]
                    with open(predict_output_path, "a", encoding="utf8") as f:
                        writer = csv.writer(f)
                        writer.writerow(eval_row)
                else:
                    if i >= max_sample:
                        break
                    rows_to_process.append(row)
                    
            print("rows_to_process", len(rows_to_process)) 
            print("batch generate")
            for i in tqdm(range(0, len(rows_to_process), batch_size)):
                ids, questions, prompts1, prompts2 = [], [], [], []
                for j in range(i, min(i+batch_size, len(rows_to_process))):
                    row = rows_to_process[j]
                    ids.append(row['id'])
                    questions.append(row['question'])
                    if args.ICL:
                        p1 = ICL_PROMPT_Q.format(QUESTION=row['question'].strip())
                        # if not i:
                        #     print(p1)
                    else:
                        p1 = PROMPT_Q.format(QUESTION=row['question'].strip())
                    if model_name in ['Llama2-7B-Chat', "Mistral-7B"]:
                        p1 = tokenizer.apply_chat_template([{"role": "user", "content": p1}], 
                                                        tokenize=False, add_generation_prompt=True)
                    prompts1.append(p1)

                    # if pd.isna(row['generated_text']):
                    #     row['generated_text'] = ""
                    # row['generated_text'] = str(row['generated_text']).strip()
                    # assert model_name in ['Llama2-7B-Chat', "Mistral-7B"]
                    # p2 = tokenizer.apply_chat_template([{"role": "user", "content": row['question'].strip()},
                    #                                     {"role": "assistant", "content": row['generated_text']},
                    #                                     {"role": "user", "content": PROMPT_QA}], 
                    #                                     tokenize=False, add_generation_prompt=True)
                    # prompts2.append(p2)
                    
                
                eval_texts1 = get_batch_generate(prompts1, model, tokenizer, max_new_tokens, max_token=-1)
                # eval_texts2 = get_batch_generate(prompts2, model, tokenizer, max_new_tokens)
                assert len(eval_texts1)  == len(ids)
                
                with open(predict_output_path, "a", encoding='utf8') as f:
                    writer = csv.writer(f)
                    for id, question, e1 in zip(ids, questions, eval_texts1): #, eval_texts2
                        prompt_eval_q = process_eval_text(e1)
                        # prompt_eval_qa = process_eval_text(e2)
                        writer.writerow([id, question, prompt_eval_q]) #, prompt_eval_qa

        if args.do_eval:
            if args.ICL:
                g_label = 'ICL_prompt_eval_q'
            else:
                g_label = 'prompt_eval_q'

            res = run_classification_eval(predict_output_path, g_label, dataset_path)
            with open(f"{output_dir}/res.json", 'w') as f:
                json.dump(res, f)




                    
if __name__ == "__main__":
    main()
