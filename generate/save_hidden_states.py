import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import sys
import pickle
cwd = os.getcwd()
root_path = "/".join(cwd.split("/")[:-1])
sys.path.append(root_path)
from src.utils import init_model, process_layers_to_process, decompress_pickle
import re
sys.path.append(f"{root_path}/classifier")
from prompt_eval import PROMPT_Q

def get_batch_hidden_states(ids, texts, model, tokenizer, layers_to_process, model_name, type, save_attention=False, ignore_nan=False):
    if model_name in ['Llama2-7B-Chat', "Mistral-7B-Instruct"]:
        prompts = []
        for prompt in texts:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
    else:
        prompts = texts
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model(**inputs,
                        output_hidden_states=True, output_attentions=save_attention,)
    # logits = outputs.logits # [batch, sequence_len, 32001vab]
    hidden_state = defaultdict(dict)
    for layer in layers_to_process:
        if type == 'last': # token
            all_last_token_hidden_states_layer = outputs.hidden_states[layer][:, -1, :].cpu()
        elif type in ['mean', "each"]: # token
            tmp = outputs.hidden_states[layer]
            # calculate mean of token's hidden states without padding
            mask = inputs.attention_mask.unsqueeze(-1).expand(tmp.shape)
            mask = mask.float()
            mask_tmp = tmp * mask
            if type == 'mean':
                all_mean_token_hidden_states_layer = mask_tmp.sum(dim=1) / mask.sum(dim=1)
                all_mean_token_hidden_states_layer = all_mean_token_hidden_states_layer.cpu()
            elif type == 'each':
                all_token_hidden_states_layer = mask_tmp.cpu()
        else:
            raise ValueError(f"{type} is wrong type")
        # last/mean.shape == (len(ids), 4096), each.shape == (len(ids), seq_len, 4096)
        for bidx, id in enumerate(ids): # len(ids) == batch_size
            if type == 'last':
                h = all_last_token_hidden_states_layer[bidx] # (4096,)
            elif type == 'mean':
                h = all_mean_token_hidden_states_layer[bidx] # (4096,)
            elif type == 'each':
                h = all_token_hidden_states_layer[bidx] # (seq_len, 4096)

            if torch.isnan(h).all():
                print("nan", layer, input_length)
                print("id", id)
                if type == 'last':
                    last_bad_ids = ["task1658-648ab7702b94457fab9027df4188312d", 'task1296-3642735cefa74e9885eed9fb63136142', "task522-8fdd2b5db478471282310a42717b0e52", "task170-dfc804facf1740739855a534c3927abc", "task170-0b61451206e34c4ba0ce48d3e06f7e24", "task170-5403b00a811240efa87a5ab60b77d4c1", "task170-fd5432ea188341cea01e75fdd846dc53", "task170-603d59a2efa24410bc97a53b4a3ed690", "task170-89848daa87764e2091113cb4226b543e"]
                    if model_name=='Llama2-7B-Chat' and id in last_bad_ids and layer == 32:
                        if id.startswith("task1658"):
                            bad_file = f"NI_output/Summarization/task1658_billsum_summarization/{model_name}_train/{id}_32_h.pkl"
                        elif id.startswith("task1296"):
                            bad_file = f"NI_output/Question_Answering/task1296_wiki_hop_question_answering/{model_name}_train/{id}_32_h.pkl"
                        elif id.startswith("task522"):
                            bad_file = f"NI_output/Question_Answering/task522_news_editorial_summary/{model_name}_train/{id}_32_h.pkl"
                        elif id.startswith("task170"):
                            bad_file = f"NI_output/Question_Answering_unfinished/task170_hotpotqa_answer_generation/{model_name}_train/{id}_32_h.pkl"
                        with open(bad_file, "rb") as f:
                            h = pickle.load(f)
                    else:
                        if ignore_nan:
                            continue
                        else:
                            assert False
                    
                    hidden_state[id][layer] = h # float 16
                elif type in ['mean', 'each']:
                    if ignore_nan:
                        continue
                    else:
                        assert False
                    # mean_bad_ids = [""]
                    # if model_name=='Llama2-7B-Chat' and id in mean_bad_ids and layer == 32:
                    #     if id.startswith("task170"):
                    #         bad_file = 
                    #     with open(bad_file, "rb") as f:
                    #         h = pickle.load(f)
                    # else:
                    #     assert False
            else:
                hidden_state[id][layer] = h # float 16
    return hidden_state

def init_out_files(args, output_path, layers_to_process):
    if_resave = False
    old_len = 0
    history_data = defaultdict(dict)
    if os.path.isfile(output_path) and not args.from_scratch:
        try:
            with open(output_path, "rb") as f:
                history_data = pickle.load(f)
                args.resave = False
        except Exception as e:
            print(e)
            args.resave = True
            try:
                history_data = decompress_pickle(output_path)
            except Exception as e:
                print(e)
                return history_data, True
        for id in list(history_data.keys()):
            data = history_data[id]
            for layer in layers_to_process:
                if layer in data:
                    h = data[layer]
                    if torch.isnan(h).all():
                        print("nan", id, layer)
                        history_data.pop(id)
                        break
                else:
                    history_data.pop(id)
                    break

            if id in history_data and args.delete_redundancy:
                for layer in list(history_data[id].keys()):
                    if layer not in layers_to_process:
                        history_data[id].pop(layer)
                        if_resave = True

        old_len = len(history_data)

    print("old lens", old_len)
    return history_data, if_resave



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['Llama2-7B-Chat', 'Llama2-7B', "Mistral-7B-Instruct", "T0-3B"])
    parser.add_argument("--dataset_path", default='')
    parser.add_argument("--input_dir", default='')
    parser.add_argument("--output_dir", default='')
    parser.add_argument("--type", choices=['last', 'mean'])
    parser.add_argument("--selected_ids", default='')
    parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--layers_to_process", nargs='*', type=str)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--delete_redundancy", action="store_true")
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--resave", action="store_true")
    parser.add_argument("--ignore_nan", action="store_true")
    parser.add_argument("--roughly_filter_history", action="store_true", help="roughly check whether hidden_state.pkl exist")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sub", type=int, default=-1)
    parser.add_argument("--subbatch", type=int, default=-1)
    args = parser.parse_args()

    model_name = args.model_name
    save_attention = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model(model_name, device, "left")
    tokenizer.padding_side = "left" 
    ignore_nan = args.ignore_nan

    max_sample = args.max_sample
    if max_sample == -1:
        max_sample = float("inf")
        
    batch_size = args.batch_size
    layers_to_process = process_layers_to_process(args.layers_to_process)
    print("layers_to_process", layers_to_process)
    
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
                if name == "logits.csv" and 'Llama2-7B-Chat' in root: # /share/jiziwei/natural-instructions-2.8/processed/ 的会有超长的input 导致nan
                    dataset_path = os.path.join(root, name)
                    #args.output_dir 到根目录 具体到corpus, 不用提task
                    # /home/zjiad/Hallu_source/generate/NI_output/Summarization/task618_amazonreview_summary_text_generation/Llama2-7B-Chat_val/logits.csv
                    items = dataset_path.split("/")
                    split = items[-2].split("_")[-1] #Llama2-7B-Chat_val
                    source = items[-3]
                    task = items[-4]
                    output_dir = f"{args.output_dir}/{task}/{source}/{model_name}_{split}"
                    if args.type == 'last':
                        if args.with_prompt:
                            output_path = f"{output_dir}/prompt_hidden_state.pkl"
                        else:
                            output_path = f"{output_dir}/hidden_state.pkl"
                    elif args.type == 'mean':
                        if args.with_prompt:
                            output_path = f"{output_dir}/prompt_hidden_state_mean.pkl"
                        else:
                            output_path = f"{output_dir}/hidden_state_mean.pkl"
                    if not (args.roughly_filter_history and os.path.isfile(output_path)):
                        dataset_path_list.append(dataset_path)
                        output_dir_list.append(output_dir)
        if args.sub != -1:
            print(len(dataset_path_list), args.sub, args.sub+args.subbatch)

            dataset_path_list = dataset_path_list[args.sub:args.sub+args.subbatch]
            output_dir_list = output_dir_list[args.sub:args.sub+args.subbatch]
    else:
        assert False

    selected_ids = None
    if args.selected_ids:
        with open(args.selected_ids, "r") as f:
            selected_ids = f.read().splitlines()

    print("dataset_path_list", dataset_path_list)
    print("output_dir_list", output_dir_list)
    
    for dataset_path, output_dir in tqdm(zip(dataset_path_list, output_dir_list)):
        print('dataset_path', dataset_path)
        print('output_dir', output_dir)
        if (args.max_sample == 800) and re.search("\/((test)|(val))", dataset_path):
            max_sample = 100
        if (args.max_sample == 800) and re.search("\/train", dataset_path):
            max_sample = args.max_sample
            
        os.makedirs(output_dir, exist_ok=True)
        if args.type == 'last':
            if args.with_prompt:
                output_path = f"{output_dir}/prompt_hidden_state.pkl"
            else:
                output_path = f"{output_dir}/hidden_state.pkl"
        elif args.type == 'mean':
            if args.with_prompt:
                output_path = f"{output_dir}/prompt_hidden_state_mean.pkl"
            else:
                output_path = f"{output_dir}/hidden_state_mean.pkl"
        history_data, if_resave = init_out_files(args, output_path, layers_to_process)
        
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
            if row['id'] not in history_data.keys():
                if selected_ids and row['id'] not in selected_ids:
                    continue
                rows_to_process.append(row)
                if i >= max_sample:
                    break
        print("rows_to_process", len(rows_to_process)) 
        if rows_to_process:
            print("batch save")
            for i in tqdm(range(0, len(rows_to_process), batch_size)):
                ids, questions = [], []
                for j in range(i, min(i+batch_size, len(rows_to_process))):
                    row = rows_to_process[j]
                    ids.append(row['id'])
                    if args.with_prompt:
                        q = PROMPT_Q.format(QUESTION=row['question'].strip())
                        questions.append(q)
                    else:
                        questions.append(row['question'].strip())
                try:
                    batch_hidden_state = get_batch_hidden_states(ids, questions, model, tokenizer, layers_to_process, model_name, args.type, save_attention, ignore_nan)
                except Exception as e:
                    print("error", e)
                    if args.resave or rows_to_process or if_resave:
                        print("saving...")
                        with open(output_path, "wb") as f:
                            pickle.dump(history_data, f)
                    assert False
                history_data.update(batch_hidden_state)

        if args.resave or rows_to_process or if_resave:
            print("saving...")
            with open(output_path, "wb") as f:
                pickle.dump(history_data, f)

                

if __name__ == "__main__":
    main()
