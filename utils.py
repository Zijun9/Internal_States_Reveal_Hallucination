import requests
import re
import time
import json
import csv
import os
import bz2
import pickle
import _pickle as cPickle

try:
    import openai
except:
    print("openai not installed")
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM


FULL_TASKS_LIST = ["Code_to_Text", "Data_to_Text", "Dialogue_Generation", "Explanation", "Grammar_Error_Correction", "Number_Conversion", "Overlap_Extraction", "Paraphrasing", "Preposition_Prediction", "Program_Execution", "Question_Answering", "Sentence_Compression", "Summarization", "Text_to_Code", "Title_Generation", "Translation"]

def process_qa(text):
    q1_res = re.search("Question 1: ", text).span()
    a1_res = re.search("Answer 1: ", text).span()
    q1 = text[q1_res[1]:a1_res[0]].strip()
    q2_res = re.search("Question 2: ", text).span()
    a1 = text[a1_res[1]:q2_res[0]].strip()
    a2_res = re.search("Answer 2: ", text).span()
    q2 = text[q2_res[1]:a2_res[0]].strip()
    q3_res = re.search("Question 3: ", text).span()
    a2 = text[a2_res[1]:q3_res[0]].strip()
    a3_res = re.search("Answer 3: ", text).span()
    q3 = text[q3_res[1]:a3_res[0]].strip()
    a3 = text[a3_res[1]:].strip()

    return [q1, q2, q3], [a1, a2, a3]


def get_lens(path, tokenizer):
    lens = []
    if path.endswith("csv"):
        with open(path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                question = line[1]
                lens.append(len(tokenizer.tokenize(question)))
    elif path.endswith("jsonl"):
        with jsonlines.open(path) as f:
            for l in f:
                qs, ans = process_qa(l["reply"])
                for q in enumerate(qs):
                    lens.append(len(tokenizer.tokenize(q[1])))
                    
    return lens

def init_model(model_name, device, padding_side="left", load_model=True):
    print("device", device)
    if model_name == 'Llama2-7B':
        model_path = "/share/jiziwei/Llama-2-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") #
    elif model_name == "Llama2-7B-Chat":
        model_path = "/share/jiziwei/Llama-2-7b-chat-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") #
    elif model_name == "Llama-3.1-8B-Instruct":
        model_path = "/share/jiziwei/Meta-Llama-3.1-8B-Instruct" 
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda:0") #
    elif model_name == "Mistral-7B-Instruct":
        model_path = "/share/jiziwei/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") 
    
    tokenizer.pad_token = tokenizer.eos_token 
    if load_model:
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
    else:
        model = None
    return model, tokenizer

def if_source_summary(generated_path):
    sums = ['aeslc', 'multi_news', "samsum", "ag_news_subset", "newsroom", "gem_wiki_lingua_english_en", "cnn_dailymail", "opinion_abstracts_idebate", "opinion_abstracts_rotten_tomatoes", "huggingface", "gigaword"]
    sums += ["Summarization"]
    return any([s in generated_path.split("/") for s in sums])

def process_layers_to_process(layers_to_process):
    if not layers_to_process:
        layers_to_process2 = []
    if len(layers_to_process) == 1 and "range" in layers_to_process[0]: #range33
        layers_to_process2 = sorted(list(eval(layers_to_process[0])))
    else:
        layers_to_process2 = sorted([int(x) for x in layers_to_process])
    return layers_to_process2


def get_batch_generate(prompts, model, tokenizer, max_new_tokens, max_token=2048):
    if max_token == -1:
        max_token = float("inf")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    # print("input_length", input_length)
    if input_length < max_token:
        generated_texts = sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens)
    else:
        generated_texts = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            if input_length < max_token:
                generated_texts += sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens)
            else:
                print(f"INPUT:\n{prompt}\ntoo long")
                generated_texts.append("")

    return generated_texts


def sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens):
    with torch.no_grad():
        generated_ids = model.generate(**inputs, 
                                        num_beams=1, do_sample=False, top_p=1.0, temperature=1.0,
                                        max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts




def compressed_pickle(file_path, data):
    with bz2.BZ2File(file_path, "w") as f:
        cPickle.dump(data, f)

def decompress_pickle(file_path):
    data = bz2.BZ2File(file_path, "rb")
    data = cPickle.load(data)
    return data
