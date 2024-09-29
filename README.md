This is the Source Code of Paper: [LLM Internal States Reveal Hallucination Risk Faced With a Query](http://arxiv.org/abs/2407.03282)

# Environment settings
```
conda create --name internal python=3.8.17
conda activate internal
pip install -r requirements.txt
```


# Datasets
BBC: https://huggingface.co/datasets/RealTimeData/bbc_news_alltime

Natural-Instructions: https://github.com/allenai/natural-instructions


# Generate
Internal_States_Reveal_Hallucination/generate/

```
MODEL='Llama2-7B-Chat'
for TASK in "Question_Answering"
do
CUDA_VISIBLE_DEVICES=2 python generate/generate_answer.py \
--input_dir datasets/natural-instructions-2.8/processed/${TASK} \
--output_dir NI_output/split_trans \
--model_name $MODEL \
--max_sample -1 \
--batch_size 16 \
--max_new_tokens 500 \
--from_scratch
done
```

# Save Hidden state
Internal_States_Reveal_Hallucination/generate/

```
MODEL='Llama2-7B-Chat'
for TASK in "Question_Answering"
do
((GPUID++)) 
echo $TASK
CUDA_VISIBLE_DEVICES=$GPUID python save_hidden_states.py \
--input_dir generate/NI_output/${TASK} \
--output_dir NI_output \
--layers_to_process 0 1 2 15 16 17 30 31 32 \
--model_name $MODEL \
--max_sample -1 \
--batch_size 16 \
--delete_redundancy &
done
```

# Evaluate generated text
Internal_States_Reveal_Hallucination/generate/

```
for TASK in "Question_Answering"
do
echo $TASK
# ((GPUID++)) 
CUDA_VISIBLE_DEVICES=$GPUID python generate/run_eval.py \
--generated_dir NI_output/${TASK} \
--metrics "nli" "questeval" "rouge" "ppl" \
--generate_model_name 'Llama2-7B-Chat' \
--batch_size 16
done
```


# Train classifier
Internal_States_Reveal_Hallucination/classifier/build_data.ipynb

```
for TASK in "Question_Answering"
do 
for MODEL in LlamaMLP
do
HSTYPE="last"
CUDA_VISIBLE_DEVICES=5 python train_feedforward_classifier.py \
--classifier_type $MODEL \
--layers_to_process 32 \
--input_path_o NI/${TASK}/dataset/Llama2-7B-Chat_{split}_comprehensive.csv \
--source_dirs "generate/NI_output/"${TASK} \
--generate_model_name "Llama2-7B-Chat" \
--labels "label" \
--batch_size 128 \
--lr 1e-5 \
--train_from_scratch \
--hidden_state_type ${HSTYPE} \
--ignore_missing_hs \
--save_dir_root classifier/NI/${TASK} \
--training_epoch 20

done
done
```
