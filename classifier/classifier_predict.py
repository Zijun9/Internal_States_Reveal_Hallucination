import os
import sys
import torch
from train_feedforward_classifier import LinearClassifier1, LinearClassifier12, postiveLinearClassifier1, LlamaMLPClassifier
cwd = os.getcwd()
root_path = "/".join(cwd.split("/")[:-1])
sys.path.append(root_path)
from src.utils import init_model, get_batch_generate
import numpy as np


class ClassifierPredict:
    def __init__(self, classifier_type, input_dim, classifier_path, hidden_state_model_name, device):
        self.device = device
        self.load_classifier(classifier_type, input_dim, classifier_path)
        self.load_generate_model(hidden_state_model_name)

    def load_classifier(self, classifier_type, input_dim, classifier_path):
        self.classifier_type = classifier_type
        if classifier_type == "LlamaMLP":
            classifier = LlamaMLPClassifier(input_dim, 11008)
        elif classifier_type == "ff1":
            classifier = LinearClassifier1(input_dim)
        elif classifier_type == "ff12":
            classifier = LinearClassifier12(input_dim)
        elif classifier_type == "postiveff1":
            classifier = postiveLinearClassifier1(input_dim)
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.to(self.device)
        classifier.eval()
        self.classifier = classifier

    def load_generate_model(self, hidden_state_model_name):
        generate_model, generate_tokenizer = init_model(hidden_state_model_name, self.device, "left")
        self.generate_model = generate_model
        self.generate_tokenizer = generate_tokenizer

    def predict(self, layer, prompt, hidden_state_dims=None):
        generate_model = self.generate_model
        generate_tokenizer = self.generate_tokenizer
        classifier = self.classifier

        prompt = generate_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = generate_tokenizer(prompt, padding=True, return_tensors="pt").to(generate_model.device)
        tokens = generate_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        with torch.no_grad():
            outputs = generate_model(**inputs,
                                    output_hidden_states=True)
        
        all_token_hidden_states = outputs.hidden_states[layer][0] #batch0
        if hidden_state_dims:
            all_token_hidden_states = all_token_hidden_states[:,hidden_state_dims] #seq_len, hidden_state_dim
            
        all_pred_scores, all_preds = [], []
        assert len(tokens) == len(all_token_hidden_states)
        for per_token_hidden_states in all_token_hidden_states:
            inputs = per_token_hidden_states.unsqueeze(0).to(torch.float32)
            outputs = classifier(inputs).view(-1)
            pred_score = torch.sigmoid(outputs)
            all_pred_scores.append(pred_score.cpu().item())
            preds = torch.round(pred_score)
            all_preds.append(preds.cpu().item())
            
        return tokens, all_pred_scores, all_preds

    def generate(self, prompt, max_new_tokens):
        generate_model = self.generate_model
        generate_tokenizer = self.generate_tokenizer
        prompt = generate_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        generated_texts = get_batch_generate(prompt, generate_model, generate_tokenizer, max_new_tokens)
        return generated_texts[0]
    

    def get_activation(self, prompt, hidden_state_dims=None):
        # hidden_state_dims for all layers
        
        generate_model = self.generate_model
        generate_tokenizer = self.generate_tokenizer

        prompt = generate_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = generate_tokenizer(prompt, padding=True, return_tensors="pt").to(generate_model.device)
        tokens = generate_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        with torch.no_grad():
            outputs = generate_model(**inputs,
                                    output_hidden_states=True)
            
        hidden_states = outputs.hidden_states # [layer, batch_size, seq_len, dim]

        if hidden_state_dims:
            layer_ids, layer_dims = get_layer_id(hidden_state_dims)
            selected_hidden_states = []
            for layer, dim in zip(layer_ids, layer_dims): # selected_dim
                sh = hidden_states[layer][0, :, dim] # [seq_len]
                selected_hidden_states.append(sh) # [selected_dim, seq_len]
            selected_hidden_states = torch.stack(selected_hidden_states, dim=1)  # [seq_len, selected_dim]
            
        else:
            # change the shape of hidden_states from tuple layer of [batch_size, seq_len, dim] -> [seq_len, layer*dim]
            hidden_states = torch.cat(hidden_states, dim=2) # [layer, batch_size, seq_len, dim] -> [batch_size, seq_len, layer*dim]
            selected_hidden_states = hidden_states[0] # [seq_len, layer*dim]

        assert len(tokens) == selected_hidden_states.shape[0]
        return tokens, selected_hidden_states.cpu().numpy()


    def get_inputs_embeds(self, prompt):
        generate_model = self.generate_model
        generate_tokenizer = self.generate_tokenizer

        prompt = generate_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = generate_tokenizer(prompt, padding=True, return_tensors="pt").to(generate_model.device)
        inputs_embeds = generate_model.model.embed_tokens(inputs.input_ids).to(generate_model.device) 

        tokens = generate_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        return inputs_embeds, tokens
        
    def get_grad(self, layer, inputs_embeds, hidden_state_type, hidden_state_dims=None, target_class=0):
        generate_model = self.generate_model
        classifier = self.classifier

        inputs_embeds = inputs_embeds.to(torch.float16) # batch_size, seq_len, dim
        inputs_embeds.retain_grad()
        outputs = generate_model(inputs_embeds=inputs_embeds,
                                output_hidden_states=True)

        if len(layer) == 1:
            all_token_hidden_states = outputs.hidden_states[layer[0]] # batch_size, seq_len, dim
        else:
            all_token_hidden_states = torch.cat([outputs.hidden_states[l] for l in layer], dim=2) # batch_size, seq_len, dim
        all_token_hidden_states.retain_grad() #  batch_size, seq_len, dim
        if hidden_state_type == 'last':
            last_hidden_states = all_token_hidden_states[:,-1,:].unsqueeze(1)
            hidden_states = last_hidden_states.to(torch.float32)
        elif hidden_state_type == 'mean':
            mean_hidden_states = all_token_hidden_states.mean(dim=1).unsqueeze(1)
            hidden_states = mean_hidden_states.to(torch.float32)
        else:
            assert False
        if hidden_state_dims:
            hidden_states = hidden_states[:, :, hidden_state_dims]

        
        outputs = classifier(hidden_states) 
        if outputs.shape[2] == 1: # shape [batch1, 1, 1] tensor([[[1.7854]]],
            logits = torch.sigmoid(outputs)
        elif outputs.shape[2] == 2: # [1,1,2]
            logits = torch.softmax(outputs, dim=2)
        else:
            raise ValueError(f"classifier output shape error", outputs.shape)
        # print("logits", logits)
        if self.classifier_type == "ff12":
            target = one_hot(torch.tensor([target_class]), num_classes=2).unsqueeze(0).to(self.device)
        else:
            target = one_hot(torch.tensor([target_class]), num_classes=1).unsqueeze(0).to(self.device)
        # print("target_class", target_class, "target", target)
        classifier.zero_grad()
        logits.backward(target)
        
        iput_embeds_grad = inputs_embeds.grad.detach().cpu().numpy()[0] #(1, seq_len, dim)
        all_token_hidden_states_grad = all_token_hidden_states.grad.detach().cpu().numpy()[0] #(1, seq_len, dim)
        return iput_embeds_grad, all_token_hidden_states_grad

    def get_smmoth_grad(self, layer, inputs_embeds, hidden_state_type, hidden_state_dims=None, target_class=0,
                        n_turn=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(inputs_embeds) - torch.min(inputs_embeds)).detach().cpu().numpy()

        grad_sum = np.zeros(inputs_embeds[0].shape)
        for i in range(n_turn):
            with torch.no_grad():
                noise = torch.empty(inputs_embeds.size()).normal_(0, std).to(inputs_embeds.device)
                noise_inputs_embeds = inputs_embeds + noise

            noise_inputs_embeds.requires_grad = True
            noise_inputs_embeds.retain_grad()
            grad, _ = self.get_grad([layer], noise_inputs_embeds, hidden_state_type, hidden_state_dims, target_class)
            grad_sum += process(grad)
        return grad_sum / n_turn # (seq_len, dim)
        

def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels]

def get_layer_id(hidden_state_dims):
    layer_ids, layer_dims = [], []
    for idx in hidden_state_dims:
        layer_ids.append(idx // 4096 + 1) #!!!!!!!
        layer_dims.append(idx % 4096)
    return layer_ids, layer_dims


