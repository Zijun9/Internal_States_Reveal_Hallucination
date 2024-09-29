import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms.functional import resize
from transformers import LlamaModel, LlamaConfig, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import random
random.seed(42)

# https://www.learnpytorch.io/02_pytorch_classification/

class postiveLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=True, device=device, dtype=dtype)

    def forward(self, input):
        weight = F.relu(self.weight)  # Apply ReLU to the weights
        # print(weight.shape, weight)
        return F.linear(input, weight, self.bias)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.score(x)
        return x

class LinearClassifier1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.score(x)
        return x
    
class LinearClassifier12(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.score(x)
        return x

# https://medium.com/@hkabhi916/understanding-lstm-for-sequence-classification-a-practical-guide-with-pytorch-ac40e84ad3d5
class LSTMClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim, LSTM_num_layers=2, output_size=2):
        super(LSTMClassifier2, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM_num_layers = LSTM_num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, LSTM_num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        if len(x.shape) > 3: # [batch, num_layers, seq_len, hidden_dim]
            x = x.view(x.size(0), -1, x.size(-1))# [batch, num_layers, hidden_dim]
        h0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def get_device(self):
        return next(self.parameters()).device  
    

class postiveLinearClassifier1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = postiveLinear(input_dim, 1)

    def forward(self, x):
        x = self.score(x)
        return x


class LlamaMLPClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.act_fn = nn.SiLU()
        self.score = nn.Linear(self.input_dim, 2)  # binary classification at this moment
        
    def forward(self, x):
        # x.shape # [b, num_layers, hidden_dim] 
        if len(x.shape) >2:
            x = x.view(x.size(0), -1)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = self.score(down_proj)
        return x
    
    def get_device(self):
        return next(self.parameters()).device  
    
    
class ResNetClassifier2(nn.Module):
    # for attention
    def __init__(self, channel=1):
        super(ResNetClassifier2, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def get_device(self):
        return next(self.parameters()).device  
    
    def forward(self, x):
        # print("ResNetClassifier2 input", x.shape)
        # input [b, num_layers/channel=1, seq_len, seq_len] 
        # x = resize(x, [224, 224])
        # print("ResNetClassifier2 after", x.shape)
        return self.resnet(x)
    

# https://stackoverflow.com/questions/54924582/is-it-possible-to-freeze-only-certain-embedding-weights-in-the-embedding-layer-i
class LlamaModel_ST(LlamaModel):
    """
    special token
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # print("vocab_size", self.vocab_size) #128257 trained model already updated
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.trainable_embed_tokens = nn.Embedding(1, config.hidden_size)
        
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

            trainable_input_ids = input_ids.clone()
            assert self.embed_tokens.weight.shape[0] == 128257
            mask = input_ids >= self.embed_tokens.weight.shape[0]-1 #128256
            trainable_input_ids = trainable_input_ids - (self.embed_tokens.weight.shape[0]-1)
            trainable_input_ids[~mask] = 0
            trainable_inputs_embeds = self.trainable_embed_tokens(trainable_input_ids)

            inputs_embeds[mask] = trainable_inputs_embeds[mask]

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForSequenceClassification_ST(LlamaPreTrainedModel):
    def __init__(self, config, layers_to_process, classifier_type):
        super().__init__(config)
        self.num_labels = config.num_labels
        assert self.num_labels == 2, "Only binary classification is supported"
        self.model = LlamaModel_ST(config)
        self.layers_to_process = layers_to_process

        self.classifier_type = classifier_type
        if classifier_type == "linear":
            self.score = nn.Linear(len(layers_to_process)*config.hidden_size, self.num_labels, bias=False)
        elif classifier_type == "LlamaMLP2":
            self.score = LlamaMLPClassifier2(len(layers_to_process)*config.hidden_size, hidden_dim=11008)
        elif classifier_type == "LSTM2":
            self.score = LSTMClassifier2(config.hidden_size, hidden_dim=11008)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = transformer_outputs["hidden_states"]
        layers_to_process_hidden_states = []
        for layer in self.layers_to_process:
            h =  hidden_states[layer][input_ids == 128256] # [batch, 4096]
            if h.shape[0] != hidden_states[layer].shape[0] or h.shape[-1] != hidden_states[layer].shape[-1]:
                print("layer", layer)
                print("input_ids", input_ids)
                print("input_ids == 128256", input_ids == 128256)
                print("hidden_states[layer].shape", hidden_states[layer].shape)
                print("hidden_states[layer]", hidden_states[layer])
                print("h.shape", h.shape)
            assert h.shape[0] == hidden_states[layer].shape[0] # batch
            assert h.shape[-1] == hidden_states[layer].shape[-1] # 4096
            layers_to_process_hidden_states.append(h) # [layer, [batch, 4096]]

        if self.classifier_type in ["LlamaMLP2", "linear"]:
             # [batch, 4096 * len(layers_to_process)]
            layers_to_process_hidden_states = torch.cat(layers_to_process_hidden_states, dim=-1) 
        elif self.classifier_type == "LSTM2":
            # [batch, num_layers, hidden_dim]
            layers_to_process_hidden_states = torch.stack(layers_to_process_hidden_states, dim=1)
        logits = self.score(layers_to_process_hidden_states)

        # if input_ids is not None:
        #     batch_size = input_ids.shape[0]
        # else:
        #     batch_size = inputs_embeds.shape[0]

        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        #         sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        #         sequence_lengths = sequence_lengths % input_ids.shape[-1]
        #         sequence_lengths = sequence_lengths.to(logits.device)
        #     else:
        #         sequence_lengths = -1

        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        pooled_logits = logits

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                assert False, "Not regression"
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                assert False, "Not multi_label_classification"
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

