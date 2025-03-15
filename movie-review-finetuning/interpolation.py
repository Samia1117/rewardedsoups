from transformers import AutoModel, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from collections import OrderedDict
import torch

# model1 
pos_model_name = "carolinezhang/gpt2-imdb-pos"
pos_model = AutoModelForCausalLMWithValueHead.from_pretrained(pos_model_name)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name)

# model 2
concise_model_name = "/home/users/sz159/2024-2025/samia1117-github/rewardedsoups/movie-review-finetuning/gpt2-imdb-concise-reviews-03-08"
concise_model = AutoModelForCausalLMWithValueHead.from_pretrained(concise_model_name)
concise_tokenizer = AutoTokenizer.from_pretrained(concise_model_name)

print(f'Pos statedict size = {len(pos_model.state_dict().items())}, Concise statedict size = {len(concise_model.state_dict().items())}')

# Model whose state dictionary to update
base_model_name = "lvwerra/gpt2-imdb"
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model_to_save_name_prefix = "gpt2-imdb-pos-concise-03-08-"
print(f"Interpolating the weights of model1={pos_model_name} and model2={concise_model_name} ... ")

# State dictionary to contain the interpolated weights
base_model_sd = OrderedDict()

lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for l in lambdas: 
    # Populate the empty state dictionary to be loaded onto the base model
    for k,v in concise_model.state_dict().items():
        val1 = pos_model.state_dict()[k]
        val2 = concise_model.state_dict()[k]

        # Weights that should NOT have the 'pretrained' prefix included
        no_pretrained = ['v_head.summary.weight', 'v_head.summary.bias']
        if k in no_pretrained:
            base_model_sd[k] = (l * val1) + ((1-l) * val2)
        else:
            # Weights that should have the 'pretrained' prefix added
            base_model_sd[f'pretrained_model.{k}'] = (l * val1) + ((1-l) * val2)

    # LOAD newly populated state dict into the base model 
    base_model.load_state_dict(base_model_sd)  # should print "<All keys matched successfully>"

    # SAVE model with new state dict locally (upload to hf if desired)
    base_model.save_pretrained(model_to_save_name_prefix + str(l))
    # SAVE tokenizer of the base model
    base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model_tokenizer.save_pretrained(model_to_save_name_prefix + str(l))

print("###### Done saving all interpolated models!")