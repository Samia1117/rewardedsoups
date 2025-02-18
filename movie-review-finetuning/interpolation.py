from transformers import AutoModel, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from collections import OrderedDict

# model1 
pos_model_name = "carolinezhang/gpt2-imdb-pos"
pos_model = AutoModelForCausalLMWithValueHead.from_pretrained(pos_model_name)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name)

# model 2
neg_model_name = "Samzy17/gpt2-imdb-movie-reviews-negative"
neg_model = AutoModelForCausalLMWithValueHead.from_pretrained(neg_model_name)
neg_tokenizer = AutoTokenizer.from_pretrained(neg_model_name)

# print(f'Pos state dict size = {len(pos_model.state_dict().items())}, 
#       Neg state dict size = {len(neg_model.state_dict().items())}')

# Model whose state dictionary to update
base_model_name = "lvwerra/gpt2-imdb"
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# model_to_save_name = 'gpt2-imdb-pos-neg-interpolated-'
model_to_save_name = 'gpt2-imdb-pos-neg-interpolated-fixed'
# State dictionary to contain the average weights
base_model_sd = OrderedDict()

lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for l in lambdas: 
    # i = 0
    # Populate the state dictionary to be loaded onto the base model
    for k,v in neg_model.state_dict().items():
        val1 = pos_model.state_dict()[k]
        val2 = neg_model.state_dict()[k]
        
        # Weights that should NOT have the 'pretrained' prefix included
        no_pretrained = ['v_head.summary.weight', 'v_head.summary.bias']
        if k in no_pretrained:
            # average weights!
            base_model_sd[k] = (l * val1) + ((1-l) * val2)
        else:
            # Weights that should have the 'pretrained' prefix added
            base_model_sd[f'pretrained_model.{k}'] = (l * val1) + ((1-l) * val2)
        # i +=1

    # LOAD newly populated state dict into the base model 
    base_model.load_state_dict(base_model_sd)  # should print "<All keys matched successfully>"

    # SAVE model with new state dict locally (upload to hf if desired)
    base_model.save_pretrained(model_to_save_name + str(l))
    # SAVE tokenizer of the base model
    base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model_tokenizer.save_pretrained(model_to_save_name + str(l))
