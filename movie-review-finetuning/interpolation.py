from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from collections import OrderedDict
import torch


def get_max_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor where each element is the max of the corresponding elements 
    from the two input tensors.

    Args:
        tensor1 (torch.Tensor): First 2D tensor.
        tensor2 (torch.Tensor): Second 2D tensor.

    Returns:
        torch.Tensor: A new tensor with the maximum values.
    """
    return torch.max(tensor1, tensor2)

def get_min_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor where each element is the max of the corresponding elements 
    from the two input tensors.

    Args:
        tensor1 (torch.Tensor): First 2D tensor.
        tensor2 (torch.Tensor): Second 2D tensor.

    Returns:
        torch.Tensor: A new tensor with the maximum values.
    """
    return torch.min(tensor1, tensor2)

# model1 
pos_model_name = "carolinezhang/gpt2-imdb-pos"
pos_model = AutoModelForCausalLMWithValueHead.from_pretrained(pos_model_name)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name)

# model 2
neg_model_name = "Samzy17/gpt2-imdb-movie-reviews-negative"
neg_model = AutoModelForCausalLMWithValueHead.from_pretrained(neg_model_name)
neg_tokenizer = AutoTokenizer.from_pretrained(neg_model_name)

print(f'Pos state dict size = {len(pos_model.state_dict().items())}, Neg state dict size = {len(neg_model.state_dict().items())}')

# Model whose state dictionary to update
base_model_name = "lvwerra/gpt2-imdb"
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model_to_save_name = "gpt2-imdb-min-weights-model"
print(f"Interpolating the weights of model1={pos_model_name} and model2={neg_model_name} using a Max() function")

# State dictionary to contain the interpolated weights
base_model_sd = OrderedDict()

# Populate the state dictionary to be loaded onto the base model
for k,v in neg_model.state_dict().items():
    val1 = pos_model.state_dict()[k]
    val2 = neg_model.state_dict()[k]
    
    # Weights that should NOT have the 'pretrained' prefix included
    no_pretrained = ['v_head.summary.weight', 'v_head.summary.bias']
    if k in no_pretrained:
        # NON-LINEAR OPERATION: get MIN/MAX of the two weights
        base_model_sd[k] = get_min_tensor(val1, val2)
    else:
        # Weights that should have the 'pretrained' prefix added
        base_model_sd[f'pretrained_model.{k}'] = get_min_tensor(val1, val2)

# LOAD newly populated state dict into the base model 
base_model.load_state_dict(base_model_sd)  # should print "<All keys matched successfully>"

# SAVE model with new state dict locally (upload to hf if desired)
base_model.save_pretrained(model_to_save_name)
# SAVE tokenizer of the base model
base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model_tokenizer.save_pretrained(model_to_save_name)
