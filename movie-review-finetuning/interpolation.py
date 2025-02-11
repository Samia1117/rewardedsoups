from transformers import AutoModel, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

pos_model_name = "carolinezhang/gpt2-imdb-pos"
neg_model_name = "Samzy17/gpt2-imdb-movie-reviews-negative"

# print(model.state_dict().keys())
# for layer_name in model.state_dict().keys(): 
#     weights = model.state_dict()[layer_name].detach().numpy()
#     print(layer_name)
#     print(weights)

pos_model = AutoModelForCausalLMWithValueHead.from_pretrained(pos_model_name)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name)

neg_model = AutoModelForCausalLMWithValueHead.from_pretrained(neg_model_name)
neg_tokenizer = AutoTokenizer.from_pretrained(neg_model_name)

pos_state_dict = pos_model.state_dict()
neg_state_dict = neg_model.state_dict()

avg_state_dict = {}

for key in pos_state_dict.keys():
    pos_weights = pos_state_dict[key]  
    neg_weights = neg_state_dict[key]  

    avg_weights = 0.5*pos_weights + 0.5*neg_weights

    key = "pretrained_model."+key

    avg_state_dict[key] = avg_weights

#using the pos model, either should work bc they have the same architecture
avg_model = AutoModelForCausalLMWithValueHead.from_pretrained(pos_model_name)

avg_model.load_state_dict(avg_state_dict)

prompt = "This movie was"
inputs = pos_tokenizer(prompt, return_tensors="pt")
outputs = avg_model.generate(**inputs, max_length=50)
print(pos_tokenizer.decode(outputs[0]))
