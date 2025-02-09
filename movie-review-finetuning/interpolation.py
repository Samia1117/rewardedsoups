from transformers import AutoModel, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

model_name = "carolinezhang/gpt2-imdb-pos"

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model.state_dict().keys())
for layer_name in model.state_dict().keys(): 
    weights = model.state_dict()[layer_name].detach().numpy()
    print(layer_name)
    print(weights)

prompt = "This movie was"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
