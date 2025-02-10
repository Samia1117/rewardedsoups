from transformers import AutoModel, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

model_name = "carolinezhang/gpt2-imdb-pos"

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(model.state_dict().keys())
# for layer_name in model.state_dict().keys(): 
#     weights = model.state_dict()[layer_name].detach().numpy()
#     print(layer_name)
#     print(weights)

first_key = list(model.state_dict().keys())[0]
test_avg = 0.5*(model.state_dict()[first_key].detach().numpy()) + 0.5*(model.state_dict()[first_key].detach().numpy())
print(test_avg)

model.state_dict()[first_key] = "new value"
print(model.state_dict()[first_key])

prompt = "This movie was"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
