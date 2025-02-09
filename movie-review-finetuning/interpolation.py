from transformers import AutoModel, AutoTokenizer

model_name = "carolinezhang/gpt2-imdb-pos-v2/gpt2-imdb-pos-v2"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "This movie was"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
