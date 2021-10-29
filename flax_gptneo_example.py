from transformers import GPT2Tokenizer, FlaxGPTNeoModel

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')

inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print("Outputs: {}".format(outputs))
print("Outputs dict: {}".format(outputs.__dict__))
print("Last hidden states: {}".format(last_hidden_states))