from transformers import BertTokenizer, FlaxBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
outputs = model(**inputs)
print(outputs)
print(outputs.__dict__)