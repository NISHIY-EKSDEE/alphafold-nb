from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
text = "Hello, my dog is cute"
encoded_input = tokenizer(text, return_tensors="np")
output = model(**encoded_input)
print(output)