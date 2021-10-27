from transformers import BertTokenizer, FlaxBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
outputs = model(**inputs)
print("Prediction logits: {}".format(outputs.prediction_logits))
print("Seq relationship logits: {}".format(outputs.seq_relationship_logits))