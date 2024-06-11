import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        return prediction.item()
