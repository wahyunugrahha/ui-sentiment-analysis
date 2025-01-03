import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😢"}

@st.cache_resource
def load_model(learning_rate, epoch):
    model_filename = f'model/model_{learning_rate}E{epoch}.pth'
    model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    model.load_state_dict(torch.load(model_filename, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

class SentimentModel:
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.model = load_model(self.learning_rate, self.epoch)
        self.tokenizer = load_tokenizer()

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence, prediction = torch.max(probabilities, dim=-1)
        return prediction.item(), confidence.item()
