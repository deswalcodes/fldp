from transformers import AutoTokenizer, AutoModel
import torch

class BertEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  

    def embed_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
       
        cls_embedding = outputs.last_hidden_state[:,0,:]
        return cls_embedding.squeeze().numpy()
