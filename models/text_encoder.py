"""
Med-Nexus Clinical Text Encoder
Encodes clinical reports and retrieved RAG evidence using a robust ClinicalBERT backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    # Fallback if transformers isn't installed
    AutoTokenizer = None
    AutoModel = None

class ClinicalTextEncoder(nn.Module):
    """
    Encodes clinical text using Bio_ClinicalBERT.
    Project 2: RAG text feature extraction.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        if AutoTokenizer is not None and AutoModel is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name)
                bert_hidden_size = self.bert.config.hidden_size
            except Exception as e:
                print(f"[Warning] Could not load HuggingFace model '{model_name}': {e}. Using fallback Embedding bag.")
                self.tokenizer = None
                self.bert = None
                bert_hidden_size = 768
                self.fallback_embed = nn.EmbeddingBag(vocab_size, bert_hidden_size, sparse=False)
        else:
            self.tokenizer = None
            self.bert = None
            bert_hidden_size = 768
            self.fallback_embed = nn.EmbeddingBag(vocab_size, bert_hidden_size, sparse=False)
            
        self.fc_proj = nn.Linear(bert_hidden_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for raw token tensors.
        """
        if self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooled_output = self.fallback_embed(input_ids)
            
        proj = self.fc_proj(pooled_output)
        return F.normalize(proj, p=2, dim=-1)
        
    def encode_text(self, text_list: List[str], device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Encodes a list of strings into normalized embedding vectors.
        """
        if self.tokenizer is not None and self.bert is not None:
            encoded_input = self.tokenizer(
                text_list, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            self.bert.to(device)
            self.fc_proj.to(device)
            
            return self.forward(input_ids, attention_mask)
        else:
            # Fallback simple hashing tokenizer
            self.fallback_embed.to(device)
            self.fc_proj.to(device)
            
            B = len(text_list)
            input_ids = torch.zeros((B, 10), dtype=torch.long, device=device)
            for i, text in enumerate(text_list):
                words = text.split()[:10]
                for j, word in enumerate(words):
                    input_ids[i, j] = hash(word) % 10000
                    
            return self.forward(input_ids)
