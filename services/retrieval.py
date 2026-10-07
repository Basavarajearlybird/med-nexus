"""
Med-Nexus RAG Evidence Retrieval Service
Performs semantic vector similarity search over clinical reports using embeddings
and inserts distractor passages for retrieval noise benchmarking.
"""

import random
from typing import List
import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class EvidenceRetrievalService:
    """Retrieves clinical evidence reports using semantic vector similarity."""

    KNOWLEDGE_BASE = [
        "Lungs are clear bilaterally without focal consolidation, pneumothorax, or pleural effusion.",
        "Cardiomediastinal silhouette is within normal limits. No acute cardiopulmonary process.",
        "Normal chest radiographic examination. Lungs are well aerated with clear costophrenic angles.",
        "Focal opacity in the lower lobe concerning for acute lobar pneumonia.",
        "Bilateral patchy infiltrates with diffuse alveolar opacities and prominent interstitial markings.",
        "Moderate pleural effusion with associated retrocardiac atelectasis and consolidative changes.",
        "Patient underwent previous cholecystectomy. No abdominal acute process.",
        "Degenerative joint disease of the thoracic spine noted with marginal osteophytes.",
        "Scalene muscle hypertrophy without cervical rib anomaly.",
        "Normal study of right upper extremity radiograph.",
        "Slight tortuosity of the thoracic aorta without acute dissection.",
        "Calcified granuloma in the right upper lobe, chronically stable."
    ]

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.model = None
        
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                # Precompute embeddings for the knowledge base
                self.kb_embeddings = self._encode(self.KNOWLEDGE_BASE)
            except Exception as e:
                print(f"[Warning] Retrieval service fallback due to {e}")
                self.kb_embeddings = None
        else:
            self.kb_embeddings = None

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """Encodes text to semantic embeddings."""
        if self.tokenizer is None or self.model is None:
            # Deterministic lexical fallback for smoke tests only.
            vec=torch.zeros((len(texts), 256), device=self.device)
            for i,t in enumerate(texts):
                for tok in t.lower().split():
                    vec[i, hash(tok) % 256] += 1.0
            return F.normalize(vec, p=2, dim=-1)
            
        with torch.no_grad():
            encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :]
            return F.normalize(embeddings, p=2, dim=-1)

    def retrieve_evidence(self, query: str, top_k: int = 3, k_distractors: int = 0) -> List[str]:
        """
        Retrieves top_k evidence passages via semantic similarity and optionally appends k_distractors.
        """
        if self.kb_embeddings is not None:
            # Semantic search
            query_emb = self._encode([query])
            cos_scores = torch.matmul(query_emb, self.kb_embeddings.transpose(0, 1)).squeeze(0)
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.KNOWLEDGE_BASE)))
            indices = top_results.indices.cpu().numpy()
            retrieved = [self.KNOWLEDGE_BASE[idx] for idx in indices]
        else:
            # Fallback simple keyword search
            words = set(query.lower().split())
            scored = []
            for passage in self.KNOWLEDGE_BASE:
                p_words = set(passage.lower().split())
                score = len(words.intersection(p_words)) / float(max(1, len(words)))
                scored.append((score, passage))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved = [item[1] for item in scored[:top_k]]

        if k_distractors > 0:
            distractors = [p for p in self.KNOWLEDGE_BASE if p not in retrieved]
            if not distractors: return retrieved
            selected_distractors = random.sample(distractors, k=min(k_distractors, len(distractors)))
            retrieved.extend(selected_distractors)

        return retrieved
