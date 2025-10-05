"""
Knowledge Graph Embedding Models: TransE and RotatE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class TransE(nn.Module):
    """TransE: Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Normalize embeddings
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor):
        """
        Forward pass for scoring triples
        Args:
            heads: (batch_size,) entity IDs
            relations: (batch_size,) relation IDs  
            tails: (batch_size,) entity IDs
        Returns:
            scores: (batch_size,) scores for each triple
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # TransE scoring: ||h + r - t||
        score = torch.norm(h + r - t, p=2, dim=1)
        return score
    
    def score_triple(self, head: int, relation: int, tail: int):
        """Score a single triple"""
        heads = torch.LongTensor([head])
        relations = torch.LongTensor([relation])
        tails = torch.LongTensor([tail])
        
        with torch.no_grad():
            score = self.forward(heads, relations, tails)
        return score.item()
    
    def get_entity_embedding(self, entity_id: int):
        """Get embedding for a single entity"""
        with torch.no_grad():
            return self.entity_embeddings.weight[entity_id].cpu().numpy()
    
    def get_relation_embedding(self, relation_id: int):
        """Get embedding for a single relation"""
        with torch.no_grad():
            return self.relation_embeddings.weight[relation_id].cpu().numpy()
    
    def predict_tail(self, head: int, relation: int, k: int = 10):
        """Predict top-k tail entities for given head and relation"""
        h = self.entity_embeddings.weight[head].unsqueeze(0)
        r = self.relation_embeddings.weight[relation].unsqueeze(0)
        
        # Compute scores for all entities as tails
        all_tails = self.entity_embeddings.weight
        scores = torch.norm(h + r - all_tails, p=2, dim=1)
        
        # Get top-k with lowest scores (lower is better for TransE)
        top_k_scores, top_k_indices = torch.topk(scores, k, largest=False)
        
        return top_k_indices.cpu().numpy(), top_k_scores.detach().cpu().numpy()


class RotatE(nn.Module):
    """RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 9.0):
        super(RotatE, self).__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even for RotatE"
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Entity embeddings are complex numbers (real + imaginary parts)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        
        # Relation embeddings are phases (angles in complex plane)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Constrain relation embeddings to [0, 2π)
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data / (
            2 * np.pi / (embedding_dim))
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor):
        """
        Forward pass using complex number rotation
        """
        # Get embeddings
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # Split into real and imaginary parts
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        
        # Relation as rotation in complex plane
        # r represents phase: e^{i*phase}
        r_re = torch.cos(r)
        r_im = torch.sin(r)
        
        # Complex multiplication: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        
        # Distance: ||h * r - t||
        score = torch.sqrt(
            (hr_re - t_re) ** 2 + (hr_im - t_im) ** 2
        ).sum(dim=-1)
        
        return score
    
    def score_triple(self, head: int, relation: int, tail: int):
        """Score a single triple"""
        heads = torch.LongTensor([head])
        relations = torch.LongTensor([relation])
        tails = torch.LongTensor([tail])
        
        with torch.no_grad():
            score = self.forward(heads, relations, tails)
        return score.item()
    
    def get_entity_embedding(self, entity_id: int):
        """Get embedding for a single entity"""
        with torch.no_grad():
            return self.entity_embeddings.weight[entity_id].cpu().numpy()
    
    def get_relation_embedding(self, relation_id: int):
        """Get embedding for a single relation"""
        with torch.no_grad():
            return self.relation_embeddings.weight[relation_id].cpu().numpy()


class KGEmbeddingModel:
    """Wrapper class for KG embedding models with training utilities"""
    
    def __init__(self, model_type: str = 'TransE', **kwargs):
        self.model_type = model_type
        
        if model_type == 'TransE':
            self.model = TransE(**kwargs)
        elif model_type == 'RotatE':
            self.model = RotatE(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def setup_optimizer(self, learning_rate: float = 0.001):
        """Initialize optimizer"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_entities': self.model.num_entities,
            'num_relations': self.model.num_relations,
            'embedding_dim': self.model.embedding_dim,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """Get entity embedding as numpy array"""
        return self.model.get_entity_embedding(entity_id)
    
    def get_all_entity_embeddings(self) -> np.ndarray:
        """Get all entity embeddings"""
        with torch.no_grad():
            return self.model.entity_embeddings.weight.cpu().numpy()
    
    def score_triples(self, triples: List[Tuple[int, int, int]]) -> np.ndarray:
        """Score multiple triples"""
        heads = torch.LongTensor([t[0] for t in triples]).to(self.device)
        relations = torch.LongTensor([t[1] for t in triples]).to(self.device)
        tails = torch.LongTensor([t[2] for t in triples]).to(self.device)
        
        with torch.no_grad():
            scores = self.model(heads, relations, tails)
        
        return scores.cpu().numpy()
    
    def predict_tail(self, head: int, relation: int, k: int = 10):
        """Predict top-k tail entities"""
        return self.model.predict_tail(head, relation, k)
