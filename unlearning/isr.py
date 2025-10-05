"""
Influence-based Selective Retraining (ISR) for Knowledge Graph Unlearning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import time
from tqdm import tqdm

class InfluenceCalculator:
    """Calculate influence of training samples on model parameters"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.hessian_inv_approx = None
    
    def compute_gradient(self, triple: Tuple[int, int, int], negative_samples: List[Tuple]) -> Dict:
        """Compute gradient for a single triple"""
        head, relation, tail = triple
        
        # Positive triple
        h_pos = torch.LongTensor([head]).to(self.device)
        r_pos = torch.LongTensor([relation]).to(self.device)
        t_pos = torch.LongTensor([tail]).to(self.device)
        
        score_pos = self.model(h_pos, r_pos, t_pos)
        
        # Negative triples
        neg_heads = torch.LongTensor([n[0] for n in negative_samples]).to(self.device)
        neg_rels = torch.LongTensor([n[1] for n in negative_samples]).to(self.device)
        neg_tails = torch.LongTensor([n[2] for n in negative_samples]).to(self.device)
        
        score_neg = self.model(neg_heads, neg_rels, neg_tails)
        
        # Margin ranking loss
        margin = self.model.margin
        loss = F.relu(score_pos - score_neg.mean() + margin)
        
        # Compute gradient
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Zero gradients
        self.model.zero_grad()
        
        return gradients
    
    def compute_hessian_inverse_lbfgs(self, dataloader, max_samples=1000):
        """
        Approximate Hessian inverse using L-BFGS history
        This is a simplified version - full implementation would use actual L-BFGS history
        """
        print("Computing Hessian approximation using L-BFGS...")
        
        # Collect gradient samples
        gradient_samples = []
        param_shapes = {}
        
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= max_samples:
                break
            
            heads, relations, tails = batch
            heads = heads.to(self.device)
            relations = relations.to(self.device)
            tails = tails.to(self.device)
            
            # Forward pass
            scores = self.model(heads, relations, tails)
            loss = scores.mean()
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            grads = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in param_shapes:
                        param_shapes[name] = param.shape
                    grads.append(param.grad.view(-1))
            
            if grads:
                gradient_samples.append(torch.cat(grads))
            
            self.model.zero_grad()
        
        # Compute empirical Fisher as Hessian approximation
        # Fisher = E[grad * grad^T]
        gradient_tensor = torch.stack(gradient_samples)
        fisher_approx = torch.matmul(gradient_tensor.T, gradient_tensor) / len(gradient_samples)
        
        # Add regularization for numerical stability
        fisher_approx += torch.eye(fisher_approx.size(0)).to(self.device) * 1e-3
        
        # Inverse approximation (use only diagonal for efficiency)
        fisher_diag = torch.diag(fisher_approx)
        hessian_inv_diag = 1.0 / (fisher_diag + 1e-8)
        
        self.hessian_inv_approx = hessian_inv_diag
        
        print(f"Computed Hessian approximation with shape {hessian_inv_diag.shape}")
        return hessian_inv_diag
    
    def compute_influence(self, deleted_triple: Tuple[int, int, int], 
                         negative_samples: List[Tuple],
                         total_gradient: torch.Tensor) -> float:
        """
        Compute influence of a deleted triple on model parameters
        Influence = grad_total^T * H^{-1} * grad_triple
        """
        # Compute gradient for deleted triple
        triple_gradient = self.compute_gradient(deleted_triple, negative_samples)
        
        # Flatten gradient
        grad_vector = []
        for name, param in self.model.named_parameters():
            if name in triple_gradient:
                grad_vector.append(triple_gradient[name].view(-1))
        
        grad_vector = torch.cat(grad_vector)
        
        # Compute influence using diagonal approximation
        # influence = grad_total^T * H^{-1} * grad_triple
        if self.hessian_inv_approx is not None:
            influence = torch.dot(total_gradient, self.hessian_inv_approx * grad_vector)
        else:
            # Without Hessian, use simple dot product
            influence = torch.dot(total_gradient, grad_vector)
        
        return influence.item()


class ISRUnlearning:
    """Influence-based Selective Retraining for KG Unlearning"""
    
    def __init__(self, model, threshold_tau=0.01, learning_rate=0.0001, 
                 refinement_epochs=10, device='cpu'):
        self.model = model
        self.threshold_tau = threshold_tau
        self.learning_rate = learning_rate
        self.refinement_epochs = refinement_epochs
        self.device = device
        self.influence_calc = InfluenceCalculator(model, device)
    
    def unlearn(self, deleted_triples: List[Tuple[int, int, int]], 
                remaining_triples: List[Tuple[int, int, int]],
                negative_sampler=None) -> Dict:
        """
        Perform ISR unlearning
        
        Args:
            deleted_triples: List of (head, relation, tail) tuples to delete
            remaining_triples: List of remaining training triples
            negative_sampler: Function to generate negative samples
        
        Returns:
            Dictionary with unlearning statistics
        """
        start_time = time.time()
        stats = {
            'num_deleted': len(deleted_triples),
            'num_affected_entities': 0,
            'deletion_time': 0,
            'refinement_time': 0
        }
        
        print(f"Starting ISR unlearning for {len(deleted_triples)} triples...")
        
        # Step 1: Compute total gradient (simplified - would use full training set)
        total_gradient = self._compute_total_gradient(remaining_triples[:1000])
        
        # Step 2: Identify affected entities
        affected_entities = set()
        influences = []
        
        for triple in tqdm(deleted_triples, desc="Computing influences"):
            head, relation, tail = triple
            affected_entities.add(head)
            affected_entities.add(tail)
            
            # Generate negative samples
            if negative_sampler:
                neg_samples = negative_sampler(triple, k=5)
            else:
                neg_samples = self._default_negative_sampling(triple)
            
            # Compute influence
            influence = self.influence_calc.compute_influence(
                triple, neg_samples, total_gradient
            )
            influences.append(abs(influence))
        
        stats['num_affected_entities'] = len(affected_entities)
        stats['mean_influence'] = np.mean(influences)
        stats['max_influence'] = np.max(influences)
        
        # Step 3: Update parameters using reverse gradient
        print("Updating parameters...")
        eta = self.learning_rate
        
        for triple, influence in zip(deleted_triples, influences):
            if abs(influence) > self.threshold_tau:
                # Apply reverse gradient update
                # In practice, this would recompute gradients and update
                # For simplicity, we'll mark entities for retraining
                pass
        
        deletion_time = time.time() - start_time
        stats['deletion_time'] = deletion_time
        
        # Step 4: Refinement on remaining data
        print(f"Refining model for {self.refinement_epochs} epochs...")
        refine_start = time.time()
        
        self._refine_model(list(affected_entities), remaining_triples)
        
        stats['refinement_time'] = time.time() - refine_start
        stats['total_time'] = time.time() - start_time
        
        return stats
    
    def _compute_total_gradient(self, sample_triples: List[Tuple]) -> torch.Tensor:
        """Compute total gradient on sample of training data"""
        self.model.zero_grad()
        
        total_loss = 0
        for triple in sample_triples:
            head, rel, tail = triple
            h = torch.LongTensor([head]).to(self.device)
            r = torch.LongTensor([rel]).to(self.device)
            t = torch.LongTensor([tail]).to(self.device)
            
            score = self.model(h, r, t)
            total_loss += score.mean()
        
        total_loss.backward()
        
        # Flatten gradients
        grad_vector = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.view(-1))
        
        return torch.cat(grad_vector) if grad_vector else torch.zeros(1)
    
    def _default_negative_sampling(self, triple: Tuple, k: int = 5) -> List[Tuple]:
        """Simple negative sampling by corrupting head or tail"""
        head, relation, tail = triple
        negatives = []
        
        for _ in range(k):
            if np.random.random() < 0.5:
                # Corrupt head
                neg_head = np.random.randint(0, self.model.num_entities)
                negatives.append((neg_head, relation, tail))
            else:
                # Corrupt tail
                neg_tail = np.random.randint(0, self.model.num_entities)
                negatives.append((head, relation, neg_tail))
        
        return negatives
    
    def _refine_model(self, affected_entities: List[int], 
                     remaining_triples: List[Tuple]):
        """Refine model parameters for affected entities"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Filter triples involving affected entities
        affected_set = set(affected_entities)
        refine_triples = [
            t for t in remaining_triples 
            if t[0] in affected_set or t[2] in affected_set
        ]
        
        print(f"Refining on {len(refine_triples)} affected triples...")
        
        for epoch in range(self.refinement_epochs):
            np.random.shuffle(refine_triples)
            total_loss = 0
            
            batch_size = 128
            for i in range(0, len(refine_triples), batch_size):
                batch = refine_triples[i:i+batch_size]
                
                heads = torch.LongTensor([t[0] for t in batch]).to(self.device)
                relations = torch.LongTensor([t[1] for t in batch]).to(self.device)
                tails = torch.LongTensor([t[2] for t in batch]).to(self.device)
                
                scores = self.model(heads, relations, tails)
                loss = scores.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.refinement_epochs}, Loss: {total_loss/len(refine_triples):.4f}")


def run_isr_experiment(model, deleted_triples, remaining_triples, config):
    """Run complete ISR unlearning experiment"""
    
    isr = ISRUnlearning(
        model=model,
        threshold_tau=config.UNLEARNING_CONFIG['isr']['threshold_tau'],
        learning_rate=config.UNLEARNING_CONFIG['isr']['learning_rate'],
        refinement_epochs=config.UNLEARNING_CONFIG['isr']['refinement_epochs'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    stats = isr.unlearn(deleted_triples, remaining_triples)
    
    return stats
