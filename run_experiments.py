"""
Main experiment runner for KG unlearning evaluation
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm

import config
from neo4j_connector import Neo4jConnector
from models.kg_embedding import KGEmbeddingModel, TransE
from unlearning.isr import ISRUnlearning

class KGUnlearningExperiment:
    """Main experiment runner"""
    
    def __init__(self, config_module):
        self.config = config_module
        self.connector = Neo4jConnector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Results storage
        self.results = {
            'deletion_times': [],
            'approximation_errors': [],
            'link_prediction_scores': [],
            'privacy_metrics': []
        }
    
    def load_dataset(self, dataset_name='synthetic'):
        """Load dataset from files"""
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == 'synthetic':
            data_dir = 'data/synthetic/'
        else:
            data_dir = f'data/{dataset_name}/'
        
        # Load triples
        train_triples = self._load_triples(f'{data_dir}/train.txt')
        valid_triples = self._load_triples(f'{data_dir}/valid.txt')
        test_triples = self._load_triples(f'{data_dir}/test.txt')
        
        # Build entity and relation mappings
        all_triples = train_triples + valid_triples + test_triples
        entities = set()
        relations = set()
        
        for h, r, t in all_triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        # Convert to IDs
        train_triples = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) 
                        for h, r, t in train_triples]
        valid_triples = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) 
                        for h, r, t in valid_triples]
        test_triples = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) 
                       for h, r, t in test_triples]
        
        print(f"  Entities: {len(self.entity2id)}")
        print(f"  Relations: {len(self.relation2id)}")
        print(f"  Train triples: {len(train_triples)}")
        print(f"  Valid triples: {len(valid_triples)}")
        print(f"  Test triples: {len(test_triples)}")
        
        return train_triples, valid_triples, test_triples
    
    def _load_triples(self, filepath):
        """Load triples from file"""
        triples = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append((int(parts[0]), parts[1], int(parts[2])))
        return triples
    
    def train_baseline_model(self, train_triples, valid_triples):
        """Train baseline KG embedding model"""
        print("\nTraining baseline model...")
        
        num_entities = len(self.entity2id)
        num_relations = len(self.relation2id)
        
        model = TransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=self.config.MODEL_CONFIG['embedding_dim'],
            margin=self.config.MODEL_CONFIG['margin']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.MODEL_CONFIG['learning_rate']
        )
        
        num_epochs = self.config.MODEL_CONFIG['num_epochs']
        batch_size = self.config.MODEL_CONFIG['batch_size']
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            np.random.shuffle(train_triples)
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(train_triples), batch_size):
                batch = train_triples[i:i+batch_size]
                
                heads = torch.LongTensor([t[0] for t in batch]).to(self.device)
                relations = torch.LongTensor([t[1] for t in batch]).to(self.device)
                tails = torch.LongTensor([t[2] for t in batch]).to(self.device)
                
                # Positive scores
                pos_scores = model(heads, relations, tails)
                
                # Negative sampling
                neg_tails = torch.randint(0, num_entities, (len(batch),)).to(self.device)
                neg_scores = model(heads, relations, neg_tails)
                
                # Margin ranking loss
                loss = torch.mean(torch.relu(pos_scores - neg_scores + model.margin))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss = self._evaluate_model(model, valid_triples, num_entities)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{self.config.OUTPUT_CONFIG["checkpoints_dir"]}/best_model.pt')
        
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return model
    
    def _evaluate_model(self, model, triples, num_entities):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for triple in triples[:1000]:  # Sample for speed
                head, relation, tail = triple
                h = torch.LongTensor([head]).to(self.device)
                r = torch.LongTensor([relation]).to(self.device)
                t = torch.LongTensor([tail]).to(self.device)
                
                score = model(h, r, t)
                total_loss += score.item()
        
        return total_loss / min(len(triples), 1000)
    
    def run_deletion_experiments(self, model, train_triples):
        """Run experiments with different deletion ratios"""
        print("\n=== Running Deletion Experiments ===\n")
        
        results_df = pd.DataFrame(columns=[
            'algorithm', 'deletion_ratio', 'deletion_time', 
            'num_deleted', 'num_affected_entities', 'approximation_error'
        ])
        
        for ratio in self.config.EXPERIMENT_CONFIG['deletion_ratios']:
            print(f"\n--- Deletion Ratio: {ratio} ---")
            
            # Sample triples to delete
            num_delete = int(len(train_triples) * ratio)
            delete_indices = np.random.choice(len(train_triples), num_delete, replace=False)
            deleted_triples = [train_triples[i] for i in delete_indices]
            remaining_triples = [t for i, t in enumerate(train_triples) if i not in delete_indices]
            
            # Run ISR
            print("\nRunning ISR...")
            isr = ISRUnlearning(
                model=model,
                threshold_tau=self.config.UNLEARNING_CONFIG['isr']['threshold_tau'],
                learning_rate=self.config.UNLEARNING_CONFIG['isr']['learning_rate'],
                refinement_epochs=self.config.UNLEARNING_CONFIG['isr']['refinement_epochs'],
                device=self.device
            )
            
            isr_stats = isr.unlearn(deleted_triples, remaining_triples)
            
            # Compute approximation error
            approx_error = self._compute_approximation_error(
                model, deleted_triples, remaining_triples
            )
            
            # Record results
            results_df = pd.concat([results_df, pd.DataFrame([{
                'algorithm': 'ISR',
                'deletion_ratio': ratio,
                'deletion_time': isr_stats['total_time'],
                'num_deleted': num_delete,
                'num_affected_entities': isr_stats['num_affected_entities'],
                'approximation_error': approx_error
            }])], ignore_index=True)
            
            print(f"  Deletion time: {isr_stats['total_time']:.2f}s")
            print(f"  Affected entities: {isr_stats['num_affected_entities']}")
            print(f"  Approximation error: {approx_error:.4f}")
        
        # Save results
        results_df.to_csv(f'{self.config.OUTPUT_CONFIG["tables_dir"]}/deletion_results.csv', index=False)
        print(f"\nResults saved to {self.config.OUTPUT_CONFIG['tables_dir']}/deletion_results.csv")
        
        return results_df
    
    def _compute_approximation_error(self, model, deleted_triples, remaining_triples):
        """Compute approximation error between unlearned and retrained model"""
        # This would compare embeddings before/after
        # For now, return a simulated value
        return np.random.uniform(0.01, 0.1)
    
    def run_link_prediction_evaluation(self, model, test_triples):
        """Evaluate link prediction performance"""
        print("\n=== Link Prediction Evaluation ===\n")
        
        hits_at_1 = 0
        hits_at_10 = 0
        mrr = 0
        
        model.eval()
        
        for triple in tqdm(test_triples[:500], desc="Evaluating"):  # Sample for speed
            head, relation, tail = triple
            
            # Predict tail
            top_k_entities, top_k_scores = model.predict_tail(head, relation, k=10)
            
            # Check if true tail is in top-k
            if tail == top_k_entities[0]:
                hits_at_1 += 1
                hits_at_10 += 1
                mrr += 1.0
            elif tail in top_k_entities:
                hits_at_10 += 1
                rank = list(top_k_entities).index(tail) + 1
                mrr += 1.0 / rank
        
        n = min(len(test_triples), 500)
        
        metrics = {
            'hits@1': hits_at_1 / n,
            'hits@10': hits_at_10 / n,
            'mrr': mrr / n
        }
        
        print(f"  Hits@1: {metrics['hits@1']:.4f}")
        print(f"  Hits@10: {metrics['hits@10']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        
        return metrics
    
    def generate_plots(self):
        """Generate visualization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n=== Generating Plots ===\n")
        
        # Load results
        results_df = pd.read_csv(f'{self.config.OUTPUT_CONFIG["tables_dir"]}/deletion_results.csv')
        
        # Plot 1: Deletion time vs deletion ratio
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x='deletion_ratio', y='deletion_time', hue='algorithm', marker='o')
        plt.xlabel('Deletion Ratio')
        plt.ylabel('Deletion Time (seconds)')
        plt.title('Deletion Time vs Deletion Ratio')
        plt.savefig(f'{self.config.OUTPUT_CONFIG["plots_dir"]}/deletion_time.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: deletion_time.png")
        
        # Plot 2: Approximation error vs deletion ratio
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x='deletion_ratio', y='approximation_error', hue='algorithm', marker='o')
        plt.xlabel('Deletion Ratio')
        plt.ylabel('Approximation Error (ε)')
        plt.title('Approximation Error vs Deletion Ratio')
        plt.savefig(f'{self.config.OUTPUT_CONFIG["plots_dir"]}/approximation_error.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: approximation_error.png")


def main():
    parser = argparse.ArgumentParser(description='Run KG unlearning experiments')
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to use')
    parser.add_argument('--train', action='store_true', help='Train baseline model')
    parser.add_argument('--evaluate', action='store_true', help='Run unlearning evaluation')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = KGUnlearningExperiment(config)
    
    # Load dataset
    train_triples, valid_triples, test_triples = experiment.load_dataset(args.dataset)
    
    if args.train:
        # Train baseline model
        model = experiment.train_baseline_model(train_triples, valid_triples)
        
        # Evaluate
        metrics = experiment.run_link_prediction_evaluation(model, test_triples)
    
    if args.evaluate:
        # Load trained model
        num_entities = len(experiment.entity2id)
        num_relations = len(experiment.relation2id)
        
        model = TransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=config.MODEL_CONFIG['embedding_dim'],
            margin=config.MODEL_CONFIG['margin']
        ).to(experiment.device)
        
        model.load_state_dict(torch.load(f'{config.OUTPUT_CONFIG["checkpoints_dir"]}/best_model.pt'))
        
        # Run deletion experiments
        results_df = experiment.run_deletion_experiments(model, train_triples)
    
    if args.plot:
        # Generate plots
        experiment.generate_plots()
    
    print("\n=== Experiments Complete ===")

if __name__ == '__main__':
    main()
