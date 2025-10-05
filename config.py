"""
Configuration file for KG unlearning experiments
"""

import os

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
# Dataset Configuration
DATASET_CONFIG = {
    'synthetic': {
        'num_entities': 10000,
        'num_relations': 20,
        'num_triples': 100000,
        'degree_distribution': 'power_law',  # 'uniform' or 'power_law'
        'alpha': 2.5  # Power law exponent
    },
    'fb15k237': {
        'path': 'data/FB15k-237/',
        'train': 'train.txt',
        'valid': 'valid.txt',
        'test': 'test.txt'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 128,
    'learning_rate': 0.001,
    'batch_size': 512,
    'num_epochs': 100,
    'margin': 1.0,
    'model_type': 'TransE',  # 'TransE' or 'RotatE'
    'negative_samples': 5
}

# Unlearning Configuration
UNLEARNING_CONFIG = {
    'isr': {
        'threshold_tau': 0.01,
        'learning_rate': 0.0001,
        'refinement_epochs': 10,
        'hessian_approx': 'lbfgs'  # 'lbfgs' or 'conjugate_gradient'
    },
    'skgl': {
        'num_shards': 8,
        'partition_method': 'metis',  # 'metis' or 'random'
        'min_cut': True
    },
    'igf': {
        'lambda_reg': 0.1,
        'beta_tv': 0.01,
        'gate_init': 1.0,
        'forgetting_epochs': 50
    }
}

# Privacy Configuration
PRIVACY_CONFIG = {
    'epsilon': 1.0,
    'delta': 1e-5,
    'clipping_threshold': 1.0,
    'noise_multiplier': 1.0,
    'budget_allocation': {
        'link_prediction': 0.4,
        'sparql': 0.3,
        'pattern_mining': 0.3
    }
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'deletion_ratios': [0.001, 0.005, 0.01, 0.05, 0.1],
    'num_runs': 5,
    'random_seed': 42,
    'metrics': ['deletion_time', 'approximation_error', 'link_prediction_accuracy'],
    'save_checkpoints': True
}

# Output Configuration
OUTPUT_CONFIG = {
    'results_dir': 'results/',
    'plots_dir': 'results/plots/',
    'tables_dir': 'results/tables/',
    'checkpoints_dir': 'checkpoints/',
    'logs_dir': 'logs/'
}

# Create output directories
for dir_path in OUTPUT_CONFIG.values():
    os.makedirs(dir_path, exist_ok=True)
