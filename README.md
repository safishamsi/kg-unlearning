# Knowledge Graph Unlearning Experiments

Implementation of "Towards Forgetting in Knowledge Graphs: A Unified Framework for Machine Unlearning and Differential Privacy"

## Authors
Safi Shamsi, Laraib Hasan  
School of Computer Science, University of Birmingham

## Setup Instructions

### Prerequisites
- Python 3.8+
- Neo4j Database (Desktop or Docker)
- 8GB RAM minimum

### Installation

1. Install Neo4j:
```bash
# Option 1: Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Option 2: Neo4j Desktop
# Download from https://neo4j.com/download/
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. Generate synthetic knowledge graph:
```bash
python generate_dataset.py --entities 10000 --relations 20 --density 0.01
```

2. Train baseline model:
```bash
python train_baseline.py --epochs 100 --embedding_dim 128
```

3. Run unlearning experiments:
```bash
# ISR (Influence-based Selective Retraining)
python run_isr.py --delete_ratio 0.01

# SKGL (Sharded Knowledge Graph Learning)
python run_skgl.py --num_shards 8 --delete_ratio 0.01

# IGF (Incremental Graph Forgetting)
python run_igf.py --delete_ratio 0.01 --lambda_reg 0.1
```

4. Run privacy experiments:
```bash
python run_dp_queries.py --epsilon 1.0 --delta 1e-5
```

5. Generate results and plots:
```bash
python generate_results.py --output results/
```

## Project Structure

```
kg_unlearning_experiments/
├── README.md
├── requirements.txt
├── config.py                  # Configuration parameters
├── generate_dataset.py        # Synthetic KG generation
├── neo4j_connector.py         # Neo4j database interface
├── models/
│   ├── kg_embedding.py        # TransE/RotatE embeddings
│   ├── training.py            # Training utilities
│   └── evaluation.py          # Evaluation metrics
├── unlearning/
│   ├── isr.py                 # Influence-based Selective Retraining
│   ├── skgl.py                # Sharded Knowledge Graph Learning
│   └── igf.py                 # Incremental Graph Forgetting
├── privacy/
│   ├── dp_mechanisms.py       # Differential privacy mechanisms
│   ├── query_sensitivity.py   # Sensitivity analysis
│   └── budget_tracker.py      # Privacy budget management
├── experiments/
│   ├── run_isr.py
│   ├── run_skgl.py
│   ├── run_igf.py
│   └── run_dp_queries.py
├── utils/
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Plotting utilities
│   └── logger.py              # Experiment logging
└── results/                   # Experimental results
    ├── plots/
    └── tables/
```

## Datasets

### Synthetic Dataset
- 10,000 entities
- 20 relation types
- ~100,000 triples
- Power-law degree distribution

### Real-world Datasets (Optional)
- FB15k-237 (14,541 entities, 237 relations)
- YAGO3-10 (123,182 entities, 37 relations)

Download scripts provided in `data/download_datasets.sh`

## Experiments

### Experiment 1: Unlearning Efficiency
Measure deletion time vs dataset size for ISR, SKGL, IGF

### Experiment 2: Forgetting Quality
Measure approximation error ε between unlearned and retrained models

### Experiment 3: Utility Preservation
Link prediction accuracy before/after unlearning

### Experiment 4: Privacy-Utility Tradeoff
Query accuracy vs privacy budget (ε)

### Experiment 5: Scalability
Performance on datasets from 1K to 100K entities

## Expected Results

Results will be saved in `results/` directory:
- `deletion_time.csv` - Time measurements
- `approximation_error.csv` - Forgetting quality
- `link_prediction_results.csv` - Accuracy metrics
- `privacy_utility.csv` - DP query results
- `plots/` - Visualization figures

## Citation

```bibtex
@inproceedings{shamsi2026forgetting,
  title={Towards Forgetting in Knowledge Graphs: A Unified Framework for Machine Unlearning and Differential Privacy},
  author={Shamsi, Safi and Hasan, Laraib},
  booktitle={Proceedings of EACL 2026},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact:
- Safi Shamsi (University of Birmingham)
- Laraib Hasan (University of Birmingham)
