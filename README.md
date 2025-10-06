# Knowledge Graph Unlearning Experiments

Implementation of "Towards Forgetting in Knowledge Graphs: A Unified Framework for Machine Unlearning and Differential Privacy"

**Anonymous Submission for EACL 2026**

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
