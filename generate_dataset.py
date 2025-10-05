"""
Generate synthetic knowledge graph with power-law degree distribution
"""

import numpy as np
import argparse
from neo4j import GraphDatabase
from tqdm import tqdm
import config

class SyntheticKGGenerator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """Clear existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def generate_power_law_degrees(self, num_entities, alpha=2.5, min_degree=1):
        """Generate degrees following power-law distribution"""
        # Sample from power law: P(k) ~ k^(-alpha)
        degrees = np.random.zipf(alpha, num_entities)
        degrees = np.clip(degrees, min_degree, num_entities // 10)
        return degrees
    
    def generate_entities(self, num_entities):
        """Create entity nodes"""
        print(f"Creating {num_entities} entities...")
        with self.driver.session() as session:
            for i in tqdm(range(num_entities)):
                session.run(
                    "CREATE (e:Entity {id: $id, name: $name})",
                    id=i,
                    name=f"Entity_{i}"
                )
    
    def generate_relations(self, num_relations):
        """Define relation types"""
        relations = [f"REL_{i}" for i in range(num_relations)]
        return relations
    
    def generate_triples(self, num_entities, num_relations, target_triples, 
                        degree_dist='power_law', alpha=2.5):
        """Generate triples with specified degree distribution"""
        print(f"Generating {target_triples} triples...")
        
        relations = self.generate_relations(num_relations)
        
        if degree_dist == 'power_law':
            out_degrees = self.generate_power_law_degrees(num_entities, alpha)
            # Normalize to match target number of triples
            out_degrees = (out_degrees / out_degrees.sum() * target_triples).astype(int)
        else:
            # Uniform distribution
            out_degrees = np.full(num_entities, target_triples // num_entities)
        
        triples_created = 0
        batch_size = 1000
        batch = []
        
        with self.driver.session() as session:
            for head_id in tqdm(range(num_entities)):
                num_edges = out_degrees[head_id]
                
                for _ in range(num_edges):
                    tail_id = np.random.randint(0, num_entities)
                    if tail_id == head_id:
                        continue
                        
                    relation = np.random.choice(relations)
                    
                    batch.append({
                        'head': head_id,
                        'tail': tail_id,
                        'relation': relation
                    })
                    
                    if len(batch) >= batch_size:
                        self._create_batch_triples(session, batch)
                        triples_created += len(batch)
                        batch = []
            
            # Create remaining triples
            if batch:
                self._create_batch_triples(session, batch)
                triples_created += len(batch)
        
        print(f"Created {triples_created} triples")
        return triples_created
    
    def _create_batch_triples(self, session, batch):
        """Create batch of triples efficiently"""
        query = """
        UNWIND $triples AS triple
        MATCH (h:Entity {id: triple.head})
        MATCH (t:Entity {id: triple.tail})
        CREATE (h)-[r:RELATION {type: triple.relation}]->(t)
        """
        session.run(query, triples=batch)
    
    def get_statistics(self):
        """Get graph statistics"""
        with self.driver.session() as session:
            num_entities = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
            num_triples = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()['count']
            
            # Get degree statistics
            degree_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATION]->()
            WITH e, count(r) as out_degree
            RETURN 
                min(out_degree) as min_degree,
                max(out_degree) as max_degree,
                avg(out_degree) as avg_degree,
                stdev(out_degree) as std_degree
            """
            stats = session.run(degree_query).single()
            
            return {
                'num_entities': num_entities,
                'num_triples': num_triples,
                'min_degree': stats['min_degree'],
                'max_degree': stats['max_degree'],
                'avg_degree': stats['avg_degree'],
                'std_degree': stats['std_degree']
            }
    
    def export_to_files(self, output_dir='data/'):
        """Export graph to text files for processing"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with self.driver.session() as session:
            # Export entities
            entities = session.run("MATCH (e:Entity) RETURN e.id as id, e.name as name").data()
            with open(f"{output_dir}/entities.txt", 'w') as f:
                for entity in entities:
                    f.write(f"{entity['id']}\t{entity['name']}\n")
            
            # Export triples
            triples = session.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "RETURN h.id as head, r.type as relation, t.id as tail"
            ).data()
            
            with open(f"{output_dir}/triples.txt", 'w') as f:
                for triple in triples:
                    f.write(f"{triple['head']}\t{triple['relation']}\t{triple['tail']}\n")
            
            # Split into train/valid/test
            np.random.shuffle(triples)
            n = len(triples)
            train_end = int(0.8 * n)
            valid_end = int(0.9 * n)
            
            with open(f"{output_dir}/train.txt", 'w') as f:
                for triple in triples[:train_end]:
                    f.write(f"{triple['head']}\t{triple['relation']}\t{triple['tail']}\n")
            
            with open(f"{output_dir}/valid.txt", 'w') as f:
                for triple in triples[train_end:valid_end]:
                    f.write(f"{triple['head']}\t{triple['relation']}\t{triple['tail']}\n")
            
            with open(f"{output_dir}/test.txt", 'w') as f:
                for triple in triples[valid_end:]:
                    f.write(f"{triple['head']}\t{triple['relation']}\t{triple['tail']}\n")
            
            print(f"Exported to {output_dir}")
            print(f"  Train: {train_end} triples")
            print(f"  Valid: {valid_end - train_end} triples")
            print(f"  Test: {n - valid_end} triples")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic knowledge graph')
    parser.add_argument('--entities', type=int, default=10000, help='Number of entities')
    parser.add_argument('--relations', type=int, default=20, help='Number of relation types')
    parser.add_argument('--triples', type=int, default=100000, help='Target number of triples')
    parser.add_argument('--distribution', choices=['power_law', 'uniform'], 
                       default='power_law', help='Degree distribution')
    parser.add_argument('--alpha', type=float, default=2.5, 
                       help='Power law exponent')
    parser.add_argument('--clear', action='store_true', 
                       help='Clear existing database')
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticKGGenerator(
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD
    )
    
    try:
        if args.clear:
            generator.clear_database()
        
        # Generate graph
        generator.generate_entities(args.entities)
        generator.generate_triples(
            args.entities,
            args.relations,
            args.triples,
            args.distribution,
            args.alpha
        )
        
        # Print statistics
        stats = generator.get_statistics()
        print("\nGraph Statistics:")
        print(f"  Entities: {stats['num_entities']}")
        print(f"  Triples: {stats['num_triples']}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Max Degree: {stats['max_degree']}")
        print(f"  Degree Std: {stats['std_degree']:.2f}")
        
        # Export to files
        generator.export_to_files('data/synthetic/')
        
    finally:
        generator.close()

if __name__ == '__main__':
    main()
