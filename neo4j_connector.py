"""
Neo4j database connector and query utilities
"""

from neo4j import GraphDatabase
import numpy as np
from typing import List, Tuple, Dict
import config

class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        self.driver.close()
    
    def get_all_triples(self) -> List[Tuple[int, str, int]]:
        """Get all triples from the graph"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "RETURN h.id as head, r.type as relation, t.id as tail"
            )
            return [(record['head'], record['relation'], record['tail']) 
                    for record in result]
    
    def get_entity_embeddings(self, entity_ids: List[int]) -> Dict:
        """Get stored embeddings for entities"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids "
                "RETURN e.id as id, e.embedding as embedding",
                ids=entity_ids
            )
            return {record['id']: record['embedding'] for record in result}
    
    def delete_entity(self, entity_id: int):
        """Delete an entity and all incident edges"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) "
                "DETACH DELETE e "
                "RETURN count(e) as deleted",
                id=entity_id
            )
            return result.single()['deleted']
    
    def delete_triples(self, triples: List[Tuple[int, str, int]]):
        """Delete specific triples"""
        with self.driver.session() as session:
            count = 0
            for head, relation, tail in triples:
                result = session.run(
                    "MATCH (h:Entity {id: $head})-[r:RELATION {type: $rel}]->(t:Entity {id: $tail}) "
                    "DELETE r "
                    "RETURN count(r) as deleted",
                    head=head, rel=relation, tail=tail
                )
                count += result.single()['deleted']
            return count
    
    def get_entity_neighbors(self, entity_id: int, max_hops: int = 2) -> List[int]:
        """Get k-hop neighbors of an entity"""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (e:Entity {{id: $id}})-[*1..{max_hops}]-(neighbor:Entity) "
                "RETURN DISTINCT neighbor.id as id",
                id=entity_id
            )
            return [record['id'] for record in result]
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Basic counts
            stats['num_entities'] = session.run(
                "MATCH (e:Entity) RETURN count(e) as count"
            ).single()['count']
            
            stats['num_triples'] = session.run(
                "MATCH ()-[r:RELATION]->() RETURN count(r) as count"
            ).single()['count']
            
            # Degree distribution
            degree_stats = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r:RELATION]->()
                WITH e, count(r) as degree
                RETURN 
                    min(degree) as min_degree,
                    max(degree) as max_degree,
                    avg(degree) as avg_degree,
                    stdev(degree) as std_degree,
                    percentileCont(degree, 0.5) as median_degree,
                    percentileCont(degree, 0.9) as p90_degree,
                    percentileCont(degree, 0.99) as p99_degree
            """).single()
            
            stats.update(dict(degree_stats))
            
            # Relation type counts
            relation_counts = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN r.type as relation, count(r) as count
                ORDER BY count DESC
            """).data()
            
            stats['relation_distribution'] = relation_counts
            
            return stats
    
    def sample_random_entities(self, num_samples: int) -> List[int]:
        """Sample random entities"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) "
                "WITH e, rand() as r "
                "ORDER BY r "
                f"LIMIT {num_samples} "
                "RETURN e.id as id"
            )
            return [record['id'] for record in result]
    
    def get_high_degree_entities(self, top_k: int = 100) -> List[Tuple[int, int]]:
        """Get entities with highest degree (hubs)"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity)-[r:RELATION]->() "
                "WITH e, count(r) as degree "
                "ORDER BY degree DESC "
                f"LIMIT {top_k} "
                "RETURN e.id as id, degree"
            )
            return [(record['id'], record['degree']) for record in result]
    
    def execute_sparql_count(self, pattern: str) -> int:
        """Execute SPARQL-like COUNT query"""
        # Simplified SPARQL to Cypher conversion
        # For real implementation, use more sophisticated parser
        with self.driver.session() as session:
            result = session.run(pattern)
            return result.single()['count'] if result else 0
    
    def backup_graph(self, backup_name: str = 'backup'):
        """Create a backup of current graph state"""
        # This would use Neo4j backup utilities
        # For now, export to files
        pass
    
    def restore_graph(self, backup_name: str = 'backup'):
        """Restore graph from backup"""
        # This would use Neo4j restore utilities
        pass

# Context manager support
class Neo4jSession:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
    
    def __enter__(self):
        return self.connector
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connector.close()

# Usage example
if __name__ == '__main__':
    connector = Neo4jConnector()
    
    try:
        stats = connector.get_graph_statistics()
        print("Graph Statistics:")
        for key, value in stats.items():
            if key != 'relation_distribution':
                print(f"  {key}: {value}")
        
        print("\nTop 10 Entities by Degree:")
        hubs = connector.get_high_degree_entities(10)
        for entity_id, degree in hubs:
            print(f"  Entity {entity_id}: degree {degree}")
        
    finally:
        connector.close()
