from neo4j import GraphDatabase
from app.config import settings

class Neo4jClientWrapper:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: dict = None):
        """
        Run a Cypher query with optional parameters and return the results.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_transaction(self, write_fn, *args, **kwargs):
        """
        Run a write transaction using a custom function.
        """
        with self.driver.session() as session:
            return session.execute_write(write_fn, *args, **kwargs)

    def clear_database(self):
        """
        Delete all nodes and relationships in the database (useful for reset).
        """
        print("Clearing Neo4j database...")
        self.run_query("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_constraints(self):
        """
        Create standard constraints (e.g. unique constraint on Entity name/id).
        """
        # Since we use dynamic node labeling, we could create unique constraints on name/id property for individual labels.
        # For simplicity, if we use various labels, we can dynamically apply constraints, or just make sure to MERGE on a unique key.
        # Neo4j supports constraints, but MERGE does not strictly require them if the query is written properly.
        # Let's add constraints for labels we expect to use frequently, such as Asset, Component, Vendor, Employee, Team, Incident, Location, Entity.
        labels = ["Asset", "Component", "Vendor", "Employee", "Team", "Incident", "Location", "Entity"]
        for label in labels:
            try:
                # Neo4j 5 syntax for constraint creation
                self.run_query(f"CREATE CONSTRAINT unique_{label.lower()}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create constraint for label {label}: {e}")

