import pandas as pd
from pathlib import Path
from app.graph.neo4j_client import Neo4jClientWrapper

class GraphIngestor:
    def __init__(self):
        self.neo4j_wrapper = Neo4jClientWrapper()

    def get_node_label(self, name: str, lookups: dict) -> str:
        """
        Determine the appropriate Neo4j label for a node based on patterns and lookups.
        """
        name_str = str(name).strip()
        name_lower = name_str.lower()
        
        if name_str.startswith("INC-"):
            return "Incident"
        elif name_str.startswith("CV-"):
            return "Component"
        elif "plant" in name_lower:
            return "Location"
        elif name_str in lookups.get("assets", set()):
            return "Asset"
        elif name_str in lookups.get("vendors", set()):
            return "Vendor"
        elif name_str in lookups.get("employees", set()):
            return "Employee"
        elif name_str in lookups.get("teams", set()):
            return "Team"
        else:
            return "Entity"

    def build_lookups(self, data_dir: Path) -> dict:
        """
        Build lookup sets from helper CSVs to assist in node classification.
        """
        lookups = {
            "assets": set(),
            "locations": set(),
            "vendors": set(),
            "employees": set(),
            "teams": set()
        }

        # Assets
        assets_csv = data_dir / "Assets.csv"
        if assets_csv.exists():
            df = pd.read_csv(assets_csv)
            if "Asset" in df.columns:
                lookups["assets"].update(df["Asset"].dropna().str.strip())
            if "Location" in df.columns:
                lookups["locations"].update(df["Location"].dropna().str.strip())
            if "Owner" in df.columns:
                lookups["employees"].update(df["Owner"].dropna().str.strip())

        # Vendors
        vendors_csv = data_dir / "Vendors.csv"
        if vendors_csv.exists():
            df = pd.read_csv(vendors_csv)
            if "Vendor" in df.columns:
                lookups["vendors"].update(df["Vendor"].dropna().str.strip())

        # Team Structure
        team_csv = data_dir / "Team_Structure.csv"
        if team_csv.exists():
            df = pd.read_csv(team_csv)
            if "Employee" in df.columns:
                lookups["employees"].update(df["Employee"].dropna().str.strip())
            if "Team" in df.columns:
                lookups["teams"].update(df["Team"].dropna().str.strip())

        return lookups

    def ingest(self, data_dir: Path):
        """
        Read graph_edges.csv, classify nodes, and ingest them into Neo4j.
        """
        # Create constraints first
        self.neo4j_wrapper.create_constraints()

        edges_csv = data_dir / "graph_edges.csv"
        if not edges_csv.exists():
            print(f"Error: graph_edges.csv not found in {data_dir}")
            return

        print("Building node classification lookups...")
        lookups = self.build_lookups(data_dir)

        print(f"Reading graph edges from {edges_csv}...")
        df = pd.read_csv(edges_csv)
        
        required_cols = {"source", "relationship", "target"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"graph_edges.csv must contain columns: {required_cols}")

        print("Starting Neo4j ingestion...")
        with self.neo4j_wrapper.driver.session() as session:
            for idx, row in df.iterrows():
                source = str(row["source"]).strip()
                relationship = str(row["relationship"]).strip().upper()
                target = str(row["target"]).strip()

                source_label = self.get_node_label(source, lookups)
                target_label = self.get_node_label(target, lookups)

                # 1. Create source node
                source_query = f"MERGE (n:{source_label} {{name: $name}})"
                session.run(source_query, {"name": source})

                # 2. Create target node
                target_query = f"MERGE (n:{target_label} {{name: $name}})"
                session.run(target_query, {"name": target})

                # 3. Create relationship
                rel_query = f"""
                MATCH (s:{source_label} {{name: $source_name}})
                MATCH (t:{target_label} {{name: $target_name}})
                MERGE (s)-[r:{relationship}]->(t)
                """
                session.run(rel_query, {
                    "source_name": source,
                    "target_name": target
                })
                
                print(f"Added: ({source}:{source_label}) -[:{relationship}]-> ({target}:{target_label})")

        print("Graph ingestion completed successfully.")
        self.neo4j_wrapper.close()

