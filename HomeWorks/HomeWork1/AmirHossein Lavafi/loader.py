import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class CoraDataLoader:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.G = None
        self.features = None
        self.labels = None
        self.node_ids = None
        
    def load_content(self, content_file='cora.content'):
        """Load the .content file with paper features and labels"""
        content_path = f"{self.data_dir}/{content_file}"
        
        # Read the content file
        df = pd.read_csv(content_path, sep='\t', header=None)
        
        # Extract paper IDs (first column)
        self.node_ids = df.iloc[:, 0].values
        
        # Extract features (all columns except first and last)
        self.features = df.iloc[:, 1:-1].values
        
        # Extract labels (last column)
        self.labels = df.iloc[:, -1].values
        
        # Create label encoder for class labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Create mapping from paper ID to index
        self.id_to_idx = {paper_id: idx for idx, paper_id in enumerate(self.node_ids)}
        
        print(f"Loaded content: {len(self.node_ids)} papers, {self.features.shape[1]} features")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Class distribution: {np.bincount(self.encoded_labels)}")
        
        return self.features, self.labels, self.node_ids
    
    def load_citations(self, cites_file='cora.cites'):
        """Load the .cites file and build the citation graph"""
        cites_path = f"{self.data_dir}/{cites_file}"
        
        # Create directed graph
        self.G = nx.DiGraph()
        
        # Add nodes with features and labels
        for idx, paper_id in enumerate(self.node_ids):
            self.G.add_node(paper_id, 
                          features=self.features[idx],
                          label=self.labels[idx],
                          encoded_label=self.encoded_labels[idx],
                          idx=idx)
        
        # Read citation edges
        citations_df = pd.read_csv(cites_path, sep='\t', header=None, 
                                 names=['cited_paper', 'citing_paper'])
        
        # Add edges (citing_paper -> cited_paper)
        for _, row in citations_df.iterrows():
            cited = row['cited_paper']
            citing = row['citing_paper']
            
            # Only add edges if both papers are in our content file
            if cited in self.id_to_idx and citing in self.id_to_idx:
                self.G.add_edge(citing, cited)  # Direction: citing -> cited
        
        print(f"Loaded citations: {self.G.number_of_edges()} edges")
        print(f"Graph is directed: {self.G.is_directed()}")
        
        return self.G
    
    def load_complete_dataset(self, content_file='cora.content', cites_file='cora.cites'):
        """Load both content and citations"""
        self.load_content(content_file)
        self.load_citations(cites_file)
        return self.G, self.features, self.labels, self.node_ids