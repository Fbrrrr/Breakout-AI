#!/usr/bin/env python3
"""
Utility functions for Breakout AI analysis and visualization

This module provides helper functions for analyzing training results,
visualizing network evolution, and testing trained models.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import neat
import logging

logger = logging.getLogger(__name__)

class TrainingAnalyzer:
    """Analyzes and visualizes NEAT training results."""
    
    def __init__(self, stats_file: str = None):
        """Initialize analyzer with optional statistics file."""
        self.stats_file = stats_file
        self.generation_data = []
        self.fitness_data = []
    
    def load_statistics(self, stats_file: str = None) -> None:
        """Load training statistics from file."""
        file_path = stats_file or self.stats_file
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Statistics file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.generation_data = data.get('generations', [])
                self.fitness_data = data.get('fitness_history', [])
        except Exception as e:
            logger.error(f"Error loading statistics: {e}")
    
    def plot_fitness_evolution(self, save_path: str = None) -> None:
        """Plot fitness evolution over generations."""
        if not self.fitness_data:
            logger.warning("No fitness data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        generations = range(len(self.fitness_data))
        max_fitness = [max(gen_fitness) for gen_fitness in self.fitness_data]
        avg_fitness = [np.mean(gen_fitness) for gen_fitness in self.fitness_data]
        min_fitness = [min(gen_fitness) for gen_fitness in self.fitness_data]
        
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness', linewidth=2)
        plt.plot(generations, min_fitness, 'g-', label='Min Fitness', linewidth=1)
        
        plt.fill_between(generations, min_fitness, max_fitness, alpha=0.2, color='gray')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fitness plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_species_evolution(self, save_path: str = None) -> None:
        """Plot species count and diversity over generations."""
        if not self.generation_data:
            logger.warning("No generation data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Species count over time
        generations = range(len(self.generation_data))
        species_counts = [gen.get('species_count', 0) for gen in self.generation_data]
        
        ax1.plot(generations, species_counts, 'purple', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Species')
        ax1.set_title('Species Diversity Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Network complexity over time
        avg_nodes = [gen.get('avg_nodes', 0) for gen in self.generation_data]
        avg_connections = [gen.get('avg_connections', 0) for gen in self.generation_data]
        
        ax2.plot(generations, avg_nodes, 'orange', label='Avg Nodes', linewidth=2)
        ax2.plot(generations, avg_connections, 'teal', label='Avg Connections', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Network Complexity')
        ax2.set_title('Network Complexity Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Species evolution plot saved to: {save_path}")
        else:
            plt.show()

class ModelTester:
    """Test and evaluate trained NEAT models."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize tester with model and config paths."""
        self.model_path = model_path
        self.config_path = config_path
        self.winner = None
        self.config = None
        self.network = None
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and configuration."""
        try:
            # Load winner genome
            with open(self.model_path, 'rb') as f:
                self.winner = pickle.load(f)
            
            # Load NEAT configuration
            self.config = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                self.config_path
            )
            
            # Create network
            self.network = neat.nn.FeedForwardNetwork.create(self.winner, self.config)
            
            logger.info(f"Model loaded successfully from: {self.model_path}")
            logger.info(f"Winner fitness: {self.winner.fitness}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_inputs(self, test_cases: List[List[float]]) -> List[float]:
        """Test the network with specific input cases."""
        if not self.network:
            raise ValueError("Network not loaded")
        
        results = []
        for inputs in test_cases:
            output = self.network.activate(inputs)
            results.append(output[0])
        
        return results
    
    def analyze_network_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the trained network."""
        if not self.winner:
            raise ValueError("Winner genome not loaded")
        
        # Count nodes and connections
        num_nodes = len(self.winner.nodes)
        num_connections = len([conn for conn in self.winner.connections.values() if conn.enabled])
        
        # Analyze connection weights
        weights = [conn.weight for conn in self.winner.connections.values() if conn.enabled]
        
        analysis = {
            'num_nodes': num_nodes,
            'num_connections': num_connections,
            'weight_stats': {
                'mean': np.mean(weights) if weights else 0,
                'std': np.std(weights) if weights else 0,
                'min': min(weights) if weights else 0,
                'max': max(weights) if weights else 0
            },
            'genome_size': self.winner.size(),
            'fitness': self.winner.fitness
        }
        
        return analysis
    
    def visualize_network(self, save_path: str = None) -> None:
        """Create a visual representation of the network structure."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            if not self.winner:
                raise ValueError("Winner genome not loaded")
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in self.winner.nodes.items():
                G.add_node(node_id, type='hidden')
            
            # Mark input and output nodes
            for i in range(self.config.genome_config.num_inputs):
                if -i-1 in G:
                    G.nodes[-i-1]['type'] = 'input'
            
            for i in range(self.config.genome_config.num_outputs):
                if i in G:
                    G.nodes[i]['type'] = 'output'
            
            # Add edges
            for conn in self.winner.connections.values():
                if conn.enabled:
                    G.add_edge(conn.key[0], conn.key[1], weight=conn.weight)
            
            # Create layout
            pos = {}
            input_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'input']
            output_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'output']
            hidden_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'hidden']
            
            # Position input nodes
            for i, node in enumerate(input_nodes):
                pos[node] = (0, i)
            
            # Position output nodes
            for i, node in enumerate(output_nodes):
                pos[node] = (2, i)
            
            # Position hidden nodes
            for i, node in enumerate(hidden_nodes):
                pos[node] = (1, i)
            
            # Draw network
            plt.figure(figsize=(12, 8))
            
            # Draw nodes by type
            nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, 
                                 node_color='lightblue', node_size=500, label='Input')
            nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, 
                                 node_color='lightcoral', node_size=500, label='Output')
            nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, 
                                 node_color='lightgreen', node_size=500, label='Hidden')
            
            # Draw edges with weights
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[abs(w)*2 for w in weights],
                                 edge_color=weights, edge_cmap=plt.cm.RdBu)
            
            # Add labels
            nx.draw_networkx_labels(G, pos)
            
            plt.title(f'Network Structure (Fitness: {self.winner.fitness:.2f})')
            plt.legend()
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Network visualization saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("NetworkX not available for network visualization")
        except Exception as e:
            logger.error(f"Error visualizing network: {e}")

def compare_models(model_paths: List[str], config_path: str) -> None:
    """Compare multiple trained models."""
    models = []
    
    for path in model_paths:
        try:
            tester = ModelTester(path, config_path)
            analysis = tester.analyze_network_structure()
            analysis['path'] = path
            models.append(analysis)
        except Exception as e:
            logger.error(f"Error loading model {path}: {e}")
    
    if not models:
        logger.error("No models loaded successfully")
        return
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    model_names = [os.path.basename(m['path']) for m in models]
    
    # Fitness comparison
    fitnesses = [m['fitness'] for m in models]
    ax1.bar(model_names, fitnesses, color='skyblue')
    ax1.set_title('Model Fitness Comparison')
    ax1.set_ylabel('Fitness Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Network size comparison
    node_counts = [m['num_nodes'] for m in models]
    conn_counts = [m['num_connections'] for m in models]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, node_counts, width, label='Nodes', color='lightgreen')
    ax2.bar(x + width/2, conn_counts, width, label='Connections', color='lightcoral')
    ax2.set_title('Network Complexity Comparison')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    
    # Weight statistics
    weight_means = [m['weight_stats']['mean'] for m in models]
    weight_stds = [m['weight_stats']['std'] for m in models]
    
    ax3.bar(model_names, weight_means, color='orange', alpha=0.7)
    ax3.set_title('Average Connection Weights')
    ax3.set_ylabel('Weight Value')
    ax3.tick_params(axis='x', rotation=45)
    
    ax4.bar(model_names, weight_stds, color='purple', alpha=0.7)
    ax4.set_title('Weight Standard Deviation')
    ax4.set_ylabel('Standard Deviation')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_training_report(stats_file: str, model_file: str, config_file: str, 
                          output_dir: str = "training_report") -> None:
    """Generate a comprehensive training report with plots and analysis."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzers
    analyzer = TrainingAnalyzer(stats_file)
    analyzer.load_statistics()
    
    # Generate plots
    print("Generating fitness evolution plot...")
    analyzer.plot_fitness_evolution(os.path.join(output_dir, "fitness_evolution.png"))
    
    print("Generating species evolution plot...")
    analyzer.plot_species_evolution(os.path.join(output_dir, "species_evolution.png"))
    
    # Analyze final model
    if os.path.exists(model_file):
        print("Analyzing final model...")
        tester = ModelTester(model_file, config_file)
        analysis = tester.analyze_network_structure()
        
        # Save network visualization
        tester.visualize_network(os.path.join(output_dir, "network_structure.png"))
        
        # Save analysis as text report
        with open(os.path.join(output_dir, "model_analysis.txt"), 'w') as f:
            f.write("BREAKOUT AI - NEAT TRAINING REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Final Model Analysis:\n")
            f.write(f"  - Fitness Score: {analysis['fitness']:.2f}\n")
            f.write(f"  - Network Nodes: {analysis['num_nodes']}\n")
            f.write(f"  - Active Connections: {analysis['num_connections']}\n")
            f.write(f"  - Average Weight: {analysis['weight_stats']['mean']:.4f}\n")
            f.write(f"  - Weight Std Dev: {analysis['weight_stats']['std']:.4f}\n")
            f.write(f"  - Weight Range: [{analysis['weight_stats']['min']:.4f}, {analysis['weight_stats']['max']:.4f}]\n")
        
        print(f"Training report generated in: {output_dir}")
    else:
        print(f"Model file not found: {model_file}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze NEAT training results")
    parser.add_argument("--stats", help="Statistics file path")
    parser.add_argument("--model", help="Trained model file path")
    parser.add_argument("--config", help="NEAT config file path")
    parser.add_argument("--output", default="training_report", help="Output directory")
    
    args = parser.parse_args()
    
    if args.stats and args.model and args.config:
        create_training_report(args.stats, args.model, args.config, args.output)
    else:
        print("Please provide --stats, --model, and --config file paths") 