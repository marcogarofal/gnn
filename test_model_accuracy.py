import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from typing import Tuple, Dict, Optional, List
import json

class MSTDirectPredictor(nn.Module):
    """GNN che predice direttamente l'MST dato solo il grafo"""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int,
                 hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # GNN layers per catturare struttura globale
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 3 == 0:
                # GCN per propagazione efficiente
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif i % 3 == 1:
                # GAT per attention sui vicini importanti
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1))
            else:
                # Edge convolution per considerare features degli edges
                self.gnn_layers.append(EdgeConv(
                    nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                ))

        # Edge classifier migliorato
        self.edge_predictor = nn.Sequential(
            # Input: concatenazione di source, dest, edge features, e context globale
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        # Contesto globale del grafo
        self.graph_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Predice quali edges appartengono all'MST

        Returns:
            edge_logits: logits per ogni edge (alta probabilità = nell'MST)
        """

        # Encode features
        h = self.node_encoder(x)
        edge_h = self.edge_encoder(edge_attr)

        # Apply GNN layers con residual connections
        for i, layer in enumerate(self.gnn_layers):
            h_new = layer(h, edge_index)
            h = F.relu(h_new)
            # Residual connection ogni 2 layers
            if i % 2 == 1 and i > 0:
                h = h + h_new

        # Calcola contesto globale
        if batch is not None:
            # Se abbiamo batch, aggrega per grafo
            from torch_geometric.nn import global_mean_pool
            global_context = global_mean_pool(h, batch)
            # Espandi per ogni nodo
            global_context = global_context[batch]
        else:
            global_context = h.mean(dim=0, keepdim=True).expand(h.size(0), -1)

        global_context = self.graph_context(global_context)

        # Predici per ogni edge
        edge_predictions = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]

            # Concatena tutte le informazioni rilevanti
            edge_repr = torch.cat([
                h[src],                 # Nodo sorgente
                h[dst],                 # Nodo destinazione
                edge_h[i],              # Features dell'edge
                global_context[src]     # Contesto globale
            ], dim=0)

            edge_pred = self.edge_predictor(edge_repr)
            edge_predictions.append(edge_pred)

        edge_logits = torch.cat(edge_predictions, dim=0)

        return edge_logits


class MSTDirectDataset:
    """Dataset semplificato: solo grafo → MST ottimale"""

    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.processed_data = []
        self._process_dataset()

    def _process_dataset(self):
        """Prepara dataset con solo MST ottimali"""

        for graph_data in self.raw_data:
            # Prendi SOLO l'MST ottimale (assumiamo sia il primo)
            optimal_tree = graph_data['trees'][0]

            if not optimal_tree['is_optimal']:
                # Cerca quello ottimale
                for tree in graph_data['trees']:
                    if tree['is_optimal']:
                        optimal_tree = tree
                        break

            data = Data(
                x=graph_data['node_features'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data['edge_attr'],
                edge_weights=graph_data['edge_attr'][:, 0],
                mst_labels=torch.tensor(optimal_tree['edge_labels'], dtype=torch.float),
                num_nodes=graph_data['num_nodes'],
                graph_id=graph_data['graph_id']
            )

            self.processed_data.append(data)

    def get_dataloaders(self, batch_size: int = 32):
        """Split in train/val/test"""

        # Split per grafi
        indices = list(range(len(self.processed_data)))
        train_idx, test_idx = train_test_split(indices, train_size=0.8, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, train_size=0.9, random_state=42)

        train_loader = DataLoader(
            [self.processed_data[i] for i in train_idx],
            batch_size=batch_size, shuffle=False  # Non shuffle per test
        )
        val_loader = DataLoader(
            [self.processed_data[i] for i in val_idx],
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            [self.processed_data[i] for i in test_idx],
            batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


class ModelTester:
    """Classe per testare accuracy del modello su diversi dataset"""

    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Carica modello
        print(f"Caricamento modello da {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.model = MSTDirectPredictor(
            node_feature_dim=checkpoint['node_feature_dim'],
            edge_feature_dim=checkpoint['edge_feature_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Modello caricato su {device}")

    def evaluate_dataset(self, data_loader: DataLoader, dataset_name: str) -> Dict[str, float]:
        """Valuta il modello su un dataset"""
        
        print(f"\nValutazione su {dataset_name}...")
        
        all_predictions = []
        all_labels = []
        all_edge_weights = []
        tree_quality_gaps = []
        valid_trees = 0
        total_graphs = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                batch = batch.to(self.device)
                
                # Forward pass
                edge_logits = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, 
                    getattr(batch, 'batch', None)
                )
                
                # Predizioni
                predictions = (torch.sigmoid(edge_logits) > 0.5).float()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(batch.mst_labels.cpu())
                all_edge_weights.append(batch.edge_weights.cpu())
                
                # Analisi per grafo singolo (se non in batch)
                if not hasattr(batch, 'batch'):
                    # Calcola peso dell'albero predetto vs ottimale
                    pred_weight = (predictions.cpu() * batch.edge_weights.cpu()).sum() / 2
                    true_weight = (batch.mst_labels.cpu() * batch.edge_weights.cpu()).sum() / 2
                    
                    if true_weight > 0:
                        gap = (pred_weight - true_weight) / true_weight
                        tree_quality_gaps.append(gap.item())
                    
                    # Verifica se l'albero predetto è valido
                    predicted_edges = (predictions.cpu() > 0.5).sum().item() / 2
                    expected_edges = batch.num_nodes - 1
                    
                    if abs(predicted_edges - expected_edges) <= 1:  # Tolleranza di 1 edge
                        valid_trees += 1
                    
                    total_graphs += 1

        # Concatena tutti i risultati
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_edge_weights = torch.cat(all_edge_weights)

        # Calcola metriche
        accuracy = (all_predictions == all_labels).float().mean()

        # Metriche per MST edges (classe positiva)
        tp = ((all_predictions == 1) & (all_labels == 1)).sum().float()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum().float()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum().float()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp + 1e-8)

        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item(),
            'specificity': specificity.item(),
            'avg_quality_gap': np.mean(tree_quality_gaps) if tree_quality_gaps else 0,
            'std_quality_gap': np.std(tree_quality_gaps) if tree_quality_gaps else 0,
            'valid_trees_ratio': valid_trees / total_graphs if total_graphs > 0 else 0,
            'total_graphs': total_graphs,
            'total_edges': len(all_predictions)
        }

        return metrics

    def test_training_dataset(self, dataset_path: str):
        """Testa il modello sul dataset di training"""
        
        print("="*80)
        print("TEST SU DATASET DI TRAINING")
        print("="*80)
        
        # Carica dataset
        dataset = MSTDirectDataset(dataset_path)
        train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = dataset.get_dataloaders(batch_size=1)
        
        print(f"Dataset caricato:")
        print(f"  - Training: {len(train_idx)} grafi")
        print(f"  - Validation: {len(val_idx)} grafi") 
        print(f"  - Test: {len(test_idx)} grafi")
        
        # Testa su tutti e tre i set
        results = {}
        
        # Training set
        results['training'] = self.evaluate_dataset(train_loader, "Training Set")
        
        # Validation set
        results['validation'] = self.evaluate_dataset(val_loader, "Validation Set")
        
        # Test set
        results['test'] = self.evaluate_dataset(test_loader, "Test Set")
        
        return results

    def test_generalization_dataset(self, test_dataset_path: str):
        """Testa il modello sul dataset di generalizzazione"""
        
        print("\n" + "="*80)
        print("TEST SU DATASET DI GENERALIZZAZIONE")
        print("="*80)
        
        # Carica dataset di test
        with open(test_dataset_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"Dataset di generalizzazione caricato: {len(test_data)} grafi")
        
        # Raggruppa per configurazione
        configs = {}
        for entry in test_data:
            config_key = entry['config_info']['config_group']
            if config_key not in configs:
                configs[config_key] = []
            configs[config_key].append(entry)
        
        print(f"Configurazioni trovate: {list(configs.keys())}")
        
        # Testa ogni configurazione
        config_results = {}
        
        for config_name, config_data in configs.items():
            print(f"\nTestando configurazione: {config_name}")
            
            # Converti in dataset format
            processed_data = []
            for graph_data in config_data:
                optimal_tree = graph_data['trees'][0]  # MST è sempre il primo
                
                data = Data(
                    x=graph_data['node_features'],
                    edge_index=graph_data['edge_index'],
                    edge_attr=graph_data['edge_attr'],
                    edge_weights=graph_data['edge_attr'][:, 0],
                    mst_labels=torch.tensor(optimal_tree['edge_labels'], dtype=torch.float),
                    num_nodes=graph_data['num_nodes'],
                    graph_id=graph_data['graph_id']
                )
                processed_data.append(data)
            
            # Crea dataloader
            config_loader = DataLoader(processed_data, batch_size=1, shuffle=False)
            
            # Valuta
            config_results[config_name] = self.evaluate_dataset(config_loader, config_name)
        
        return config_results

    def print_results_summary(self, training_results: Dict, generalization_results: Dict):
        """Stampa summary completo dei risultati"""
        
        print("\n" + "="*100)
        print("SUMMARY COMPLETO DEI RISULTATI")
        print("="*100)
        
        # Risultati training
        print("\n📊 PERFORMANCE SU DATASET DI TRAINING:")
        print("-" * 60)
        
        for set_name, metrics in training_results.items():
            print(f"\n{set_name.upper()}:")
            print(f"  🎯 Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  🔍 Precision:    {metrics['precision']:.4f}")
            print(f"  📈 Recall:       {metrics['recall']:.4f}")
            print(f"  🏆 F1 Score:     {metrics['f1_score']:.4f}")
            print(f"  📊 Specificity:  {metrics['specificity']:.4f}")
            print(f"  💰 Avg Quality Gap: {metrics['avg_quality_gap']:+.2%}")
            print(f"  ✅ Valid Trees:  {metrics['valid_trees_ratio']:.2%}")
        
        # Risultati generalizzazione
        print(f"\n📊 PERFORMANCE SU DATASET DI GENERALIZZAZIONE:")
        print("-" * 60)
        
        # Raggruppa per densità e dimensione
        density_groups = {'standard': [], 'high': [], 'low': []}
        size_groups = {25: [], 40: [], 60: []}
        
        for config_name, metrics in generalization_results.items():
            # Estrai densità e dimensione
            parts = config_name.split('_')
            density = parts[0]
            size = int(parts[1])
            
            if density in density_groups:
                density_groups[density].append(metrics)
            size_groups[size].append(metrics)
            
            # Stampa dettaglio per configurazione
            print(f"\n{config_name.replace('_', ' ').title()}:")
            print(f"  🎯 Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  🏆 F1 Score:     {metrics['f1_score']:.4f}")
            print(f"  💰 Quality Gap:  {metrics['avg_quality_gap']:+.2%}")
            print(f"  ✅ Valid Trees:  {metrics['valid_trees_ratio']:.2%}")
        
        # Analisi per densità
        print(f"\n📈 ANALISI PER DENSITÀ:")
        print("-" * 40)
        
        for density, metrics_list in density_groups.items():
            if metrics_list:
                avg_acc = np.mean([m['accuracy'] for m in metrics_list])
                avg_f1 = np.mean([m['f1_score'] for m in metrics_list])
                avg_gap = np.mean([m['avg_quality_gap'] for m in metrics_list])
                
                print(f"{density.capitalize()} Density:")
                print(f"  Avg Accuracy: {avg_acc:.4f}")
                print(f"  Avg F1:       {avg_f1:.4f}") 
                print(f"  Avg Gap:      {avg_gap:+.2%}")
        
        # Analisi per dimensione
        print(f"\n📏 ANALISI PER DIMENSIONE:")
        print("-" * 40)
        
        for size, metrics_list in size_groups.items():
            if metrics_list:
                avg_acc = np.mean([m['accuracy'] for m in metrics_list])
                avg_f1 = np.mean([m['f1_score'] for m in metrics_list])
                avg_gap = np.mean([m['avg_quality_gap'] for m in metrics_list])
                
                print(f"{size} Nodi:")
                print(f"  Avg Accuracy: {avg_acc:.4f}")
                print(f"  Avg F1:       {avg_f1:.4f}")
                print(f"  Avg Gap:      {avg_gap:+.2%}")

    def save_results(self, training_results: Dict, generalization_results: Dict, 
                     output_path: str = 'model_evaluation_results.json'):
        """Salva tutti i risultati in JSON"""
        
        results = {
            'training_dataset': training_results,
            'generalization_dataset': generalization_results,
            'summary': {
                'training_avg_f1': np.mean([m['f1_score'] for m in training_results.values()]),
                'generalization_avg_f1': np.mean([m['f1_score'] for m in generalization_results.values()]),
                'best_config': max(generalization_results.items(), key=lambda x: x[1]['f1_score'])[0],
                'worst_config': min(generalization_results.items(), key=lambda x: x[1]['f1_score'])[0]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Risultati salvati in {output_path}")
































import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import random

def visualize_predicted_mst_examples(model_tester, test_dataset_path: str, num_examples: int = 6):
    """
    Visualizza esempi di MST predetti dal modello vs MST veri
    """
    
    print("🎨 VISUALIZZAZIONE ALBERI PREDETTI")
    print("="*60)
    
    # Carica dataset di test
    with open(test_dataset_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Seleziona esempi rappresentativi dalle diverse configurazioni
    configs = {}
    for entry in test_data:
        config_key = entry['config_info']['config_group']
        if config_key not in configs:
            configs[config_key] = []
        configs[config_key].append(entry)
    
    # Prendi un esempio per alcune configurazioni interessanti
    target_configs = ['standard_25', 'high_40', 'low_60']  # Varie configurazioni
    selected_examples = []
    
    for config in target_configs:
        if config in configs:
            # Prendi il primo grafo di questa configurazione
            selected_examples.append(configs[config][0])
    
    # Se non abbiamo abbastanza esempi, aggiungiamo altri random
    while len(selected_examples) < min(num_examples, 6):
        random_config = random.choice(list(configs.keys()))
        random_graph = random.choice(configs[random_config])
        if random_graph not in selected_examples:
            selected_examples.append(random_graph)
    
    # Crea figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, graph_data in enumerate(selected_examples[:6]):
        print(f"\n📊 Esempio {idx+1}: {graph_data['config_info']['config_group']}")
        
        # Prepara data per il modello
        optimal_tree = graph_data['trees'][0]
        data = Data(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data['edge_attr'],
            edge_weights=graph_data['edge_attr'][:, 0],
            mst_labels=torch.tensor(optimal_tree['edge_labels'], dtype=torch.float),
            num_nodes=graph_data['num_nodes'],
            graph_id=graph_data['graph_id']
        ).to(model_tester.device)
        
        # Predizione del modello
        with torch.no_grad():
            edge_logits = model_tester.model(data.x, data.edge_index, data.edge_attr)
            edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
        
        # Ricostruisci il grafo originale
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        edge_weights = data.edge_weights.cpu().numpy()
        
        # Aggiungi nodi
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Aggiungi edges (solo una direzione per evitare duplicati)
        edge_list = []
        for i in range(0, edge_index.shape[1], 2):  # Step 2 per evitare duplicati
            u, v = edge_index[:, i]
            weight = edge_weights[i]
            prob = edge_probs[i]
            G.add_edge(u, v, weight=weight, prob=prob)
            edge_list.append((u, v, weight, prob))
        
        # MST vero
        true_mst = nx.minimum_spanning_tree(G, weight='weight')
        
        # MST predetto (soglia 0.5)
        predicted_edges = [(u, v) for u, v, w, p in edge_list if p > 0.5]
        
        # MST predetto con soglia ottimale (per avere esattamente n-1 edges)
        edge_list_sorted = sorted(edge_list, key=lambda x: -x[3])  # Ordina per probabilità decrescente
        predicted_mst_edges = edge_list_sorted[:data.num_nodes-1]
        
        # Statistiche
        true_weight = sum(G[u][v]['weight'] for u, v in true_mst.edges())
        pred_weight_05 = sum(G[u][v]['weight'] for u, v in predicted_edges if G.has_edge(u, v))
        pred_weight_opt = sum(w for u, v, w, p in predicted_mst_edges)
        
        # Edges corretti
        true_edges = set(true_mst.edges())
        pred_edges_05 = set(predicted_edges)
        pred_edges_opt = set((u, v) for u, v, w, p in predicted_mst_edges)
        
        correct_05 = len(true_edges & pred_edges_05)
        correct_opt = len(true_edges & pred_edges_opt)
        
        print(f"  Nodi: {data.num_nodes}, Edges totali: {G.number_of_edges()}")
        print(f"  MST vero peso: {true_weight:.2f}")
        print(f"  Predetto (0.5): {len(predicted_edges)} edges, peso: {pred_weight_05:.2f}, corretti: {correct_05}/{len(true_edges)}")
        print(f"  Predetto (opt): {len(predicted_mst_edges)} edges, peso: {pred_weight_opt:.2f}, corretti: {correct_opt}/{len(true_edges)}")
        
        # Visualizzazione
        ax = axes[idx]
        
        # Layout del grafo
        if data.num_nodes <= 30:
            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        else:
            pos = nx.spring_layout(G, seed=42, k=1, iterations=30)
        
        # Disegna tutti gli edges in grigio chiaro
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.5, alpha=0.3)
        
        # Disegna MST vero in verde
        nx.draw_networkx_edges(G, pos, edgelist=true_mst.edges(), 
                             edge_color='green', width=2, alpha=0.8, ax=ax, label='MST Vero')
        
        # Disegna MST predetto in rosso
        nx.draw_networkx_edges(G, pos, edgelist=pred_edges_opt,
                             edge_color='red', width=2, alpha=0.6, ax=ax, 
                             style='dashed', label='MST Predetto')
        
        # Edges corretti in blu
        correct_edges = true_edges & pred_edges_opt
        if correct_edges:
            nx.draw_networkx_edges(G, pos, edgelist=correct_edges,
                                 edge_color='blue', width=3, alpha=0.9, ax=ax)
        
        # Disegna nodi
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=200 if data.num_nodes <= 30 else 100, 
                             ax=ax, alpha=0.7)
        
        # Labels dei nodi (solo per grafi piccoli)
        if data.num_nodes <= 25:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        # Titolo e info
        config_info = graph_data['config_info']
        ax.set_title(f"{config_info['density_mode'].title()} Density, {config_info['target_nodes']} Nodi\n"
                    f"Peso: Vero={true_weight:.1f}, Pred={pred_weight_opt:.1f}\n"
                    f"Corretti: {correct_opt}/{len(true_edges)} edges", 
                    fontsize=10)
        ax.axis('off')
        
        # Legenda solo per il primo subplot
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='MST Vero'),
                Line2D([0], [0], color='red', lw=2, linestyle='--', label='MST Predetto'),
                Line2D([0], [0], color='blue', lw=3, label='Edges Corretti')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('predicted_mst_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return selected_examples

def analyze_prediction_patterns(model_tester, test_dataset_path: str):
    """
    Analizza i pattern delle predizioni del modello
    """
    
    print("\n🔍 ANALISI PATTERN PREDIZIONI")
    print("="*50)
    
    # Carica dataset
    with open(test_dataset_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Analizza un campione di grafi
    sample_graphs = test_data[:10]  # Primi 10 grafi
    
    all_probabilities = []
    edge_weight_vs_prob = []
    correct_predictions = []
    
    for graph_data in sample_graphs:
        # Prepara data
        optimal_tree = graph_data['trees'][0]
        data = Data(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data['edge_attr'],
            edge_weights=graph_data['edge_attr'][:, 0],
            mst_labels=torch.tensor(optimal_tree['edge_labels'], dtype=torch.float),
            num_nodes=graph_data['num_nodes']
        ).to(model_tester.device)
        
        # Predizione
        with torch.no_grad():
            edge_logits = model_tester.model(data.x, data.edge_index, data.edge_attr)
            edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
        
        edge_weights = data.edge_weights.cpu().numpy()
        true_labels = data.mst_labels.cpu().numpy()
        
        # Analizza solo una direzione degli edges
        for i in range(0, len(edge_probs), 2):
            prob = edge_probs[i]
            weight = edge_weights[i]
            is_true_mst = true_labels[i]
            
            all_probabilities.append(prob)
            edge_weight_vs_prob.append((weight, prob, is_true_mst))
            correct_predictions.append(prob > 0.5 and is_true_mst or prob <= 0.5 and not is_true_mst)
    
    # Visualizzazioni
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribuzione probabilità
    axes[0,0].hist(all_probabilities, bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(0.5, color='red', linestyle='--', label='Soglia 0.5')
    axes[0,0].set_xlabel('Probabilità Edge')
    axes[0,0].set_ylabel('Frequenza')
    axes[0,0].set_title('Distribuzione Probabilità Predette')
    axes[0,0].legend()
    
    # 2. Peso vs Probabilità
    weights = [x[0] for x in edge_weight_vs_prob]
    probs = [x[1] for x in edge_weight_vs_prob]
    is_mst = [x[2] for x in edge_weight_vs_prob]
    
    # Scatter plot colorato
    mst_edges = [(w, p) for w, p, is_m in edge_weight_vs_prob if is_m]
    non_mst_edges = [(w, p) for w, p, is_m in edge_weight_vs_prob if not is_m]
    
    if mst_edges:
        mst_w, mst_p = zip(*mst_edges)
        axes[0,1].scatter(mst_w, mst_p, c='green', alpha=0.6, label='MST Edges', s=20)
    
    if non_mst_edges:
        non_mst_w, non_mst_p = zip(*non_mst_edges)
        axes[0,1].scatter(non_mst_w, non_mst_p, c='red', alpha=0.3, label='Non-MST Edges', s=20)
    
    axes[0,1].set_xlabel('Peso Edge')
    axes[0,1].set_ylabel('Probabilità Predetta')
    axes[0,1].set_title('Peso vs Probabilità')
    axes[0,1].legend()
    
    # 3. Accuracy per bin di peso
    weight_bins = np.linspace(min(weights), max(weights), 10)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(weight_bins)-1):
        low, high = weight_bins[i], weight_bins[i+1]
        bin_indices = [j for j, w in enumerate(weights) if low <= w < high]
        
        if bin_indices:
            bin_correct = [correct_predictions[j] for j in bin_indices]
            bin_accuracies.append(np.mean(bin_correct))
            bin_centers.append((low + high) / 2)
    
    if bin_accuracies:
        axes[1,0].bar(bin_centers, bin_accuracies, alpha=0.7, 
                     width=(max(weights)-min(weights))/15)
        axes[1,0].set_xlabel('Peso Edge (bin)')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Accuracy per Peso Edge')
        axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # 4. Soglia ottimale analysis
    thresholds = np.arange(0.1, 0.9, 0.01)
    accuracies = []
    
    for thresh in thresholds:
        correct = [p > thresh and is_m or p <= thresh and not is_m 
                  for p, is_m in zip(probs, is_mst)]
        accuracies.append(np.mean(correct))
    
    best_thresh = thresholds[np.argmax(accuracies)]
    best_acc = max(accuracies)
    
    axes[1,1].plot(thresholds, accuracies)
    axes[1,1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Soglia 0.5')
    axes[1,1].axvline(best_thresh, color='green', linestyle='--', alpha=0.7, 
                     label=f'Ottimale: {best_thresh:.3f}')
    axes[1,1].set_xlabel('Soglia')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('Accuracy vs Soglia')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('prediction_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Statistiche Pattern:")
    print(f"  - Probabilità media: {np.mean(all_probabilities):.3f}")
    print(f"  - Soglia ottimale: {best_thresh:.3f} (accuracy: {best_acc:.3f})")
    print(f"  - Accuracy con soglia 0.5: {np.mean(correct_predictions):.3f}")
    
    return {
        'optimal_threshold': best_thresh,
        'optimal_accuracy': best_acc,
        'current_accuracy': np.mean(correct_predictions),
        'avg_probability': np.mean(all_probabilities)
    }

def main():
    """Esempio completo di visualizzazione"""
    
    # Carica il modello
    from test_model_accuracy import ModelTester
    tester = ModelTester('final_direct_mst_model.pth')
    
    print("🎨 Inizio visualizzazione esempi di MST predetti...")
    
    # Visualizza esempi di alberi
    examples = visualize_predicted_mst_examples(tester, 'mst_test_dataset.pkl', num_examples=6)
    
    # Analizza pattern delle predizioni
    patterns = analyze_prediction_patterns(tester, 'mst_test_dataset.pkl')
    
    print(f"\n✅ Visualizzazione completata!")
    print(f"📁 File salvati:")
    print(f"  - predicted_mst_examples.png")
    print(f"  - prediction_patterns_analysis.png")

if __name__ == "__main__":
    main()




'''
def main():
    """Test completo del modello"""
    
    # Inizializza tester
    tester = ModelTester('final_direct_mst_model.pth')
    
    # Test su dataset di training
    training_results = tester.test_training_dataset('mst_dataset.pkl')
    
    # Test su dataset di generalizzazione
    generalization_results = tester.test_generalization_dataset('mst_test_dataset.pkl')
    
    # Stampa summary
    tester.print_results_summary(training_results, generalization_results)
    
    # Salva risultati
    tester.save_results(training_results, generalization_results)
    
    print("\n🎉 Valutazione completata!")


if __name__ == "__main__":
    main()

'''