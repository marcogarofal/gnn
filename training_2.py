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
from typing import Tuple, Dict, Optional

class MSTDirectPredictor(nn.Module):
    """GNN che predice direttamente l'MST dato solo il grafo"""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int,
                 hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        # Opzioni per ridurre overfitting
        self.use_batch_norm = True
        self.dropout_rate = 0.2  # Aumentabile se necessario

        # Node encoder con batch norm
        if self.use_batch_norm:
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
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


class MSTDirectLoss(nn.Module):
    """Loss function ottimizzata per predizione diretta MST"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, edge_logits: torch.Tensor, mst_labels: torch.Tensor,
                edge_weights: torch.Tensor, num_nodes: int):
        """
        Loss che combina:
        1. Classificazione corretta degli edges
        2. Preferenza per edges con peso basso
        3. Vincolo sul numero di edges (n-1)
        4. Penalità per componenti disconnesse (indirettamente)
        """

        # 1. Binary cross entropy pesata
        edge_loss = self.bce(edge_logits, mst_labels)

        # Peso maggiore agli edges con peso basso (più probabili nell'MST)
        weight_importance = 1.0 / (edge_weights + 0.1)
        weight_importance = weight_importance / weight_importance.mean()
        edge_loss = (edge_loss * weight_importance).mean()

        # 2. Penalità sul numero di edges
        predicted_probs = torch.sigmoid(edge_logits)
        predicted_edges = predicted_probs.sum() / 2  # Diviso 2 perché non diretto
        expected_edges = num_nodes - 1
        count_penalty = torch.abs(predicted_edges - expected_edges) / expected_edges

        # 3. Entropia per incoraggiare decisioni nette (0 o 1)
        entropy = -(predicted_probs * torch.log(predicted_probs + 1e-8) +
                   (1 - predicted_probs) * torch.log(1 - predicted_probs + 1e-8))
        entropy_penalty = entropy.mean()

        # 4. Consistenza: edges simmetrici devono avere stessa predizione
        # (già gestito dal fatto che duplichiamo gli edges)

        total_loss = edge_loss + 0.5 * count_penalty + 0.1 * entropy_penalty

        metrics = {
            'edge_loss': edge_loss.item(),
            'count_penalty': count_penalty.item(),
            'entropy': entropy_penalty.item(),
            'predicted_edges': (predicted_probs > 0.5).sum().item() / 2
        }

        return total_loss, metrics


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
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            [self.processed_data[i] for i in val_idx],
            batch_size=batch_size
        )
        test_loader = DataLoader(
            [self.processed_data[i] for i in test_idx],
            batch_size=batch_size
        )

        return train_loader, val_loader, test_loader


class MSTDirectTrainer:
    """Trainer per predizione diretta MST"""

    def __init__(self, model: MSTDirectPredictor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.01, epochs=100, steps_per_epoch=100
        )
        self.loss_fn = MSTDirectLoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        # Contatori per overfitting
        self.overfitting_counter = 0
        self.max_overfitting_warnings = 5

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct_edges = 0
        total_edges = 0

        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            edge_logits = self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            # Loss
            if hasattr(batch, 'batch'):
                # Media del numero di nodi per grafo nel batch
                num_nodes = batch.num_nodes
            else:
                num_nodes = batch.x.shape[0]

            loss, metrics = self.loss_fn(
                edge_logits, batch.mst_labels,
                batch.edge_weights, num_nodes
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(edge_logits) > 0.5).float()
            correct_edges += (predictions == batch.mst_labels).sum().item()
            total_edges += edge_logits.shape[0]

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_edges / total_edges
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        tree_quality_gaps = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                edge_logits = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                num_nodes = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.shape[0]
                loss, _ = self.loss_fn(
                    edge_logits, batch.mst_labels,
                    batch.edge_weights, num_nodes
                )

                total_loss += loss.item()

                predictions = (torch.sigmoid(edge_logits) > 0.5).float()
                all_predictions.append(predictions)
                all_labels.append(batch.mst_labels)

                # Valuta qualità dell'albero predetto
                if not hasattr(batch, 'batch'):  # Singolo grafo
                    pred_weight = (predictions * batch.edge_weights).sum() / 2
                    true_weight = (batch.mst_labels * batch.edge_weights).sum() / 2
                    if true_weight > 0:
                        gap = (pred_weight - true_weight) / true_weight
                        tree_quality_gaps.append(gap.item())

        # Calcola metriche
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        accuracy = (all_predictions == all_labels).float().mean()

        # F1 score per edges nell'MST
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        avg_gap = np.mean(tree_quality_gaps) if tree_quality_gaps else 0

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'avg_quality_gap': avg_gap
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 15):
        best_f1 = 0
        best_val_loss = float('inf')
        patience_counter = 0

        # Per rilevare overfitting
        min_delta = 0.001  # Miglioramento minimo richiesto

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            # Log
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"Avg Quality Gap: {val_metrics['avg_quality_gap']:.2%}")

            # Rileva overfitting
            if epoch > 10:  # Aspetta qualche epoca prima di controllare
                train_val_gap = abs(train_metrics['loss'] - val_metrics['loss'])
                if train_val_gap > 0.5:
                    print(f"⚠️  Warning: Large gap between train/val loss ({train_val_gap:.3f})")

                # Se il training loss continua a scendere ma validation peggiora
                if (len(self.history['train_loss']) > 5 and
                    self.history['train_loss'][-1] < self.history['train_loss'][-5] and
                    self.history['val_loss'][-1] > self.history['val_loss'][-5]):
                    print("⚠️  Warning: Possible overfitting detected")

            # Early stopping basato su F1 e val loss
            improved = False

            # Criterio 1: F1 score migliora
            if val_metrics['f1'] > best_f1 + min_delta:
                best_f1 = val_metrics['f1']
                improved = True

            # Criterio 2: Val loss migliora (per evitare overfitting)
            if val_metrics['loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['loss']
                improved = True

            if improved:
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_direct_mst_model.pth')
                print("✓ Model improved and saved")
            else:
                patience_counter += 1
                print(f"✗ No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"\n{'='*50}")
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Best F1: {best_f1:.4f}, Best Val Loss: {best_val_loss:.4f}")
                print(f"{'='*50}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_direct_mst_model.pth'))
        print("\nLoaded best model from checkpoint")


class MSTPredictor:
    """Predice MST direttamente dal grafo"""

    def __init__(self, model: MSTDirectPredictor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict_mst(self, G: nx.Graph) -> Tuple[nx.Graph, Dict]:
        """
        Predice l'MST dato solo il grafo

        Returns:
            mst: l'albero predetto
            info: metriche e informazioni
        """

        # Genera features
        from mst_dataset_generator import MSTDatasetGenerator
        generator = MSTDatasetGenerator()
        node_features, edge_index, edge_attr = generator.graph_to_features(G)

        # Crea data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        ).to(self.device)

        with torch.no_grad():
            # Predici
            edge_logits = self.model(data.x, data.edge_index, data.edge_attr)
            edge_probs = torch.sigmoid(edge_logits).cpu().numpy()

        # Costruisci MST dalle predizioni
        mst = self._construct_mst_from_predictions(G, edge_probs, edge_index.cpu().numpy())

        # Calcola MST reale per confronto
        true_mst = nx.minimum_spanning_tree(G)

        # Metriche
        if mst:
            pred_weight = sum(G[u][v]['weight'] for u, v in mst.edges())
            true_weight = sum(G[u][v]['weight'] for u, v in true_mst.edges())

            # Edges corretti
            pred_edges = set(mst.edges())
            true_edges = set(true_mst.edges())
            correct_edges = len(pred_edges & true_edges)

            info = {
                'success': True,
                'predicted_weight': pred_weight,
                'optimal_weight': true_weight,
                'quality_gap': (pred_weight - true_weight) / true_weight,
                'edge_accuracy': correct_edges / len(true_edges),
                'is_valid_tree': nx.is_tree(mst),
                'num_predicted_edges': len(mst.edges()),
                'num_true_edges': len(true_mst.edges())
            }
        else:
            info = {'success': False, 'reason': 'Failed to construct valid tree'}

        return mst, info

    def _construct_mst_from_predictions(self, G: nx.Graph, edge_probs: np.ndarray,
                                       edge_index: np.ndarray) -> nx.Graph:
        """Costruisce MST dalle probabilità predette"""

        # Crea lista di edges con probabilità e pesi
        edge_candidates = []

        for i in range(0, edge_index.shape[1], 2):  # Skip reverse edges
            u, v = edge_index[:, i]
            prob = edge_probs[i]
            weight = G[u][v]['weight'] if G.has_edge(u, v) else float('inf')

            # Score combina probabilità e peso
            # Alto score = alta probabilità E basso peso
            score = prob / (weight + 0.1)

            edge_candidates.append((score, prob, u, v, weight))

        # Ordina per score decrescente
        edge_candidates.sort(reverse=True)

        # Costruisci albero usando approccio greedy
        mst = nx.Graph()
        mst.add_nodes_from(G.nodes())

        edges_added = 0
        target_edges = G.number_of_nodes() - 1

        # Prima passa: aggiungi edges con alta probabilità
        for score, prob, u, v, weight in edge_candidates:
            if prob > 0.5 and not nx.has_path(mst, u, v):
                mst.add_edge(u, v, weight=weight)
                edges_added += 1
                if edges_added >= target_edges:
                    break

        # Seconda passa: se non abbiamo abbastanza edges, aggiungi i migliori rimanenti
        if edges_added < target_edges:
            for score, prob, u, v, weight in edge_candidates:
                if not mst.has_edge(u, v) and not nx.has_path(mst, u, v):
                    mst.add_edge(u, v, weight=weight)
                    edges_added += 1
                    if edges_added >= target_edges:
                        break

        # Verifica che sia un albero valido
        if nx.is_tree(mst):
            return mst
        else:
            # Fallback: restituisci il componente connesso più grande
            if nx.number_connected_components(mst) > 1:
                largest_cc = max(nx.connected_components(mst), key=len)
                return mst.subgraph(largest_cc).copy()
            return mst


def test_model():
    """Test del modello su nuovi grafi"""

    # Carica modello
    checkpoint = torch.load('final_direct_mst_model.pth')
    model = MSTDirectPredictor(
        node_feature_dim=checkpoint['node_feature_dim'],
        edge_feature_dim=checkpoint['edge_feature_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    predictor = MSTPredictor(model)

    # Test su diversi grafi
    test_graphs = [
        ('Piccolo Erdos-Renyi', nx.erdos_renyi_graph(15, 0.3)),
        ('Medio Barabasi-Albert', nx.barabasi_albert_graph(30, 3)),
        ('Grid 5x5', nx.grid_2d_graph(5, 5))
    ]

    for name, G in test_graphs:
        # Converti in grafo semplice e aggiungi pesi
        G = nx.Graph(G)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 10.0)

        print(f"\n{name} - {G.number_of_nodes()} nodi, {G.number_of_edges()} edges")

        # Predici
        mst_pred, info = predictor.predict_mst(G)

        if info['success']:
            print(f"  Peso predetto: {info['predicted_weight']:.2f}")
            print(f"  Peso ottimale: {info['optimal_weight']:.2f}")
            print(f"  Gap: {info['quality_gap']:.2%}")
            print(f"  Edge accuracy: {info['edge_accuracy']:.2%}")
            print(f"  Valid tree: {info['is_valid_tree']}")


def main():
    # 1. Carica dataset
    print("Caricamento dataset per predizione diretta MST...")
    dataset = MSTDirectDataset('mst_dataset.pkl')
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    # 2. Crea modello
    print("Creazione modello...")
    sample = dataset.processed_data[0]
    node_feature_dim = sample.x.shape[1]
    edge_feature_dim = sample.edge_attr.shape[1]

    model = MSTDirectPredictor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=128,
        num_layers=6
    )

    print(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")

    # 3. Training
    trainer = MSTDirectTrainer(model)
    trainer.train(train_loader, val_loader, epochs=100, patience=15)

    # 4. Visualizza risultati
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(trainer.history['train_loss'], label='Train')
    plt.plot(trainer.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(trainer.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.subplot(133)
    plt.plot(trainer.history['val_f1'])
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')

    plt.tight_layout()
    plt.savefig('direct_training_history.png')
    plt.show()

    # 5. Test finale
    print("\nTest finale...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Avg Quality Gap: {test_metrics['avg_quality_gap']:.2%}")

    # 6. Salva modello
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_feature_dim': node_feature_dim,
        'edge_feature_dim': edge_feature_dim,
        'hidden_dim': 128,
        'num_layers': 6
    }, 'final_direct_mst_model.pth')

    print("\nTraining completato!")

    # 7. Test su esempi
    test_model()


if __name__ == "__main__":
    main()
