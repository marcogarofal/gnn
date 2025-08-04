import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import json
import pickle
from tqdm import tqdm
from itertools import combinations
import random
from typing import Dict, List, Tuple, Any

class MSTDatasetGenerator:
    """Generatore di dataset per MST con multiple varianti di alberi per grafo"""

    def __init__(self, num_graphs: int = 200, seed: int = 42):
        self.num_graphs = num_graphs
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_graph(self, n_nodes: int, graph_type: str) -> nx.Graph:
        """Genera un grafo connesso con pesi sugli edges"""

        if graph_type == 'erdos_renyi':
            # Assicura connettività con probabilità più alta
            p = max(0.15, 2 * np.log(n_nodes) / n_nodes)
            G = nx.erdos_renyi_graph(n_nodes, p)
        elif graph_type == 'barabasi_albert':
            m = min(3, n_nodes - 1)
            G = nx.barabasi_albert_graph(n_nodes, m)
        elif graph_type == 'watts_strogatz':
            k = min(4, n_nodes - 1)
            G = nx.watts_strogatz_graph(n_nodes, k, 0.3)
        elif graph_type == 'random_geometric':
            radius = np.sqrt(2.0 / n_nodes)
            G = nx.random_geometric_graph(n_nodes, radius)
        else:  # grid
            side = int(np.sqrt(n_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)

        # Assicura connettività
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            # Connetti componenti
            for i in range(1, len(components)):
                u = list(components[0])[0]
                v = list(components[i])[0]
                G.add_edge(u, v)

        # Aggiungi pesi realistici
        self._add_weights(G, graph_type)

        return G

    def _add_weights(self, G: nx.Graph, graph_type: str):
        """Aggiungi pesi agli edges con pattern realistici"""

        if graph_type == 'random_geometric':
            # Pesi basati su distanza euclidea
            pos = nx.get_node_attributes(G, 'pos')
            for u, v in G.edges():
                if pos:
                    dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    G[u][v]['weight'] = dist + np.random.normal(0, 0.1)
                else:
                    G[u][v]['weight'] = np.random.uniform(0.1, 10.0)
        else:
            # Pattern di peso variati
            weight_pattern = np.random.choice(['uniform', 'exponential', 'clustered'])

            if weight_pattern == 'uniform':
                for u, v in G.edges():
                    G[u][v]['weight'] = np.random.uniform(0.1, 10.0)

            elif weight_pattern == 'exponential':
                for u, v in G.edges():
                    G[u][v]['weight'] = max(0.1, np.random.exponential(2.0) + 0.1)

            else:  # clustered
                # Identifica cluster
                try:
                    clusters = list(nx.community.greedy_modularity_communities(G))
                except:
                    clusters = [{n} for n in G.nodes()]

                for u, v in G.edges():
                    same_cluster = any(u in c and v in c for c in clusters)
                    if same_cluster:
                        G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
                    else:
                        G[u][v]['weight'] = np.random.uniform(5.0, 10.0)

    def generate_spanning_trees(self, G: nx.Graph, max_trees: int = 20) -> List[nx.Graph]:
        """Genera multiple varianti di spanning tree per il grafo"""
        trees = []

        # Verifica che tutti i pesi del grafo siano positivi
        for u, v in G.edges():
            if G[u][v]['weight'] <= 0:
                G[u][v]['weight'] = 0.1

        # 1. MST ottimale (Kruskal)
        mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
        # Verifica pesi nell'MST
        for u, v in mst.edges():
            if mst[u][v]['weight'] <= 0:
                mst[u][v]['weight'] = G[u][v]['weight'] = 0.1
        trees.append(mst)

        # 2. MST con Prim da nodi diversi (può dare risultati leggermente diversi)
        for start_node in random.sample(list(G.nodes()), min(3, len(G.nodes()))):
            mst_prim = nx.minimum_spanning_tree(G, algorithm='prim')
            if not self._is_duplicate_tree(mst_prim, trees):
                trees.append(mst_prim)

        # 3. Spanning trees sub-ottimali tramite Kruskal perturbato
        edges = list(G.edges(data=True))
        for i in range(min(10, max_trees - len(trees))):
            # Perturba i pesi
            perturbed_edges = []
            perturbation_strength = 0.1 * (i + 1)  # Aumenta perturbazione

            for u, v, data in edges:
                # Assicura che la scala sia sempre positiva
                scale = max(0.01, perturbation_strength * abs(data['weight']))
                noise = np.random.normal(0, scale)
                perturbed_weight = max(0.01, data['weight'] + noise)
                perturbed_edges.append((u, v, perturbed_weight, data['weight']))

            # Ordina per peso perturbato
            perturbed_edges.sort(key=lambda x: x[2])

            # Costruisci albero
            T = nx.Graph()
            T.add_nodes_from(G.nodes())

            for u, v, _, real_weight in perturbed_edges:
                if not nx.has_path(T, u, v):
                    # Assicura peso positivo
                    T.add_edge(u, v, weight=max(0.1, abs(real_weight)))
                    if T.number_of_edges() == G.number_of_nodes() - 1:
                        break

            if nx.is_tree(T) and not self._is_duplicate_tree(T, trees):
                trees.append(T)

        # 4. Random spanning trees
        for _ in range(min(5, max_trees - len(trees))):
            try:
                random_tree = nx.random_spanning_tree(G)
                # Aggiungi pesi originali
                for u, v in random_tree.edges():
                    weight = G[u][v]['weight']
                    # Assicura che il peso sia positivo
                    random_tree[u][v]['weight'] = max(0.1, abs(weight))

                if not self._is_duplicate_tree(random_tree, trees):
                    trees.append(random_tree)
            except:
                continue

        # 5. Greedy approach: sempre scegli edge più leggero disponibile
        greedy_tree = self._greedy_spanning_tree(G)
        if greedy_tree and not self._is_duplicate_tree(greedy_tree, trees):
            trees.append(greedy_tree)

        return trees[:max_trees]

    def _greedy_spanning_tree(self, G: nx.Graph) -> nx.Graph:
        """Costruisci spanning tree con approccio greedy diverso da Kruskal"""
        T = nx.Graph()
        T.add_nodes_from(G.nodes())

        # Parti da nodo random
        start = random.choice(list(G.nodes()))
        visited = {start}

        while len(visited) < G.number_of_nodes():
            # Trova edge più leggero che connette visited a non-visited
            min_edge = None
            min_weight = float('inf')

            for u in visited:
                for v in G.neighbors(u):
                    if v not in visited and G[u][v]['weight'] < min_weight:
                        min_edge = (u, v)
                        min_weight = G[u][v]['weight']

            if min_edge:
                u, v = min_edge
                weight = G[u][v]['weight']
                T.add_edge(u, v, weight=max(0.1, abs(weight)))
                visited.add(v)
            else:
                break

        return T if nx.is_tree(T) else None

    def _is_duplicate_tree(self, T: nx.Graph, trees: List[nx.Graph]) -> bool:
        """Verifica se un albero è duplicato"""
        T_edges = set(T.edges())
        for tree in trees:
            if T_edges == set(tree.edges()):
                return True
        return False

    def compute_tree_metrics(self, G: nx.Graph, tree: nx.Graph, root: int = 0) -> Dict:
        """Calcola metriche per un albero"""
        metrics = {}

        # Verifica che tutti i pesi siano positivi prima di calcolare le metriche
        for u, v in tree.edges():
            if 'weight' not in tree[u][v] or tree[u][v]['weight'] <= 0:
                tree[u][v]['weight'] = 0.1

        # 1. Peso totale (costo)
        metrics['total_weight'] = sum(tree[u][v]['weight'] for u, v in tree.edges())

        # 2. Latenza (distanza massima dalla radice)
        if tree.number_of_nodes() > 1:
            distances = nx.single_source_dijkstra_path_length(tree, root, weight='weight')
            metrics['max_latency'] = max(distances.values())
            metrics['avg_latency'] = np.mean(list(distances.values()))
            metrics['latency_std'] = np.std(list(distances.values()))
        else:
            metrics['max_latency'] = 0
            metrics['avg_latency'] = 0
            metrics['latency_std'] = 0

        # 3. Numero di edges (sempre n-1 per un albero)
        metrics['num_edges'] = tree.number_of_edges()

        # 4. Profondità dell'albero
        try:
            metrics['tree_depth'] = nx.dag_longest_path_length(tree.to_directed()) + 1
        except:
            metrics['tree_depth'] = max(nx.shortest_path_length(tree, root).values())

        # 5. Bilanciamento (deviazione standard delle dimensioni dei sottoalberi)
        subtree_sizes = []
        for node in tree.nodes():
            if node != root and tree.degree(node) == 1:  # foglia
                path_to_root = nx.shortest_path_length(tree, node, root)
                subtree_sizes.append(path_to_root)

        metrics['balance_score'] = np.std(subtree_sizes) if subtree_sizes else 0

        return metrics

    def graph_to_features(self, G: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converte grafo in features per GNN"""

        # Node features
        node_features = []

        # Calcola betweenness centrality solo se il grafo ha più di 1 nodo
        if G.number_of_nodes() > 1:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        else:
            betweenness = {n: 0.0 for n in G.nodes()}

        for node in G.nodes():
            features = [
                G.degree(node),
                nx.clustering(G, node),
                nx.closeness_centrality(G, node) if G.number_of_nodes() > 1 else 0.0,
                betweenness[node],
                nx.triangles(G, node) if G.number_of_nodes() > 2 else 0,
            ]

            # Statistiche sui pesi degli edges connessi
            weights = [G[node][nbr]['weight'] for nbr in G[node]]
            if weights:
                features.extend([
                    np.mean(weights),
                    np.std(weights),
                    np.min(weights),
                    np.max(weights)
                ])
            else:
                features.extend([0, 0, 0, 0])

            node_features.append(features)

        # Edge index e features
        edge_index = []
        edge_attr = []

        for u, v in G.edges():
            # Aggiungi entrambe le direzioni
            edge_index.extend([[u, v], [v, u]])

            weight = G[u][v]['weight']
            edge_feat = [
                weight,
                1.0 / (weight + 0.01),
                np.log(weight + 1),
                weight ** 2,
                np.sqrt(max(0, weight))  # Gestisce valori negativi
            ]

            # Features relative ai nodi
            edge_feat.extend([
                G.degree(u) * G.degree(v),
                abs(G.degree(u) - G.degree(v)),
                min(G.degree(u), G.degree(v)),
                max(G.degree(u), G.degree(v))
            ])

            edge_attr.extend([edge_feat, edge_feat])

        return (
            torch.tensor(node_features, dtype=torch.float),
            torch.tensor(edge_index, dtype=torch.long).t(),
            torch.tensor(edge_attr, dtype=torch.float)
        )

    def create_dataset_entry(self, G: nx.Graph, trees: List[nx.Graph]) -> Dict:
        """Crea entry del dataset per un grafo e i suoi alberi"""

        # Features del grafo
        node_features, edge_index, edge_attr = self.graph_to_features(G)

        # Info per ogni albero
        tree_data = []

        # Trova MST (primo albero)
        mst_weight = sum(G[u][v]['weight'] for u, v in trees[0].edges())

        for i, tree in enumerate(trees):
            # Labels degli edges (1 se in questo albero, 0 altrimenti)
            edge_labels = []
            tree_edges = set(tree.edges()) | set((v, u) for u, v in tree.edges())

            for j in range(0, edge_index.shape[1], 2):
                u = edge_index[0, j].item()
                v = edge_index[1, j].item()
                if (u, v) in tree_edges:
                    edge_labels.extend([1, 1])
                else:
                    edge_labels.extend([0, 0])

            # Metriche dell'albero
            metrics = self.compute_tree_metrics(G, tree)

            # È l'MST ottimale?
            is_optimal = (i == 0)  # Assumiamo che il primo sia l'MST
            optimality_ratio = metrics['total_weight'] / mst_weight

            tree_info = {
                'edge_labels': edge_labels,
                'metrics': metrics,
                'is_optimal': is_optimal,
                'optimality_ratio': optimality_ratio,
                'tree_index': i
            }

            tree_data.append(tree_info)

        return {
            'graph_id': None,  # Sarà assegnato dopo
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'trees': tree_data,
            'graph_stats': {
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else -1
            }
        }

    def generate_dataset(self) -> List[Dict]:
        """Genera il dataset completo"""
        dataset = []

        print(f"Generazione di {self.num_graphs} grafi con relativi alberi...")

        for i in tqdm(range(self.num_graphs)):
            # Varia dimensione e tipo di grafo (minimo 5 nodi)
            n_nodes = np.random.randint(10, 50)
            graph_type = np.random.choice([
                'erdos_renyi', 'barabasi_albert', 'watts_strogatz',
                'random_geometric', 'grid'
            ])

            # Genera grafo
            G = self.generate_graph(n_nodes, graph_type)

            # Genera alberi
            trees = self.generate_spanning_trees(G, max_trees=15)

            # Crea entry dataset
            entry = self.create_dataset_entry(G, trees)
            entry['graph_id'] = i
            entry['graph_type'] = graph_type

            dataset.append(entry)

            # Log di progresso
            if (i + 1) % 20 == 0:
                print(f"  Generati {i + 1} grafi, ultimo: {n_nodes} nodi, {len(trees)} alberi")

        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Salva il dataset su file"""
        print(f"Salvataggio dataset in {filepath}...")

        # Salva come pickle per preservare tensori
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        # Salva anche statistiche in JSON per analisi
        stats = {
            'num_graphs': len(dataset),
            'total_trees': sum(len(d['trees']) for d in dataset),
            'avg_trees_per_graph': np.mean([len(d['trees']) for d in dataset]),
            'graph_sizes': [d['num_nodes'] for d in dataset],
            'graph_types': {}
        }

        for d in dataset:
            graph_type = d['graph_type']
            if graph_type not in stats['graph_types']:
                stats['graph_types'][graph_type] = 0
            stats['graph_types'][graph_type] += 1

        with open(filepath + '_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Dataset salvato! Statistiche:")
        print(f"  - Numero grafi: {stats['num_graphs']}")
        print(f"  - Totale alberi: {stats['total_trees']}")
        print(f"  - Media alberi per grafo: {stats['avg_trees_per_graph']:.2f}")


def main():
    # Genera dataset
    generator = MSTDatasetGenerator(num_graphs=200)
    dataset = generator.generate_dataset()

    # Salva dataset
    generator.save_dataset(dataset, 'mst_dataset')

    # Esempio di utilizzo
    print("\nEsempio primo grafo nel dataset:")
    first_graph = dataset[0]
    print(f"  - Graph ID: {first_graph['graph_id']}")
    print(f"  - Tipo: {first_graph['graph_type']}")
    print(f"  - Nodi: {first_graph['num_nodes']}")
    print(f"  - Edges: {first_graph['num_edges']}")
    print(f"  - Numero di alberi generati: {len(first_graph['trees'])}")

    print("\n  Metriche degli alberi:")
    for i, tree in enumerate(first_graph['trees'][:5]):  # Primi 5
        metrics = tree['metrics']
        print(f"    Albero {i}: peso={metrics['total_weight']:.2f}, "
              f"max_latency={metrics['max_latency']:.2f}, "
              f"optimal={tree['is_optimal']}")


if __name__ == "__main__":
    main()
