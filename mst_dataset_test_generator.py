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

class MSTTestDatasetGenerator:
    """Generatore di dataset di test per MST con configurazioni specifiche per testare generalizzazione"""

    def __init__(self, seed: int = 123):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_graph_with_density(self, n_nodes: int, graph_type: str, density_mode: str) -> nx.Graph:
        """Genera un grafo con densità specifica (standard, alta, bassa)"""

        if graph_type == 'erdos_renyi':
            if density_mode == 'standard':
                p = max(0.15, 2 * np.log(n_nodes) / n_nodes)
            elif density_mode == 'high':
                p = max(0.25, 3 * np.log(n_nodes) / n_nodes)  # Più connesso
            else:  # low
                p = max(0.08, 1.5 * np.log(n_nodes) / n_nodes)  # Meno connesso
            G = nx.erdos_renyi_graph(n_nodes, p)

        elif graph_type == 'barabasi_albert':
            if density_mode == 'standard':
                m = min(3, n_nodes - 1)
            elif density_mode == 'high':
                m = min(5, n_nodes - 1)  # Più edges per nodo
            else:  # low
                m = min(2, n_nodes - 1)  # Meno edges per nodo
            G = nx.barabasi_albert_graph(n_nodes, m)

        elif graph_type == 'watts_strogatz':
            if density_mode == 'standard':
                k = min(4, n_nodes - 1)
            elif density_mode == 'high':
                k = min(6, n_nodes - 1)  # Più vicini iniziali
            else:  # low
                k = min(2, n_nodes - 1)  # Meno vicini iniziali
            if k % 2 == 1:  # k deve essere pari per watts_strogatz
                k += 1
            G = nx.watts_strogatz_graph(n_nodes, k, 0.3)

        elif graph_type == 'random_geometric':
            if density_mode == 'standard':
                radius = np.sqrt(2.0 / n_nodes)
            elif density_mode == 'high':
                radius = np.sqrt(3.0 / n_nodes)  # Raggio maggiore = più connessioni
            else:  # low
                radius = np.sqrt(1.2 / n_nodes)  # Raggio minore = meno connessioni
            G = nx.random_geometric_graph(n_nodes, radius)

        else:  # grid
            side = int(np.sqrt(n_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
            
            # Per grid, modifichiamo aggiungendo/rimuovendo edges
            if density_mode == 'high':
                # Aggiungi alcune diagonali random
                nodes = list(G.nodes())
                for _ in range(n_nodes // 4):
                    u, v = random.sample(nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
            elif density_mode == 'low':
                # Rimuovi alcuni edges mantenendo connettività
                edges_to_remove = list(G.edges())
                random.shuffle(edges_to_remove)
                for u, v in edges_to_remove[:len(edges_to_remove)//3]:
                    G.remove_edge(u, v)
                    if not nx.is_connected(G):
                        G.add_edge(u, v)  # Ripristina se disconnette

        # Assicura connettività
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
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
            pos = nx.get_node_attributes(G, 'pos')
            for u, v in G.edges():
                if pos:
                    dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    G[u][v]['weight'] = dist + np.random.normal(0, 0.1)
                else:
                    G[u][v]['weight'] = np.random.uniform(0.1, 10.0)
        else:
            weight_pattern = np.random.choice(['uniform', 'exponential', 'clustered'])

            if weight_pattern == 'uniform':
                for u, v in G.edges():
                    G[u][v]['weight'] = np.random.uniform(0.1, 10.0)

            elif weight_pattern == 'exponential':
                for u, v in G.edges():
                    G[u][v]['weight'] = max(0.1, np.random.exponential(2.0) + 0.1)

            else:  # clustered
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

    def generate_test_spanning_trees(self, G: nx.Graph, max_trees: int = 7) -> List[nx.Graph]:
        """Genera alberi ottimizzati per test dataset"""
        trees = []

        # Verifica che tutti i pesi siano positivi
        for u, v in G.edges():
            if G[u][v]['weight'] <= 0:
                G[u][v]['weight'] = 0.1

        # 1. MST ottimale (OBBLIGATORIO)
        mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
        for u, v in mst.edges():
            if mst[u][v]['weight'] <= 0:
                mst[u][v]['weight'] = G[u][v]['weight'] = 0.1
        trees.append(mst)

        # 2. MST con Prim (potrebbe essere diverso)
        mst_prim = nx.minimum_spanning_tree(G, algorithm='prim')
        if not self._is_duplicate_tree(mst_prim, trees):
            trees.append(mst_prim)

        # 3. 2-3 spanning trees sub-ottimali (per robustezza)
        edges = list(G.edges(data=True))
        for i in range(min(3, max_trees - len(trees))):
            perturbed_edges = []
            perturbation_strength = 0.1 * (i + 1)

            for u, v, data in edges:
                scale = max(0.01, perturbation_strength * abs(data['weight']))
                noise = np.random.normal(0, scale)
                perturbed_weight = max(0.01, data['weight'] + noise)
                perturbed_edges.append((u, v, perturbed_weight, data['weight']))

            perturbed_edges.sort(key=lambda x: x[2])

            T = nx.Graph()
            T.add_nodes_from(G.nodes())

            for u, v, _, real_weight in perturbed_edges:
                if not nx.has_path(T, u, v):
                    T.add_edge(u, v, weight=max(0.1, abs(real_weight)))
                    if T.number_of_edges() == G.number_of_nodes() - 1:
                        break

            if nx.is_tree(T) and not self._is_duplicate_tree(T, trees):
                trees.append(T)

        # 4. 1-2 random spanning trees
        for _ in range(min(2, max_trees - len(trees))):
            try:
                random_tree = nx.random_spanning_tree(G)
                for u, v in random_tree.edges():
                    weight = G[u][v]['weight']
                    random_tree[u][v]['weight'] = max(0.1, abs(weight))

                if not self._is_duplicate_tree(random_tree, trees):
                    trees.append(random_tree)
            except:
                continue

        return trees[:max_trees]

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

        # Verifica che tutti i pesi siano positivi
        for u, v in tree.edges():
            if 'weight' not in tree[u][v] or tree[u][v]['weight'] <= 0:
                tree[u][v]['weight'] = 0.1

        # 1. Peso totale
        metrics['total_weight'] = sum(tree[u][v]['weight'] for u, v in tree.edges())

        # 2. Latenza
        if tree.number_of_nodes() > 1:
            distances = nx.single_source_dijkstra_path_length(tree, root, weight='weight')
            metrics['max_latency'] = max(distances.values())
            metrics['avg_latency'] = np.mean(list(distances.values()))
            metrics['latency_std'] = np.std(list(distances.values()))
        else:
            metrics['max_latency'] = 0
            metrics['avg_latency'] = 0
            metrics['latency_std'] = 0

        metrics['num_edges'] = tree.number_of_edges()

        try:
            metrics['tree_depth'] = nx.dag_longest_path_length(tree.to_directed()) + 1
        except:
            metrics['tree_depth'] = max(nx.shortest_path_length(tree, root).values()) if tree.number_of_nodes() > 1 else 0

        # Bilanciamento
        subtree_sizes = []
        for node in tree.nodes():
            if node != root and tree.degree(node) == 1:
                path_to_root = nx.shortest_path_length(tree, node, root)
                subtree_sizes.append(path_to_root)

        metrics['balance_score'] = np.std(subtree_sizes) if subtree_sizes else 0

        return metrics

    def graph_to_features(self, G: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converte grafo in features per GNN"""

        node_features = []

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

        edge_index = []
        edge_attr = []

        for u, v in G.edges():
            edge_index.extend([[u, v], [v, u]])

            weight = G[u][v]['weight']
            edge_feat = [
                weight,
                1.0 / (weight + 0.01),
                np.log(weight + 1),
                weight ** 2,
                np.sqrt(max(0, weight))
            ]

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

    def create_dataset_entry(self, G: nx.Graph, trees: List[nx.Graph], config_info: Dict) -> Dict:
        """Crea entry del dataset per un grafo e i suoi alberi"""

        node_features, edge_index, edge_attr = self.graph_to_features(G)

        tree_data = []
        mst_weight = sum(G[u][v]['weight'] for u, v in trees[0].edges())

        for i, tree in enumerate(trees):
            edge_labels = []
            tree_edges = set(tree.edges()) | set((v, u) for u, v in tree.edges())

            for j in range(0, edge_index.shape[1], 2):
                u = edge_index[0, j].item()
                v = edge_index[1, j].item()
                if (u, v) in tree_edges:
                    edge_labels.extend([1, 1])
                else:
                    edge_labels.extend([0, 0])

            metrics = self.compute_tree_metrics(G, tree)
            is_optimal = (i == 0)
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
            'graph_id': None,
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
            },
            'config_info': config_info  # Informazioni sulla configurazione di test
        }

    def generate_test_dataset(self) -> List[Dict]:
        """Genera il dataset di test completo con le 9 configurazioni specifiche"""
        
        dataset = []
        graph_id = 0

        # Configurazioni di test
        node_sizes = [25, 40, 60]
        density_modes = ['standard', 'high', 'low']
        graph_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'random_geometric', 'grid']

        print("Generazione dataset di test con 9 configurazioni...")
        print("Configurazioni: 3 densità × 3 dimensioni = 9 gruppi, 5 grafi per gruppo = 45 grafi totali")

        for density_mode in density_modes:
            for n_nodes in node_sizes:
                print(f"\nGenerazione gruppo: {density_mode} density, {n_nodes} nodi")
                
                for i in range(5):  # 5 grafi per configurazione
                    # Scegli tipo di grafo random
                    graph_type = np.random.choice(graph_types)
                    
                    # Genera grafo
                    G = self.generate_graph_with_density(n_nodes, graph_type, density_mode)
                    
                    # Genera alberi (meno alberi per test)
                    trees = self.generate_test_spanning_trees(G, max_trees=7)
                    
                    # Info configurazione
                    config_info = {
                        'density_mode': density_mode,
                        'target_nodes': n_nodes,
                        'actual_nodes': G.number_of_nodes(),
                        'graph_type': graph_type,
                        'config_group': f"{density_mode}_{n_nodes}",
                        'graph_in_group': i + 1
                    }
                    
                    # Crea entry dataset
                    entry = self.create_dataset_entry(G, trees, config_info)
                    entry['graph_id'] = graph_id
                    entry['graph_type'] = graph_type
                    
                    dataset.append(entry)
                    graph_id += 1
                    
                    print(f"  Grafo {i+1}/5: {graph_type}, {G.number_of_nodes()} nodi, "
                          f"{G.number_of_edges()} edges, {len(trees)} alberi, "
                          f"densità={nx.density(G):.3f}")

        return dataset

    def save_test_dataset(self, dataset: List[Dict], filepath: str):
        """Salva il dataset di test con statistiche dettagliate"""
        print(f"\nSalvataggio dataset di test in {filepath}...")

        # Salva dataset
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        # Statistiche dettagliate per il test
        stats = {
            'num_graphs': len(dataset),
            'total_trees': sum(len(d['trees']) for d in dataset),
            'avg_trees_per_graph': np.mean([len(d['trees']) for d in dataset]),
            'configurations': {}
        }

        # Raggruppa per configurazione
        for d in dataset:
            config = d['config_info']
            config_key = f"{config['density_mode']}_{config['target_nodes']}"
            
            if config_key not in stats['configurations']:
                stats['configurations'][config_key] = {
                    'count': 0,
                    'densities': [],
                    'node_counts': [],
                    'edge_counts': [],
                    'graph_types': {}
                }
            
            config_stats = stats['configurations'][config_key]
            config_stats['count'] += 1
            config_stats['densities'].append(d['graph_stats']['density'])
            config_stats['node_counts'].append(d['num_nodes'])
            config_stats['edge_counts'].append(d['num_edges'])
            
            graph_type = d['graph_type']
            if graph_type not in config_stats['graph_types']:
                config_stats['graph_types'][graph_type] = 0
            config_stats['graph_types'][graph_type] += 1

        # Calcola medie per configurazione
        for config_key, config_stats in stats['configurations'].items():
            config_stats['avg_density'] = np.mean(config_stats['densities'])
            config_stats['avg_nodes'] = np.mean(config_stats['node_counts'])
            config_stats['avg_edges'] = np.mean(config_stats['edge_counts'])

        # Salva statistiche
        with open(filepath + '_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # Stampa summary
        print(f"\nDataset di test generato!")
        print(f"  - Grafi totali: {stats['num_graphs']}")
        print(f"  - Alberi totali: {stats['total_trees']}")
        print(f"  - Media alberi per grafo: {stats['avg_trees_per_graph']:.1f}")
        print(f"\nConfigurazioni generate:")
        
        for config_key, config_stats in stats['configurations'].items():
            density_mode, nodes = config_key.split('_')
            print(f"  - {density_mode.capitalize()} density, {nodes} nodi: "
                  f"{config_stats['count']} grafi, "
                  f"densità media: {config_stats['avg_density']:.3f}, "
                  f"edges medi: {config_stats['avg_edges']:.1f}")


def main():
    """Genera il dataset di test per valutare generalizzazione"""
    
    # Crea generatore con seed diverso dal training
    generator = MSTTestDatasetGenerator(seed=123)
    
    # Genera dataset
    test_dataset = generator.generate_test_dataset()
    
    # Salva
    generator.save_test_dataset(test_dataset, 'mst_test_dataset')
    
    # Esempi delle configurazioni generate
    print("\n" + "="*80)
    print("ESEMPI DELLE CONFIGURAZIONI GENERATE:")
    print("="*80)
    
    configs_shown = set()
    for entry in test_dataset:
        config_key = entry['config_info']['config_group']
        if config_key not in configs_shown and len(configs_shown) < 6:  # Mostra 6 esempi
            configs_shown.add(config_key)
            config = entry['config_info']
            stats = entry['graph_stats']
            
            print(f"\nConfigurazione: {config['density_mode']} density, {config['target_nodes']} nodi")
            print(f"  Grafo esempio {entry['graph_id']}: {entry['graph_type']}")
            print(f"  Nodi: {entry['num_nodes']}, Edges: {entry['num_edges']}")
            print(f"  Densità: {stats['density']:.3f}, Clustering: {stats['avg_clustering']:.3f}")
            print(f"  Alberi generati: {len(entry['trees'])}")
            
            # MST weight
            mst_weight = entry['trees'][0]['metrics']['total_weight']
            print(f"  Peso MST: {mst_weight:.2f}")


if __name__ == "__main__":
    main()