import pickle
import numpy as np
import torch
import json
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx

def load_and_explore_dataset(dataset_path='mst_dataset.pkl', num_samples=5):
    """Carica e esplora il dataset MST"""
    
    print("="*80)
    print("ESPLORAZIONE DATASET MST")
    print("="*80)
    
    # 1. Carica il dataset
    print(f"\n1. Caricamento dataset da: {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print("✓ Dataset caricato con successo!")
    except FileNotFoundError:
        print("✗ File non trovato! Assicurati di aver generato il dataset.")
        print("  Esegui prima: python mst_dataset_generator.py")
        return
    except Exception as e:
        print(f"✗ Errore nel caricamento: {e}")
        return
    
    # 2. Informazioni generali
    print(f"\n2. INFORMAZIONI GENERALI")
    print(f"   - Numero totale di grafi: {len(dataset)}")
    
    # Conta alberi totali
    total_trees = sum(len(graph['trees']) for graph in dataset)
    avg_trees = total_trees / len(dataset) if dataset else 0
    print(f"   - Numero totale di alberi: {total_trees}")
    print(f"   - Media alberi per grafo: {avg_trees:.2f}")
    
    # Statistiche sui grafi
    node_counts = [g['num_nodes'] for g in dataset]
    edge_counts = [g['num_edges'] for g in dataset]
    
    print(f"\n   Statistiche sui grafi:")
    print(f"   - Nodi: min={min(node_counts)}, max={max(node_counts)}, media={np.mean(node_counts):.1f}")
    print(f"   - Edges: min={min(edge_counts)}, max={max(edge_counts)}, media={np.mean(edge_counts):.1f}")
    
    # Tipi di grafi
    graph_types = {}
    for g in dataset:
        gtype = g.get('graph_type', 'unknown')
        graph_types[gtype] = graph_types.get(gtype, 0) + 1
    
    print(f"\n   Distribuzione tipi di grafo:")
    for gtype, count in sorted(graph_types.items()):
        print(f"   - {gtype}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # 3. Mostra esempi dettagliati
    print(f"\n3. DETTAGLI DEI PRIMI {num_samples} GRAFI")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        graph_data = dataset[i]
        print(f"\n   GRAFO {i+1}:")
        print(f"   - ID: {graph_data['graph_id']}")
        print(f"   - Tipo: {graph_data.get('graph_type', 'N/A')}")
        print(f"   - Nodi: {graph_data['num_nodes']}")
        print(f"   - Edges: {graph_data['num_edges']}")
        print(f"   - Densità: {graph_data['graph_stats']['density']:.3f}")
        
        # Features dei nodi
        node_features = graph_data['node_features']
        print(f"   - Shape node features: {node_features.shape}")
        print(f"   - Esempio features primo nodo: {node_features[0].numpy()[:5]}...")  # Prime 5 features
        
        # Features degli edges
        edge_attr = graph_data['edge_attr']
        print(f"   - Shape edge features: {edge_attr.shape}")
        
        # Info sugli alberi
        print(f"   - Numero di alberi generati: {len(graph_data['trees'])}")
        
        for j, tree in enumerate(graph_data['trees'][:3]):  # Primi 3 alberi
            metrics = tree['metrics']
            print(f"\n     Albero {j+1}:")
            print(f"     - Ottimale: {'Sì' if tree['is_optimal'] else 'No'}")
            print(f"     - Peso totale: {metrics['total_weight']:.2f}")
            print(f"     - Latenza max: {metrics['max_latency']:.2f}")
            print(f"     - Latenza media: {metrics['avg_latency']:.2f}")
            print(f"     - Profondità: {metrics['tree_depth']}")
            print(f"     - Ratio ottimalità: {tree['optimality_ratio']:.3f}")
            
            # Conta edges nell'albero
            edges_in_tree = sum(tree['edge_labels']) // 2  # Diviso 2 perché non diretto
            print(f"     - Edges nell'albero: {edges_in_tree}")
    
    # 4. Analisi qualità degli alberi
    print(f"\n4. ANALISI QUALITÀ DEGLI ALBERI")
    print("="*80)
    
    all_ratios = []
    optimal_count = 0
    
    for graph in dataset:
        for tree in graph['trees']:
            all_ratios.append(tree['optimality_ratio'])
            if tree['is_optimal']:
                optimal_count += 1
    
    print(f"   - Alberi ottimali: {optimal_count}/{total_trees} ({optimal_count/total_trees*100:.1f}%)")
    print(f"   - Optimality ratio: min={min(all_ratios):.3f}, max={max(all_ratios):.3f}, media={np.mean(all_ratios):.3f}")
    
    # 5. Visualizza un esempio
    print(f"\n5. VISUALIZZAZIONE ESEMPIO")
    print("="*80)
    
    # Prendi il primo grafo piccolo per visualizzazione
    small_graph = None
    for g in dataset:
        if 10 <= g['num_nodes'] <= 20:
            small_graph = g
            break
    
    if small_graph:
        print(f"   Visualizzo grafo con {small_graph['num_nodes']} nodi")
        
        # Ricostruisci il grafo NetworkX
        G = nx.Graph()
        edge_index = small_graph['edge_index']
        edge_attr = small_graph['edge_attr']
        
        # Aggiungi nodi
        for i in range(small_graph['num_nodes']):
            G.add_node(i)
        
        # Aggiungi edges (solo una direzione)
        for i in range(0, edge_index.shape[1], 2):
            u, v = edge_index[:, i].numpy()
            weight = edge_attr[i, 0].item()  # Primo attributo è il peso
            G.add_edge(u, v, weight=weight)
        
        # Trova MST ottimale nel dataset
        optimal_tree_labels = None
        for tree in small_graph['trees']:
            if tree['is_optimal']:
                optimal_tree_labels = tree['edge_labels']
                break
        
        # Visualizza
        plt.figure(figsize=(15, 5))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 1. Grafo completo
        plt.subplot(131)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title(f"Grafo Originale\n{G.number_of_nodes()} nodi, {G.number_of_edges()} edges")
        plt.axis('off')
        
        # 2. MST dal dataset
        if optimal_tree_labels:
            plt.subplot(132)
            nx.draw_networkx_edges(G, pos, alpha=0.1)
            
            # Disegna solo edges nell'MST
            mst_edges = []
            for i in range(0, edge_index.shape[1], 2):
                if optimal_tree_labels[i] == 1:
                    u, v = edge_index[:, i].numpy()
                    mst_edges.append((u, v))
            
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges, 
                                 edge_color='green', width=3)
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=300)
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Calcola peso
            mst_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
            plt.title(f"MST dal Dataset\nPeso: {mst_weight:.2f}")
            plt.axis('off')
        
        # 3. Distribuzione pesi
        plt.subplot(133)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        plt.hist(weights, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Peso')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione Pesi degli Edges')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_example.png', dpi=150)
        plt.show()
        
        print("   ✓ Visualizzazione salvata come 'dataset_example.png'")
    else:
        print("   Nessun grafo piccolo trovato per la visualizzazione")
    
    # 6. Salva statistiche in JSON
    stats_file = 'dataset_statistics.json'
    stats = {
        'total_graphs': len(dataset),
        'total_trees': total_trees,
        'avg_trees_per_graph': avg_trees,
        'node_stats': {
            'min': int(min(node_counts)),
            'max': int(max(node_counts)),
            'mean': float(np.mean(node_counts))
        },
        'edge_stats': {
            'min': int(min(edge_counts)),
            'max': int(max(edge_counts)),
            'mean': float(np.mean(edge_counts))
        },
        'graph_types': graph_types,
        'optimal_trees_ratio': optimal_count / total_trees if total_trees > 0 else 0
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n6. EXPORT")
    print(f"   ✓ Statistiche salvate in: {stats_file}")
    
    return dataset


def analyze_single_graph(dataset, graph_idx=0):
    """Analizza in dettaglio un singolo grafo"""
    
    if graph_idx >= len(dataset):
        print(f"Errore: indice {graph_idx} fuori range (max: {len(dataset)-1})")
        return
    
    graph = dataset[graph_idx]
    
    print(f"\nANALISI DETTAGLIATA GRAFO {graph_idx}")
    print("="*80)
    
    # Informazioni base
    print(f"ID: {graph['graph_id']}")
    print(f"Tipo: {graph.get('graph_type', 'N/A')}")
    print(f"Nodi: {graph['num_nodes']}")
    print(f"Edges: {graph['num_edges']}")
    
    # Analisi features
    print(f"\nFEATURES DEI NODI:")
    node_features = graph['node_features'].numpy()
    print(f"Shape: {node_features.shape}")
    print(f"Range valori: [{node_features.min():.3f}, {node_features.max():.3f}]")
    print(f"Features medie: {node_features.mean(axis=0)[:5]}...")  # Prime 5
    
    print(f"\nFEATURES DEGLI EDGES:")
    edge_features = graph['edge_attr'].numpy()
    print(f"Shape: {edge_features.shape}")
    print(f"Pesi (prima colonna): min={edge_features[:, 0].min():.3f}, max={edge_features[:, 0].max():.3f}")
    
    # Confronto alberi
    print(f"\nCONFRONTO ALBERI ({len(graph['trees'])} totali):")
    print(f"{'Albero':<10} {'Ottimale':<10} {'Peso':<12} {'Latenza Max':<12} {'Edges':<8}")
    print("-"*60)
    
    for i, tree in enumerate(graph['trees']):
        metrics = tree['metrics']
        n_edges = sum(tree['edge_labels']) // 2
        optimal = 'Sì' if tree['is_optimal'] else 'No'
        
        print(f"{i+1:<10} {optimal:<10} {metrics['total_weight']:<12.2f} "
              f"{metrics['max_latency']:<12.2f} {n_edges:<8}")


def main():
    """Funzione principale"""
    
    import sys
    
    # Parametri da linea di comando
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'mst_dataset.pkl'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Esplora dataset
    dataset = load_and_explore_dataset(dataset_path, num_samples)
    
    if dataset:
        # Analisi interattiva
        while True:
            print("\n" + "="*80)
            print("OPZIONI:")
            print("1. Analizza un grafo specifico")
            print("2. Cerca grafi per caratteristiche")
            print("3. Esporta subset del dataset")
            print("4. Esci")
            
            choice = input("\nScelta: ")
            
            if choice == '1':
                try:
                    idx = int(input(f"Indice grafo (0-{len(dataset)-1}): "))
                    analyze_single_graph(dataset, idx)
                except:
                    print("Indice non valido")
                    
            elif choice == '2':
                print("\nCerca per:")
                print("1. Numero di nodi")
                print("2. Tipo di grafo")
                print("3. Numero di alberi")
                
                search_choice = input("Scelta: ")
                
                if search_choice == '1':
                    try:
                        min_nodes = int(input("Minimo nodi: "))
                        max_nodes = int(input("Massimo nodi: "))
                        
                        found = []
                        for i, g in enumerate(dataset):
                            if min_nodes <= g['num_nodes'] <= max_nodes:
                                found.append(i)
                        
                        print(f"\nTrovati {len(found)} grafi:")
                        for idx in found[:10]:  # Mostra primi 10
                            g = dataset[idx]
                            print(f"  - Grafo {idx}: {g['num_nodes']} nodi, tipo {g.get('graph_type', 'N/A')}")
                    except:
                        print("Input non valido")
                        
                elif search_choice == '2':
                    gtype = input("Tipo di grafo: ")
                    found = []
                    for i, g in enumerate(dataset):
                        if g.get('graph_type', '').lower() == gtype.lower():
                            found.append(i)
                    
                    print(f"\nTrovati {len(found)} grafi di tipo '{gtype}'")
                    for idx in found[:10]:
                        g = dataset[idx]
                        print(f"  - Grafo {idx}: {g['num_nodes']} nodi")
                        
            elif choice == '3':
                try:
                    n = int(input("Quanti grafi esportare: "))
                    subset = dataset[:n]
                    
                    filename = f'mst_dataset_subset_{n}.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(subset, f)
                    
                    print(f"✓ Subset salvato in: {filename}")
                except:
                    print("Errore nell'export")
                    
            elif choice == '4':
                break
            else:
                print("Scelta non valida")


if __name__ == "__main__":
    main()