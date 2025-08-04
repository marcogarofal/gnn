import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mst_direct_predictor import MSTDirectPredictor, MSTPredictor
import time
import os
from datetime import datetime

def load_trained_model(model_path='final_direct_mst_model.pth'):
    """Carica il modello addestrato"""

    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Ricrea il modello con gli stessi parametri
    model = MSTDirectPredictor(
        node_feature_dim=checkpoint['node_feature_dim'],
        edge_feature_dim=checkpoint['edge_feature_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )

    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])

    # Crea predictor
    predictor = MSTPredictor(model)

    return predictor


def create_test_graph(graph_type='erdos_renyi', n_nodes=30):
    """Crea un grafo di test con pesi"""

    if graph_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(n_nodes, 0.2)
    elif graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(n_nodes, 3)
    elif graph_type == 'grid':
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif graph_type == 'complete':
        G = nx.complete_graph(n_nodes)
    else:
        G = nx.random_geometric_graph(n_nodes, 0.3)

    # Aggiungi pesi casuali
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 10.0)

    return G


def visualize_comparison(G, mst_predicted, mst_true):
    """Visualizza il confronto tra MST predetto e reale"""

    pos = nx.spring_layout(G, k=2, iterations=50)

    plt.figure(figsize=(15, 5))

    # 1. Grafo originale
    plt.subplot(131)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Mostra pesi
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title(f"Grafo Originale\n{G.number_of_nodes()} nodi, {G.number_of_edges()} edges")
    plt.axis('off')

    # 2. MST Predetto
    plt.subplot(132)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=1)
    if mst_predicted:
        nx.draw_networkx_edges(mst_predicted, pos, edge_color='red', width=3)
        pred_weight = sum(G[u][v]['weight'] for u, v in mst_predicted.edges())
    else:
        pred_weight = 0
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"MST Predetto (GNN)\nPeso: {pred_weight:.2f}")
    plt.axis('off')

    # 3. MST Reale
    plt.subplot(133)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=1)
    nx.draw_networkx_edges(mst_true, pos, edge_color='green', width=3)
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)

    true_weight = sum(G[u][v]['weight'] for u, v in mst_true.edges())
    plt.title(f"MST Ottimale (Kruskal)\nPeso: {true_weight:.2f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Mostra edges diversi
    if mst_predicted:
        pred_edges = set(mst_predicted.edges())
        true_edges = set(mst_true.edges())

        # Considera anche edges invertiti
        pred_edges_undirected = pred_edges | {(v, u) for u, v in pred_edges}
        true_edges_undirected = true_edges | {(v, u) for u, v in true_edges}

        correct = pred_edges_undirected & true_edges_undirected
        wrong = pred_edges_undirected - true_edges_undirected
        missed = true_edges_undirected - pred_edges_undirected

        print(f"\nAnalisi degli edges:")
        print(f"  Corretti: {len(correct)//2}/{len(true_edges)} edges")
        print(f"  Sbagliati (falsi positivi): {len(wrong)//2} edges")
        print(f"  Mancanti (falsi negativi): {len(missed)//2} edges")


def test_on_single_graph(predictor, G):
    """Testa il modello su un singolo grafo"""

    print(f"\nTest su grafo con {G.number_of_nodes()} nodi e {G.number_of_edges()} edges")

    # Predici MST con GNN
    start_time = time.time()
    mst_predicted, info = predictor.predict_mst(G)
    gnn_time = time.time() - start_time

    # Calcola MST reale con Kruskal
    start_time = time.time()
    mst_true = nx.minimum_spanning_tree(G)
    kruskal_time = time.time() - start_time

    # Stampa risultati
    print(f"\nRisultati:")
    if info['success']:
        print(f"  ✓ Predizione riuscita")
        print(f"  Peso predetto: {info['predicted_weight']:.2f}")
        print(f"  Peso ottimale: {info['optimal_weight']:.2f}")
        print(f"  Gap di qualità: {info['quality_gap']:.2%}")
        print(f"  Accuratezza edges: {info['edge_accuracy']:.2%}")
        print(f"  Albero valido: {info['is_valid_tree']}")
    else:
        print(f"  ✗ Predizione fallita: {info['reason']}")

    print(f"\nTempi di esecuzione:")
    print(f"  GNN: {gnn_time*1000:.2f} ms")
    print(f"  Kruskal: {kruskal_time*1000:.2f} ms")
    print(f"  Speedup: {kruskal_time/gnn_time:.2f}x")

    return mst_predicted, mst_true, info


def test_on_multiple_graphs(predictor):
    """Testa il modello su diversi tipi e dimensioni di grafi"""

    test_configs = [
        # (nome, tipo, numero_nodi)
        ("Piccolo Erdos-Renyi", "erdos_renyi", 15),
        ("Medio Erdos-Renyi", "erdos_renyi", 50),
        ("Piccolo Barabasi-Albert", "barabasi_albert", 20),
        ("Grid 5x5", "grid", 25),
        ("Piccolo Completo", "complete", 10),
        ("Random Geometric", "geometric", 30),
    ]

    results = []

    print("="*60)
    print("TEST SU DIVERSI TIPI DI GRAFI")
    print("="*60)

    for name, graph_type, n_nodes in test_configs:
        print(f"\n{name}:")

        # Crea grafo
        G = create_test_graph(graph_type, n_nodes)

        # Testa
        mst_pred, mst_true, info = test_on_single_graph(predictor, G)

        if info['success']:
            results.append({
                'name': name,
                'nodes': n_nodes,
                'edges': G.number_of_edges(),
                'quality_gap': info['quality_gap'],
                'edge_accuracy': info['edge_accuracy']
            })

    # Riassunto
    print("\n" + "="*60)
    print("RIASSUNTO PERFORMANCE")
    print("="*60)
    print(f"{'Grafo':<25} {'Nodi':<8} {'Edges':<8} {'Gap %':<10} {'Acc %':<10}")
    print("-"*60)

    for r in results:
        print(f"{r['name']:<25} {r['nodes']:<8} {r['edges']:<8} "
              f"{r['quality_gap']*100:<10.1f} {r['edge_accuracy']*100:<10.1f}")

    avg_gap = np.mean([r['quality_gap'] for r in results])
    avg_acc = np.mean([r['edge_accuracy'] for r in results])

    print("-"*60)
    print(f"{'MEDIA':<25} {'':<8} {'':<8} {avg_gap*100:<10.1f} {avg_acc*100:<10.1f}")


def interactive_test(predictor):
    """Test interattivo dove l'utente può creare il proprio grafo"""

    print("\n" + "="*60)
    print("TEST INTERATTIVO")
    print("="*60)

    while True:
        print("\nOpzioni:")
        print("1. Crea grafo casuale")
        print("2. Crea grafo personalizzato")
        print("3. Esci")

        choice = input("\nScelta: ")

        if choice == '1':
            n = int(input("Numero di nodi: "))
            p = float(input("Probabilità edges (0-1): "))
            G = nx.erdos_renyi_graph(n, p)

            # Aggiungi pesi
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.uniform(0.1, 10.0)

        elif choice == '2':
            print("\nCrea il tuo grafo:")
            n = int(input("Numero di nodi: "))
            G = nx.Graph()
            G.add_nodes_from(range(n))

            print("Aggiungi edges (formato: 'nodo1 nodo2 peso', 'done' per finire):")
            while True:
                edge_input = input("Edge: ")
                if edge_input.lower() == 'done':
                    break
                try:
                    u, v, w = edge_input.split()
                    G.add_edge(int(u), int(v), weight=float(w))
                    print(f"  Aggiunto edge ({u},{v}) con peso {w}")
                except:
                    print("  Formato non valido. Usa: 'nodo1 nodo2 peso'")

        elif choice == '3':
            break
        else:
            print("Scelta non valida")
            continue

        if G.number_of_edges() == 0:
            print("Il grafo non ha edges!")
            continue

        # Testa e visualizza
        mst_pred, mst_true, info = test_on_single_graph(predictor, G)

        visualize = input("\nVuoi visualizzare il risultato? (s/n): ")
        if visualize.lower() == 's':
            visualize_comparison(G, mst_pred, mst_true)


def main():
    print("TESTING DEL MODELLO MST ADDESTRATO")
    print("="*60)

    # 1. Carica modello
    print("Caricamento modello addestrato...")
    try:
        predictor = load_trained_model('final_direct_mst_model.pth')
        print("✓ Modello caricato con successo!")
    except FileNotFoundError:
        print("✗ File del modello non trovato. Assicurati di aver addestrato il modello prima.")
        print("  Esegui: python mst_direct_predictor.py")
        return
    except Exception as e:
        print(f"✗ Errore nel caricamento: {e}")
        return

    # 2. Test su un esempio semplice
    print("\n" + "="*60)
    print("ESEMPIO SEMPLICE")
    print("="*60)

    # Crea un piccolo grafo di esempio
    G = create_test_graph('erdos_renyi', 20)
    mst_pred, mst_true, info = test_on_single_graph(predictor, G)

    # Visualizza
    print("\nVisualizzazione del risultato...")
    visualize_comparison(G, mst_pred, mst_true)

    # 3. Test su diversi grafi
    test_on_multiple_graphs(predictor)

    # 4. Test interattivo (opzionale)
    interactive = input("\n\nVuoi provare il test interattivo? (s/n): ")
    if interactive.lower() == 's':
        interactive_test(predictor)

    print("\n" + "="*60)
    print("TEST COMPLETATO!")
    print("="*60)


if __name__ == "__main__":
    main()
