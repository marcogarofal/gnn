import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mst_direct_predictor import MSTDirectPredictor, MSTPredictor
import time
import os
from datetime import datetime
import random

def load_model():
    """Carica il modello addestrato"""
    checkpoint = torch.load('final_direct_mst_model.pth', map_location='cpu')
    model = MSTDirectPredictor(
        node_feature_dim=checkpoint['node_feature_dim'],
        edge_feature_dim=checkpoint['edge_feature_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return MSTPredictor(model)


def generate_random_graph():
    """Genera un grafo completamente casuale ogni volta"""

    # Parametri casuali
    n_nodes = random.randint(10, 100)

    # Tipo di grafo casuale
    graph_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz',
                   'random_geometric', 'powerlaw_tree', 'grid', 'complete']
    graph_type = random.choice(graph_types)

    print(f"\nGenerando {graph_type} con {n_nodes} nodi...")

    try:
        if graph_type == 'erdos_renyi':
            p = random.uniform(0.05, 0.3)
            G = nx.erdos_renyi_graph(n_nodes, p)

        elif graph_type == 'barabasi_albert':
            m = random.randint(1, min(5, n_nodes-1))
            G = nx.barabasi_albert_graph(n_nodes, m)

        elif graph_type == 'watts_strogatz':
            k = random.randint(4, min(10, n_nodes-1))
            p = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n_nodes, k, p)

        elif graph_type == 'random_geometric':
            radius = random.uniform(0.1, 0.4)
            G = nx.random_geometric_graph(n_nodes, radius)

        elif graph_type == 'powerlaw_tree':
            G = nx.powerlaw_tree(n_nodes)

        elif graph_type == 'grid':
            side = int(np.sqrt(n_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)

        else:  # complete
            if n_nodes > 50:  # Limita per grafi completi
                n_nodes = random.randint(10, 50)
            G = nx.complete_graph(n_nodes)

    except:
        # Fallback se qualcosa va storto
        G = nx.erdos_renyi_graph(n_nodes, 0.2)

    # Assicura che sia connesso
    if not nx.is_connected(G):
        # Prendi la componente più grande
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    # Aggiungi pesi casuali con distribuzione casuale
    weight_distributions = ['uniform', 'exponential', 'normal', 'power_law']
    weight_dist = random.choice(weight_distributions)

    for u, v in G.edges():
        if weight_dist == 'uniform':
            weight = random.uniform(0.1, 20.0)
        elif weight_dist == 'exponential':
            weight = np.random.exponential(3.0) + 0.1
        elif weight_dist == 'normal':
            weight = max(0.1, np.random.normal(5.0, 2.0))
        else:  # power_law
            weight = np.random.pareto(2.0) + 0.1

        G[u][v]['weight'] = min(weight, 100.0)  # Cap massimo

    return G, graph_type, weight_dist


def test_single_random_graph(predictor, verbose=True):
    """Testa su un singolo grafo casuale"""

    # Genera grafo casuale
    G, graph_type, weight_dist = generate_random_graph()

    if verbose:
        print(f"  Tipo: {graph_type}")
        print(f"  Distribuzione pesi: {weight_dist}")
        print(f"  Nodi: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Test GNN
    start_time = time.time()
    mst_pred, info = predictor.predict_mst(G)
    gnn_time = time.time() - start_time

    # Test Kruskal
    start_time = time.time()
    mst_true = nx.minimum_spanning_tree(G)
    kruskal_time = time.time() - start_time

    # Calcola pesi
    if info['success'] and mst_pred:
        gnn_weight = sum(G[u][v]['weight'] for u, v in mst_pred.edges())
    else:
        gnn_weight = float('inf')

    kruskal_weight = sum(G[u][v]['weight'] for u, v in mst_true.edges())

    # Risultati
    result = {
        'graph_type': graph_type,
        'weight_dist': weight_dist,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'success': info['success'],
        'gnn_weight': gnn_weight,
        'kruskal_weight': kruskal_weight,
        'quality_gap': (gnn_weight - kruskal_weight) / kruskal_weight if kruskal_weight > 0 else 1.0,
        'edge_accuracy': info.get('edge_accuracy', 0.0),
        'gnn_time': gnn_time,
        'kruskal_time': kruskal_time,
        'speedup': kruskal_time / gnn_time if gnn_time > 0 else 0
    }

    if verbose:
        print(f"\n  PESI DEGLI MST:")
        print(f"  Kruskal (ottimale): {kruskal_weight:.2f}")
        print(f"  GNN (predetto):     {gnn_weight:.2f}")
        print(f"  Differenza:         {gnn_weight - kruskal_weight:.2f}")
        print(f"  Gap:                {result['quality_gap']*100:.1f}%")
        print(f"  Accuratezza edges:  {result['edge_accuracy']*100:.1f}%")
        print(f"  Tempo GNN:          {gnn_time*1000:.1f} ms")
        print(f"  Tempo Kruskal:      {kruskal_time*1000:.1f} ms")

    return result, G, mst_pred, mst_true


def visualize_with_weights(G, mst_pred, mst_true, save=True):
    """Visualizza confronto con i pesi ben evidenziati"""

    pos = nx.spring_layout(G, k=2, iterations=50)
    plt.figure(figsize=(18, 6))

    # Calcola pesi
    if mst_pred:
        pred_weight = sum(G[u][v]['weight'] for u, v in mst_pred.edges())
    else:
        pred_weight = float('inf')
    true_weight = sum(G[u][v]['weight'] for u, v in mst_true.edges())

    # 1. Grafo originale
    plt.subplot(131)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Grafo Originale\n{G.number_of_nodes()} nodi, {G.number_of_edges()} edges")
    plt.axis('off')

    # 2. MST Kruskal (ottimale)
    plt.subplot(132)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    nx.draw_networkx_edges(mst_true, pos, edge_color='green', width=3)
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=200)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Box con peso totale
    plt.text(0.5, -0.1, f"PESO TOTALE: {true_weight:.2f}",
             transform=plt.gca().transAxes,
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
             fontsize=12, fontweight='bold')

    plt.title(f"MST Kruskal (Ottimale)")
    plt.axis('off')

    # 3. MST GNN
    plt.subplot(133)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    if mst_pred:
        nx.draw_networkx_edges(mst_pred, pos, edge_color='red', width=3)
        color = 'lightcoral'
        gap_text = f"Gap: {((pred_weight/true_weight - 1) * 100):.1f}%"
    else:
        color = 'gray'
        gap_text = "FALLITO"
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=200)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Box con peso totale
    plt.text(0.5, -0.1, f"PESO TOTALE: {pred_weight:.2f}\n{gap_text}",
             transform=plt.gca().transAxes,
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
             fontsize=12, fontweight='bold')

    plt.title(f"MST GNN (Predetto)")
    plt.axis('off')

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mst_weights_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confronto salvato in: {filename}")

    plt.show()


def continuous_random_test(predictor, num_tests=100, show_plots=False):
    """Esegue test continui su grafi sempre diversi"""

    print(f"TESTING CONTINUO SU {num_tests} GRAFI CASUALI")
    print("="*60)

    results = []

    # Statistiche per tipo
    stats_by_type = {}
    stats_by_size = {'small': [], 'medium': [], 'large': []}

    # Traccia i migliori e peggiori risultati
    best_result = None
    worst_result = None
    best_graph = None
    worst_graph = None

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        print("-" * 40)

        try:
            result, G, mst_pred, mst_true = test_single_random_graph(predictor)
            results.append(result)

            # Aggiorna migliore e peggiore
            if result['success']:
                if best_result is None or result['quality_gap'] < best_result['quality_gap']:
                    best_result = result
                    best_graph = (G, mst_pred, mst_true)

                if worst_result is None or result['quality_gap'] > worst_result['quality_gap']:
                    worst_result = result
                    worst_graph = (G, mst_pred, mst_true)

            # Mostra plot per alcuni test se richiesto
            if show_plots and i < 5 and result['success']:
                visualize_with_weights(G, mst_pred, mst_true)

            # Aggrega per tipo
            gtype = result['graph_type']
            if gtype not in stats_by_type:
                stats_by_type[gtype] = []
            stats_by_type[gtype].append(result)

            # Aggrega per dimensione
            if result['n_nodes'] < 30:
                stats_by_size['small'].append(result)
            elif result['n_nodes'] < 70:
                stats_by_size['medium'].append(result)
            else:
                stats_by_size['large'].append(result)

            # Ogni 10 test, mostra statistiche parziali
            if (i + 1) % 10 == 0:
                print(f"\n--- Statistiche parziali dopo {i+1} test ---")
                successful = [r for r in results if r['success']]
                if successful:
                    avg_gap = np.mean([r['quality_gap'] for r in successful])
                    avg_kruskal_weight = np.mean([r['kruskal_weight'] for r in successful])
                    avg_gnn_weight = np.mean([r['gnn_weight'] for r in successful])
                    print(f"Peso medio Kruskal: {avg_kruskal_weight:.2f}")
                    print(f"Peso medio GNN: {avg_gnn_weight:.2f}")
                    print(f"Gap medio: {avg_gap*100:.2f}%")

        except Exception as e:
            print(f"  Errore nel test: {e}")
            continue

    # Statistiche finali
    print("\n" + "="*60)
    print("STATISTICHE FINALI")
    print("="*60)

    successful = [r for r in results if r['success']]
    print(f"\nTest completati con successo: {len(successful)}/{len(results)}")

    if successful:
        # Statistiche globali
        print("\nSTATISTICHE GLOBALI:")
        print(f"  Peso medio Kruskal: {np.mean([r['kruskal_weight'] for r in successful]):.2f}")
        print(f"  Peso medio GNN: {np.mean([r['gnn_weight'] for r in successful]):.2f}")
        print(f"  Gap medio: {np.mean([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap mediano: {np.median([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap min: {min([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap max: {max([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Accuratezza media: {np.mean([r['edge_accuracy'] for r in successful])*100:.2f}%")

        # Mostra migliore e peggiore risultato
        if best_result:
            print(f"\nMIGLIOR RISULTATO:")
            print(f"  Tipo: {best_result['graph_type']}, {best_result['n_nodes']} nodi")
            print(f"  Peso Kruskal: {best_result['kruskal_weight']:.2f}")
            print(f"  Peso GNN: {best_result['gnn_weight']:.2f}")
            print(f"  Gap: {best_result['quality_gap']*100:.2f}%")

            # Visualizza il migliore
            if best_graph:
                print("\n  Visualizzazione del miglior risultato...")
                visualize_with_weights(*best_graph)

        if worst_result:
            print(f"\nPEGGIOR RISULTATO:")
            print(f"  Tipo: {worst_result['graph_type']}, {worst_result['n_nodes']} nodi")
            print(f"  Peso Kruskal: {worst_result['kruskal_weight']:.2f}")
            print(f"  Peso GNN: {worst_result['gnn_weight']:.2f}")
            print(f"  Gap: {worst_result['quality_gap']*100:.2f}%")

            # Visualizza il peggiore
            if worst_graph:
                print("\n  Visualizzazione del peggior risultato...")
                visualize_with_weights(*worst_graph)

        # Per tipo di grafo
        print("\nPER TIPO DI GRAFO:")
        for gtype, type_results in stats_by_type.items():
            type_successful = [r for r in type_results if r['success']]
            if type_successful:
                print(f"\n{gtype} ({len(type_successful)} test):")
                print(f"  Peso medio Kruskal: {np.mean([r['kruskal_weight'] for r in type_successful]):.2f}")
                print(f"  Peso medio GNN: {np.mean([r['gnn_weight'] for r in type_successful]):.2f}")
                print(f"  Gap medio: {np.mean([r['quality_gap'] for r in type_successful])*100:.2f}%")
                print(f"  Accuratezza: {np.mean([r['edge_accuracy'] for r in type_successful])*100:.2f}%")

    return results


def visualize_random_test_results(results):
    """Visualizza statistiche dei test casuali"""

    successful = [r for r in results if r['success']]
    if not successful:
        print("Nessun test completato con successo!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Distribuzione del gap
    ax = axes[0, 0]
    gaps = [r['quality_gap']*100 for r in successful]
    ax.hist(gaps, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(gaps), color='red', linestyle='--', label=f'Media: {np.mean(gaps):.1f}%')
    ax.set_xlabel('Gap di Qualità (%)')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione del Gap di Qualità')
    ax.legend()

    # 2. Gap vs dimensione grafo
    ax = axes[0, 1]
    nodes = [r['n_nodes'] for r in successful]
    ax.scatter(nodes, gaps, alpha=0.5)
    ax.set_xlabel('Numero di Nodi')
    ax.set_ylabel('Gap di Qualità (%)')
    ax.set_title('Gap vs Dimensione del Grafo')

    # Linea di tendenza
    z = np.polyfit(nodes, gaps, 1)
    p = np.poly1d(z)
    ax.plot(sorted(nodes), p(sorted(nodes)), "r--", alpha=0.8)

    # 3. Performance per tipo di grafo
    ax = axes[1, 0]
    graph_types = {}
    for r in successful:
        gtype = r['graph_type']
        if gtype not in graph_types:
            graph_types[gtype] = []
        graph_types[gtype].append(r['quality_gap']*100)

    types = list(graph_types.keys())
    means = [np.mean(graph_types[t]) for t in types]
    ax.bar(range(len(types)), means)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.set_ylabel('Gap Medio (%)')
    ax.set_title('Performance per Tipo di Grafo')

    # 4. Speedup vs dimensione
    ax = axes[1, 1]
    speedups = [r['speedup'] for r in successful]
    ax.scatter(nodes, speedups, alpha=0.5, color='green')
    ax.set_xlabel('Numero di Nodi')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup GNN vs Kruskal')
    ax.axhline(y=1, color='red', linestyle='--', label='Nessun speedup')
    ax.legend()

    plt.tight_layout()

    # Salva
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_test_statistics_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Statistiche salvate in: {filename}")
    plt.close()


def main():
    print("TEST CONTINUO SU GRAFI CASUALI")
    print("="*60)

    # Carica modello
    print("Caricamento modello...")
    try:
        predictor = load_model()
        print("✓ Modello caricato!")
    except Exception as e:
        print(f"✗ Errore: {e}")
        return

    # Menu
    while True:
        print("\nOpzioni:")
        print("1. Test rapido (10 grafi casuali)")
        print("2. Test esteso (100 grafi casuali)")
        print("3. Test personalizzato (scegli numero)")
        print("4. Test infinito (premi Ctrl+C per fermare)")
        print("5. Esci")

        choice = input("\nScelta: ")

        if choice == '1':
            results = continuous_random_test(predictor, 10)
            visualize_random_test_results(results)

        elif choice == '2':
            results = continuous_random_test(predictor, 100)
            visualize_random_test_results(results)

        elif choice == '3':
            n = int(input("Numero di test: "))
            results = continuous_random_test(predictor, n)
            visualize_random_test_results(results)

        elif choice == '4':
            print("\nTest infinito - Premi Ctrl+C per fermare")
            print("="*60)
            i = 0
            results = []
            try:
                while True:
                    i += 1
                    print(f"\nTest {i}")
                    result, _, _, _ = test_single_random_graph(predictor)
                    results.append(result)

                    if i % 50 == 0:
                        print(f"\n--- Completati {i} test ---")

            except KeyboardInterrupt:
                print(f"\n\nInterrotto dopo {i} test")
                if results:
                    visualize_random_test_results(results)

        elif choice == '5':
            break
        else:
            print("Scelta non valida")

    print("\nTest completato!")


if __name__ == "__main__":
    main()
