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


def generate_custom_graph(graph_type='erdos_renyi', n_nodes=30, **kwargs):
    """Genera un grafo personalizzato con parametri specifici"""

    print(f"\nGenerando {graph_type} con {n_nodes} nodi...")

    try:
        if graph_type == 'erdos_renyi':
            p = kwargs.get('p', random.uniform(0.05, 0.3))
            G = nx.erdos_renyi_graph(n_nodes, p)
            print(f"  Probabilità edges: {p:.2f}")

        elif graph_type == 'barabasi_albert':
            m = kwargs.get('m', random.randint(1, min(5, n_nodes-1)))
            G = nx.barabasi_albert_graph(n_nodes, m)
            print(f"  Edges per nuovo nodo: {m}")

        elif graph_type == 'watts_strogatz':
            k = kwargs.get('k', random.randint(4, min(10, n_nodes-1)))
            p = kwargs.get('p', random.uniform(0.1, 0.5))
            G = nx.watts_strogatz_graph(n_nodes, k, p)
            print(f"  k={k}, probabilità rewiring={p:.2f}")

        elif graph_type == 'random_geometric':
            radius = kwargs.get('radius', random.uniform(0.1, 0.4))
            G = nx.random_geometric_graph(n_nodes, radius)
            print(f"  Raggio: {radius:.2f}")

        elif graph_type == 'powerlaw_tree':
            gamma = kwargs.get('gamma', 3)
            G = nx.powerlaw_tree(n_nodes, gamma)
            print(f"  Gamma: {gamma}")

        elif graph_type == 'grid':
            rows = kwargs.get('rows', int(np.sqrt(n_nodes)))
            cols = kwargs.get('cols', int(np.sqrt(n_nodes)))
            G = nx.grid_2d_graph(rows, cols)
            G = nx.convert_node_labels_to_integers(G)
            print(f"  Griglia: {rows}x{cols}")

        elif graph_type == 'complete':
            G = nx.complete_graph(n_nodes)
            print(f"  Grafo completo")

        elif graph_type == 'cycle':
            G = nx.cycle_graph(n_nodes)
            print(f"  Grafo ciclico")

        elif graph_type == 'star':
            G = nx.star_graph(n_nodes - 1)  # n_nodes include il centro
            print(f"  Grafo a stella")

        elif graph_type == 'wheel':
            G = nx.wheel_graph(n_nodes)
            print(f"  Grafo a ruota")

        elif graph_type == 'ladder':
            G = nx.ladder_graph(n_nodes // 2)
            print(f"  Grafo a scala")

        elif graph_type == 'circular_ladder':
            G = nx.circular_ladder_graph(n_nodes // 2)
            print(f"  Scala circolare")

        elif graph_type == 'hypercube':
            n = int(np.log2(n_nodes))
            G = nx.hypercube_graph(n)
            print(f"  Ipercubo di dimensione {n}")

        elif graph_type == 'tree':
            r = kwargs.get('r', 2)  # branching factor
            h = kwargs.get('h', int(np.log(n_nodes) / np.log(r)))  # height
            G = nx.balanced_tree(r, h)
            print(f"  Albero bilanciato: r={r}, h={h}")

        else:
            print(f"  Tipo sconosciuto '{graph_type}', uso erdos_renyi")
            G = nx.erdos_renyi_graph(n_nodes, 0.2)

    except Exception as e:
        print(f"  Errore nella generazione: {e}")
        print(f"  Fallback a erdos_renyi")
        G = nx.erdos_renyi_graph(n_nodes, 0.2)

    # Assicura che sia connesso
    if not nx.is_connected(G):
        # Prendi la componente più grande
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
        print(f"  Ridotto a componente connessa: {G.number_of_nodes()} nodi")

    # Aggiungi pesi
    weight_dist = kwargs.get('weight_dist', random.choice(['uniform', 'exponential', 'normal', 'power_law']))

    for u, v in G.edges():
        if weight_dist == 'uniform':
            low = kwargs.get('weight_min', 0.1)
            high = kwargs.get('weight_max', 20.0)
            weight = random.uniform(low, high)
        elif weight_dist == 'exponential':
            scale = kwargs.get('weight_scale', 3.0)
            weight = np.random.exponential(scale) + 0.1
        elif weight_dist == 'normal':
            mean = kwargs.get('weight_mean', 5.0)
            std = kwargs.get('weight_std', 2.0)
            weight = max(0.1, np.random.normal(mean, std))
        else:  # power_law
            alpha = kwargs.get('weight_alpha', 2.0)
            weight = np.random.pareto(alpha) + 0.1

        G[u][v]['weight'] = min(weight, 100.0)  # Cap massimo

    print(f"  Distribuzione pesi: {weight_dist}")
    print(f"  Grafo finale: {G.number_of_nodes()} nodi, {G.number_of_edges()} edges")

    return G, graph_type, weight_dist


def interactive_custom_test(predictor):
    """Test interattivo con parametri personalizzati"""

    print("\n" + "="*60)
    print("TEST PERSONALIZZATO")
    print("="*60)

    # Mostra tipi disponibili
    graph_types = [
        'erdos_renyi', 'barabasi_albert', 'watts_strogatz',
        'random_geometric', 'powerlaw_tree', 'grid',
        'complete', 'cycle', 'star', 'wheel',
        'ladder', 'circular_ladder', 'hypercube', 'tree'
    ]

    print("\nTipi di grafo disponibili:")
    for i, gtype in enumerate(graph_types, 1):
        print(f"{i:2d}. {gtype}")

    # Scegli tipo
    try:
        choice = int(input("\nScegli il tipo (numero): "))
        graph_type = graph_types[choice - 1]
    except:
        graph_type = 'erdos_renyi'
        print(f"Scelta non valida, uso {graph_type}")

    # Numero di nodi
    try:
        n_nodes = int(input("Numero di nodi (default 30): ") or "30")
    except:
        n_nodes = 30

    # Parametri specifici per tipo
    kwargs = {}

    if graph_type == 'erdos_renyi':
        try:
            p = float(input("Probabilità edges (0-1, default random): ") or "-1")
            if 0 <= p <= 1:
                kwargs['p'] = p
        except:
            pass

    elif graph_type == 'barabasi_albert':
        try:
            m = int(input(f"Edges per nuovo nodo (1-{min(5, n_nodes-1)}, default random): ") or "-1")
            if 1 <= m < n_nodes:
                kwargs['m'] = m
        except:
            pass

    elif graph_type == 'watts_strogatz':
        try:
            k = int(input("k (vicini iniziali, default random): ") or "-1")
            if k > 0:
                kwargs['k'] = k
            p = float(input("Probabilità rewiring (0-1, default random): ") or "-1")
            if 0 <= p <= 1:
                kwargs['p'] = p
        except:
            pass

    elif graph_type == 'random_geometric':
        try:
            radius = float(input("Raggio (0-1, default random): ") or "-1")
            if 0 < radius <= 1:
                kwargs['radius'] = radius
        except:
            pass

    # Distribuzione pesi
    print("\nDistribuzione pesi:")
    print("1. uniform")
    print("2. exponential")
    print("3. normal")
    print("4. power_law")

    try:
        weight_choice = int(input("Scegli (1-4, default random): ") or "0")
        weight_dists = ['uniform', 'exponential', 'normal', 'power_law']
        if 1 <= weight_choice <= 4:
            kwargs['weight_dist'] = weight_dists[weight_choice - 1]
    except:
        pass

    # Genera e testa
    G, gtype, wdist = generate_custom_graph(graph_type, n_nodes, **kwargs)

    # Test
    print("\n" + "-"*40)
    result, G, mst_pred, mst_true = test_single_random_graph(predictor, verbose=True)

    # Visualizza
    visualize = input("\nVuoi visualizzare il risultato? (s/n): ")
    if visualize.lower() == 's':
        visualize_with_weights(G, mst_pred, mst_true)

    return result


def generate_random_graph():
    """Genera un grafo completamente casuale ogni volta"""

    # Parametri casuali
    n_nodes = random.randint(10, 100)

    # Tipo di grafo casuale
    graph_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz',
                   'random_geometric', 'powerlaw_tree', 'grid', 'complete']
    graph_type = random.choice(graph_types)

    # Genera usando la funzione custom con parametri random
    return generate_custom_graph(graph_type, n_nodes)


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

    # CONTATORI VITTORIE
    victories = {
        'kruskal_wins': 0,
        'gnn_wins': 0,
        'ties': 0,
        'gnn_failures': 0
    }

    # Casi speciali dove GNN trova soluzione migliore
    gnn_better_cases = []

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        print("-" * 40)

        try:
            result, G, mst_pred, mst_true = test_single_random_graph(predictor)
            results.append(result)

            # CONTA VITTORIE
            if not result['success']:
                victories['gnn_failures'] += 1
            else:
                # Confronta i pesi con tolleranza per errori di arrotondamento
                weight_diff = result['gnn_weight'] - result['kruskal_weight']
                tolerance = 0.0001  # Tolleranza per float

                if abs(weight_diff) < tolerance:
                    victories['ties'] += 1
                    print("  🤝 PAREGGIO! Stessi pesi")
                elif weight_diff < 0:
                    victories['gnn_wins'] += 1
                    print(f"  🏆 GNN VINCE! Risparmio: {-weight_diff:.2f}")
                    # Salva caso interessante
                    gnn_better_cases.append({
                        'result': result,
                        'graph': G,
                        'mst_pred': mst_pred,
                        'mst_true': mst_true,
                        'savings': -weight_diff
                    })
                else:
                    victories['kruskal_wins'] += 1

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

                # Mostra conteggio vittorie parziale
                print(f"\nCONTEGGIO VITTORIE:")
                print(f"  Kruskal: {victories['kruskal_wins']}")
                print(f"  GNN: {victories['gnn_wins']}")
                print(f"  Pareggi: {victories['ties']}")
                print(f"  Fallimenti GNN: {victories['gnn_failures']}")

        except Exception as e:
            print(f"  Errore nel test: {e}")
            continue

    # Statistiche finali
    print("\n" + "="*60)
    print("STATISTICHE FINALI")
    print("="*60)

    successful = [r for r in results if r['success']]
    print(f"\nTest completati con successo: {len(successful)}/{len(results)}")

    # RISULTATI VITTORIE
    print("\n" + "="*60)
    print("🏁 CONTEGGIO FINALE VITTORIE 🏁")
    print("="*60)
    total_tests = len(results)
    total_successful = victories['kruskal_wins'] + victories['gnn_wins'] + victories['ties']

    print(f"\nSu {total_tests} test totali:")
    print(f"  ✓ Test riusciti: {total_successful}")
    print(f"  ✗ Fallimenti GNN: {victories['gnn_failures']}")

    if total_successful > 0:
        print(f"\nSu {total_successful} test riusciti:")
        print(f"  🥇 KRUSKAL vince: {victories['kruskal_wins']} ({victories['kruskal_wins']/total_successful*100:.1f}%)")
        print(f"  🥈 GNN vince: {victories['gnn_wins']} ({victories['gnn_wins']/total_successful*100:.1f}%)")
        print(f"  🤝 Pareggi: {victories['ties']} ({victories['ties']/total_successful*100:.1f}%)")

    # Mostra casi dove GNN ha vinto
    if gnn_better_cases:
        print(f"\n⭐ CASI INTERESSANTI: GNN ha trovato soluzioni migliori in {len(gnn_better_cases)} casi!")
        print("Dettagli dei primi 3 casi:")
        for i, case in enumerate(gnn_better_cases[:3]):
            print(f"\n  Caso {i+1}:")
            print(f"    Tipo grafo: {case['result']['graph_type']}")
            print(f"    Nodi: {case['result']['n_nodes']}")
            print(f"    Peso Kruskal: {case['result']['kruskal_weight']:.2f}")
            print(f"    Peso GNN: {case['result']['gnn_weight']:.2f}")
            print(f"    Risparmio: {case['savings']:.2f}")

        # Visualizza il caso più eclatante
        if gnn_better_cases:
            best_gnn_case = max(gnn_better_cases, key=lambda x: x['savings'])
            print(f"\n🌟 MIGLIOR CASO GNN (risparmio massimo: {best_gnn_case['savings']:.2f}):")
            visualize_with_weights(
                best_gnn_case['graph'],
                best_gnn_case['mst_pred'],
                best_gnn_case['mst_true']
            )

    if successful:
        # Statistiche globali
        print("\n" + "="*60)
        print("STATISTICHE DETTAGLIATE")
        print("="*60)
        print("\nSTATISTICHE GLOBALI:")
        print(f"  Peso medio Kruskal: {np.mean([r['kruskal_weight'] for r in successful]):.2f}")
        print(f"  Peso medio GNN: {np.mean([r['gnn_weight'] for r in successful]):.2f}")
        print(f"  Gap medio: {np.mean([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap mediano: {np.median([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap min: {min([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Gap max: {max([r['quality_gap'] for r in successful])*100:.2f}%")
        print(f"  Accuratezza media: {np.mean([r['edge_accuracy'] for r in successful])*100:.2f}%")

        # Mostra migliore e peggiore risultato
        if best_result and best_result['quality_gap'] >= 0:
            print(f"\nMIGLIOR RISULTATO GNN (gap minimo positivo):")
            print(f"  Tipo: {best_result['graph_type']}, {best_result['n_nodes']} nodi")
            print(f"  Peso Kruskal: {best_result['kruskal_weight']:.2f}")
            print(f"  Peso GNN: {best_result['gnn_weight']:.2f}")
            print(f"  Gap: {best_result['quality_gap']*100:.2f}%")

        if worst_result:
            print(f"\nPEGGIOR RISULTATO GNN:")
            print(f"  Tipo: {worst_result['graph_type']}, {worst_result['n_nodes']} nodi")
            print(f"  Peso Kruskal: {worst_result['kruskal_weight']:.2f}")
            print(f"  Peso GNN: {worst_result['gnn_weight']:.2f}")
            print(f"  Gap: {worst_result['quality_gap']*100:.2f}%")

        # Per tipo di grafo con conteggio vittorie
        print("\nVITTORIE PER TIPO DI GRAFO:")
        for gtype, type_results in stats_by_type.items():
            type_successful = [r for r in type_results if r['success']]
            if type_successful:
                type_kruskal_wins = sum(1 for r in type_successful if r['quality_gap'] > 0.0001)
                type_gnn_wins = sum(1 for r in type_successful if r['quality_gap'] < -0.0001)
                type_ties = len(type_successful) - type_kruskal_wins - type_gnn_wins

                print(f"\n{gtype} ({len(type_successful)} test):")
                print(f"  Kruskal vince: {type_kruskal_wins}")
                print(f"  GNN vince: {type_gnn_wins}")
                print(f"  Pareggi: {type_ties}")
                print(f"  Gap medio: {np.mean([r['quality_gap'] for r in type_successful])*100:.2f}%")

    return results, victories, gnn_better_cases


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
        print("5. Test con parametri custom")
        print("6. Test su tipo specifico")
        print("7. Esci")

        choice = input("\nScelta: ")

        if choice == '1':
            results, victories, gnn_better_cases = continuous_random_test(predictor, 10)
            visualize_random_test_results(results)

        elif choice == '2':
            results, victories, gnn_better_cases = continuous_random_test(predictor, 100)
            visualize_random_test_results(results)

        elif choice == '3':
            n = int(input("Numero di test: "))
            results, victories, gnn_better_cases = continuous_random_test(predictor, n)
            visualize_random_test_results(results)

        elif choice == '4':
            print("\nTest infinito - Premi Ctrl+C per fermare")
            print("="*60)
            i = 0
            results = []
            victories = {'kruskal_wins': 0, 'gnn_wins': 0, 'ties': 0, 'gnn_failures': 0}
            try:
                while True:
                    i += 1
                    print(f"\nTest {i}")
                    result, G, mst_pred, mst_true = test_single_random_graph(predictor)
                    results.append(result)

                    # Aggiorna conteggio vittorie
                    if not result['success']:
                        victories['gnn_failures'] += 1
                    else:
                        weight_diff = result['gnn_weight'] - result['kruskal_weight']
                        if abs(weight_diff) < 0.0001:
                            victories['ties'] += 1
                        elif weight_diff < 0:
                            victories['gnn_wins'] += 1
                        else:
                            victories['kruskal_wins'] += 1

                    if i % 50 == 0:
                        print(f"\n--- Completati {i} test ---")
                        print(f"Kruskal: {victories['kruskal_wins']}, GNN: {victories['gnn_wins']}, Pareggi: {victories['ties']}")

            except KeyboardInterrupt:
                print(f"\n\nInterrotto dopo {i} test")
                print(f"\nRISULTATI FINALI:")
                print(f"  Kruskal vince: {victories['kruskal_wins']}")
                print(f"  GNN vince: {victories['gnn_wins']}")
                print(f"  Pareggi: {victories['ties']}")
                print(f"  Fallimenti: {victories['gnn_failures']}")
                if results:
                    visualize_random_test_results(results)

        elif choice == '5':
            # Test con parametri personalizzati
            interactive_custom_test(predictor)

        elif choice == '6':
            # Test su tipo specifico
            print("\nTest su tipo specifico di grafo")

            # Mostra tipi
            graph_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz',
                          'random_geometric', 'grid', 'complete', 'tree']
            for i, gt in enumerate(graph_types, 1):
                print(f"{i}. {gt}")

            try:
                type_choice = int(input("\nScegli tipo (1-7): "))
                graph_type = graph_types[type_choice - 1]
            except:
                graph_type = 'erdos_renyi'

            # Parametri per il test
            try:
                n_tests = int(input("Numero di test (default 20): ") or "20")
                n_nodes = int(input("Numero di nodi fisso (default varia 10-100): ") or "0")
            except:
                n_tests = 20
                n_nodes = 0

            print(f"\nTest su {n_tests} grafi {graph_type}")
            results = []
            victories = {'kruskal_wins': 0, 'gnn_wins': 0, 'ties': 0, 'gnn_failures': 0}

            for i in range(n_tests):
                print(f"\nTest {i+1}/{n_tests}")

                # Usa numero di nodi fisso o random
                nodes = n_nodes if n_nodes > 0 else random.randint(10, 100)

                # Genera grafo del tipo scelto
                G, _, wdist = generate_custom_graph(graph_type, nodes)

                # Test
                result, G, mst_pred, mst_true = test_single_random_graph(predictor, verbose=True)
                results.append(result)

                # Conteggio vittorie
                if not result['success']:
                    victories['gnn_failures'] += 1
                else:
                    weight_diff = result['gnn_weight'] - result['kruskal_weight']
                    if abs(weight_diff) < 0.0001:
                        victories['ties'] += 1
                    elif weight_diff < 0:
                        victories['gnn_wins'] += 1
                    else:
                        victories['kruskal_wins'] += 1

            # Risultati finali
            print(f"\n{'='*60}")
            print(f"RISULTATI PER {graph_type.upper()}")
            print(f"{'='*60}")
            print(f"Kruskal vince: {victories['kruskal_wins']}")
            print(f"GNN vince: {victories['gnn_wins']}")
            print(f"Pareggi: {victories['ties']}")
            print(f"Fallimenti: {victories['gnn_failures']}")

            if results:
                successful = [r for r in results if r['success']]
                if successful:
                    print(f"\nGap medio: {np.mean([r['quality_gap'] for r in successful])*100:.2f}%")
                    print(f"Accuratezza media: {np.mean([r['edge_accuracy'] for r in successful])*100:.2f}%")

        elif choice == '7':
            break
        else:
            print("Scelta non valida")

    print("\nTest completato!")


if __name__ == "__main__":
    main()
