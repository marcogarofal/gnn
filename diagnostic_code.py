import torch
import numpy as np
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

def diagnose_model_issues(tester, dataset_path):
    """Diagnosi approfondita dei problemi del modello"""
    
    print("🔍 DIAGNOSI APPROFONDITA DEI PROBLEMI")
    print("="*60)
    
    # Carica un campione di dati
    from test_model_accuracy import MSTDirectDataset
    dataset = MSTDirectDataset(dataset_path)
    sample_loader = DataLoader(dataset.processed_data[:10], batch_size=1, shuffle=False)
    
    edge_counts = []
    probability_distributions = []
    quality_gaps = []
    
    with torch.no_grad():
        for i, batch in enumerate(sample_loader):
            batch = batch.to(tester.device)
            
            # Forward pass
            edge_logits = tester.model(batch.x, batch.edge_index, batch.edge_attr)
            edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
            
            # Analisi dettagliata
            n_nodes = batch.num_nodes
            expected_edges = n_nodes - 1
            
            print(f"\n📊 GRAFO {i+1} ({n_nodes} nodi, {expected_edges} edges attesi):")
            
            # 1. Distribuzione delle probabilità
            print(f"  Probabilità edges:")
            print(f"    Min: {edge_probs.min():.3f}, Max: {edge_probs.max():.3f}")
            print(f"    Media: {edge_probs.mean():.3f}, Std: {edge_probs.std():.3f}")
            
            # 2. Edges predetti con diverse soglie
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            print("  Edges predetti per soglia:")
            for thresh in thresholds:
                pred_edges = (edge_probs > thresh).sum() / 2  # Diviso 2 per grafo non diretto
                print(f"    {thresh}: {pred_edges:.0f} edges")
            
            # 3. Soglia ottimale per questo grafo
            best_thresh = None
            min_diff = float('inf')
            for thresh in np.arange(0.1, 0.9, 0.01):
                pred_edges = (edge_probs > thresh).sum() / 2
                diff = abs(pred_edges - expected_edges)
                if diff < min_diff:
                    min_diff = diff
                    best_thresh = thresh
            
            pred_edges_optimal = (edge_probs > best_thresh).sum() / 2
            print(f"  Soglia ottimale: {best_thresh:.2f} → {pred_edges_optimal:.0f} edges (diff: {min_diff:.0f})")
            
            # 4. Calcolo Quality Gap corretto
            predictions_05 = (edge_probs > 0.5).astype(float)
            predictions_opt = (edge_probs > best_thresh).astype(float)
            
            edge_weights = batch.edge_weights.cpu().numpy()
            true_labels = batch.mst_labels.cpu().numpy()
            
            # Peso MST vero
            true_weight = (true_labels * edge_weights).sum() / 2
            
            # Peso predetto (soglia 0.5)
            pred_weight_05 = (predictions_05 * edge_weights).sum() / 2
            
            # Peso predetto (soglia ottimale)
            pred_weight_opt = (predictions_opt * edge_weights).sum() / 2
            
            gap_05 = (pred_weight_05 - true_weight) / true_weight if true_weight > 0 else 0
            gap_opt = (pred_weight_opt - true_weight) / true_weight if true_weight > 0 else 0
            
            print(f"  Pesi:")
            print(f"    MST vero: {true_weight:.2f}")
            print(f"    Predetto (0.5): {pred_weight_05:.2f} → Gap: {gap_05:+.2%}")
            print(f"    Predetto (opt): {pred_weight_opt:.2f} → Gap: {gap_opt:+.2%}")
            
            # 5. Accuracy degli edges
            accuracy_05 = (predictions_05 == true_labels).mean()
            accuracy_opt = (predictions_opt == true_labels).mean()
            
            print(f"  Accuracy:")
            print(f"    Soglia 0.5: {accuracy_05:.3f}")
            print(f"    Soglia opt: {accuracy_opt:.3f}")
            
            # Salva per analisi aggregate
            edge_counts.append({
                'expected': expected_edges,
                'pred_05': (edge_probs > 0.5).sum() / 2,
                'pred_opt': pred_edges_optimal,
                'best_thresh': best_thresh
            })
            
            probability_distributions.append(edge_probs)
            quality_gaps.append({'gap_05': gap_05, 'gap_opt': gap_opt})
    
    # Analisi aggregate
    print(f"\n📈 ANALISI AGGREGATA:")
    print("="*40)
    
    # 1. Distribuzione soglie ottimali
    best_thresholds = [ec['best_thresh'] for ec in edge_counts]
    print(f"Soglie ottimali: media={np.mean(best_thresholds):.3f}, "
          f"std={np.std(best_thresholds):.3f}")
    
    # 2. Errori nel numero di edges
    errors_05 = [abs(ec['pred_05'] - ec['expected']) for ec in edge_counts]
    errors_opt = [abs(ec['pred_opt'] - ec['expected']) for ec in edge_counts]
    
    print(f"Errore medio edges:")
    print(f"  Soglia 0.5: {np.mean(errors_05):.1f} ± {np.std(errors_05):.1f}")
    print(f"  Soglia opt: {np.mean(errors_opt):.1f} ± {np.std(errors_opt):.1f}")
    
    # 3. Quality gaps
    gaps_05 = [qg['gap_05'] for qg in quality_gaps]
    gaps_opt = [qg['gap_opt'] for qg in quality_gaps]
    
    print(f"Quality gap medio:")
    print(f"  Soglia 0.5: {np.mean(gaps_05):+.2%} ± {np.std(gaps_05):.2%}")
    print(f"  Soglia opt: {np.mean(gaps_opt):+.2%} ± {np.std(gaps_opt):.2%}")
    
    # Visualizzazioni
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribuzione probabilità
    all_probs = np.concatenate(probability_distributions)
    axes[0,0].hist(all_probs, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(0.5, color='red', linestyle='--', label='Soglia 0.5')
    axes[0,0].axvline(np.mean(best_thresholds), color='green', linestyle='--', label='Soglia media ottimale')
    axes[0,0].set_xlabel('Probabilità Edge')
    axes[0,0].set_ylabel('Frequenza')
    axes[0,0].set_title('Distribuzione Probabilità Edges')
    axes[0,0].legend()
    
    # Plot 2: Soglie ottimali vs dimensione grafo
    expected_edges = [ec['expected'] for ec in edge_counts]
    axes[0,1].scatter(expected_edges, best_thresholds)
    axes[0,1].set_xlabel('Numero Nodi - 1')
    axes[0,1].set_ylabel('Soglia Ottimale')
    axes[0,1].set_title('Soglia Ottimale vs Dimensione Grafo')
    
    # Plot 3: Errori nel numero di edges
    x = range(len(errors_05))
    axes[1,0].bar([i-0.2 for i in x], errors_05, width=0.4, label='Soglia 0.5', alpha=0.7)
    axes[1,0].bar([i+0.2 for i in x], errors_opt, width=0.4, label='Soglia Ottimale', alpha=0.7)
    axes[1,0].set_xlabel('Grafo')
    axes[1,0].set_ylabel('Errore Assoluto Edges')
    axes[1,0].set_title('Errori Predizione Numero Edges')
    axes[1,0].legend()
    
    # Plot 4: Quality gaps
    axes[1,1].bar([i-0.2 for i in x], [g*100 for g in gaps_05], width=0.4, label='Soglia 0.5', alpha=0.7)
    axes[1,1].bar([i+0.2 for i in x], [g*100 for g in gaps_opt], width=0.4, label='Soglia Ottimale', alpha=0.7)
    axes[1,1].set_xlabel('Grafo')
    axes[1,1].set_ylabel('Quality Gap (%)')
    axes[1,1].set_title('Quality Gap per Grafo')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('model_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_optimal_threshold': np.mean(best_thresholds),
        'avg_error_05': np.mean(errors_05),
        'avg_error_opt': np.mean(errors_opt),
        'avg_gap_05': np.mean(gaps_05),
        'avg_gap_opt': np.mean(gaps_opt)
    }

# Utilizzo
if __name__ == "__main__":
    from test_model_accuracy import ModelTester
    
    tester = ModelTester('final_direct_mst_model.pth')
    results = diagnose_model_issues(tester, 'mst_dataset.pkl')
    
    print(f"\n🎯 RACCOMANDAZIONI:")
    print("="*40)
    
    if results['avg_optimal_threshold'] < 0.4:
        print("⚠️  Soglia ottimale troppo bassa → Il modello è troppo conservativo")
        print("   → Riduci il peso della penalità entropia nella loss")
        
    elif results['avg_optimal_threshold'] > 0.6:
        print("⚠️  Soglia ottimale troppo alta → Il modello predice troppi edges")
        print("   → Aumenta il peso della penalità sul numero di edges")
        
    if results['avg_error_05'] > 5:
        print("🔴 Errore nel numero di edges troppo alto")
        print("   → Il modello non ha imparato il vincolo n-1")
        print("   → Considera di aumentare count_penalty nella loss")
        
    if abs(results['avg_gap_05']) > 0.1:
        print("🔴 Quality Gap significativo")
        print("   → Il modello non sceglie gli edges con peso minimo")
        print("   → Aumenta il peso della componente weight_importance")
    
    print(f"\n💡 Prova a rilanciare il training con soglia dinamica: {results['avg_optimal_threshold']:.3f}")