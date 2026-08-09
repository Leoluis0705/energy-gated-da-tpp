#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Analyze selection efficiency with random baseline')
    parser.add_argument('--history-grad', required=True,
                      help='Path to grad selection history CSV')
    parser.add_argument('--history-greed', required=True,
                      help='Path to greed selection history CSV')
    parser.add_argument('--history-active', required=True,
                      help='Path to active learning selection history CSV')
    parser.add_argument('--history-modulus', required=True,
                      help='Path to modulus clustering selection history CSV')
    parser.add_argument('--history-dropout', required=True,  # New argument for Dropout
                      help='Path to dropout selection history CSV')
    parser.add_argument('--true-features', required=True,
                      help='Path to true features file (id_prop.csv format)')
    parser.add_argument('--output-image', default='efficiency_with_random.png',
                      help='Output image filename')
    parser.add_argument('--target-feature', type=float, required=True,
                      help='Target feature value')
    parser.add_argument('--tolerance', type=float, default=0.2,
                      help='Acceptable deviation from target (default: 0.2)')
    parser.add_argument('--random-trials', type=int, default=50,
                      help='Number of random trials for baseline (default: 50)')
    parser.add_argument('--custom-annotation', required=True,
                      help='Custom annotation text to display below main title')
    
    args = parser.parse_args()

    lower_bound = args.target_feature - args.tolerance
    upper_bound = args.target_feature + args.tolerance

    try:
        # Read all history files including the new dropout file
        history_grad = pd.read_csv(args.history_grad, dtype={'id': str})
        history_greed = pd.read_csv(args.history_greed, dtype={'id': str})
        history_active = pd.read_csv(args.history_active, dtype={'id': str})
        history_modulus = pd.read_csv(args.history_modulus, dtype={'id': str})
        history_dropout = pd.read_csv(args.history_dropout, dtype={'id': str})  # New

        true_features = pd.read_csv(args.true_features, 
                                  header=None, 
                                  names=['id', 'true_feature'],
                                  dtype={'id': str})
        
        # Process and mark valid samples
        def process_history(history_df):
            merged = history_df.merge(true_features, on='id', how='left')
            merged['actual_valid'] = ((merged['true_feature'] >= lower_bound) & 
                                    (merged['true_feature'] <= upper_bound)).astype(int)
            return merged
        
        history_grad = process_history(history_grad)
        history_greed = process_history(history_greed)
        history_active = process_history(history_active)
        history_modulus = process_history(history_modulus)
        history_dropout = process_history(history_dropout)  # New processing
        
        all_valid_ids = true_features[
            true_features['true_feature'].between(lower_bound, upper_bound)
        ]['id'].tolist()
        
    except Exception as e:
        print(f"Data processing error: {str(e)}")
        return

    def random_sampling(total_samples, batch_size=50):
        np.random.seed(42)
        all_ids = true_features['id'].tolist()
        cumulative = []
        for i in tqdm(range(total_samples), desc="Simulating random sampling"):
            sampled = np.random.choice(all_ids, size=batch_size, replace=False)
            valid = sum(1 for id in sampled if id in all_valid_ids)
            cumulative.append(valid + (cumulative[-1] if cumulative else 0))
        return np.array(cumulative)

    # Processing with checkpoints
    def process_history_with_checkpoints(history_df, color, marker, label):
        sorted_df = history_df.sort_values('iteration')
        cumulative = sorted_df.groupby('iteration')['actual_valid'].sum().cumsum().values
        resources = np.arange(1, len(cumulative)+1) * 50
        
        # Checkpoint analysis
        checkpoints = [100, 200, 400, 800, 1600]
        checkpoint_results = {}
        for cp in checkpoints:
            idx = np.searchsorted(resources, cp)
            if idx < len(cumulative):
                checkpoint_results[cp] = cumulative[idx]
            else:
                checkpoint_results[cp] = cumulative[-1] if len(cumulative) > 0 else 0
        
        return {
            'x': resources,
            'y': cumulative,
            'color': color,
            'marker': marker,
            'label': label,
            'checkpoints': checkpoint_results
        }

    # All algorithms including the new Dropout
    algorithms = [
        process_history_with_checkpoints(history_active, '#9467bd', '^', 'Ourwork'),
        process_history_with_checkpoints(history_grad, '#1f77b4', 'o', 'Explore'),
        process_history_with_checkpoints(history_greed, '#2ca02c', 's', 'Greedy'),
        process_history_with_checkpoints(history_modulus, '#ff7f0e', 'D', 'Modulus'),
        process_history_with_checkpoints(history_dropout, '#e377c2', 'p', 'MC_Dropout'),  # New
    ]

    # Print checkpoint results
    print("\nCheckpoint Results:")
    checkpoints = [100, 200, 400, 800, 1600]
    print(f"{'Method':<15}", end="")
    for cp in checkpoints:
        print(f"{cp:>8}", end="")
    print()
    
    for algo in algorithms:
        print(f"{algo['label']:<15}", end="")
        for cp in checkpoints:
            print(f"{algo['checkpoints'][cp]:>8}", end="")
        print()

    # Determine maximum iterations for random baseline
    max_iterations = max(len(algo['y']) for algo in algorithms) if algorithms else 0
    
    # Calculate random baseline
    random_runs = []
    for _ in range(args.random_trials):
        random_runs.append(random_sampling(max_iterations))
    random_median = np.median(random_runs, axis=0)
    random_x = np.arange(1, max_iterations+1) * 50

    # Generate perfect efficiency line
    max_resource = max(algo['x'][-1] for algo in algorithms) if algorithms else 0
    perfect_x = np.arange(0, max_resource + 5, 5)
    perfect_y = perfect_x / 1
    
    perfect_y = np.minimum(perfect_y, len(all_valid_ids))
    perfect_x = perfect_y * 1

    # Set font parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'figure.titlesize': 12
    })

    plt.figure(figsize=(12, 7))
    
    # Plot algorithm curves
    for algo in algorithms:
        plt.plot(algo['x'], algo['y'], 
                 color=algo['color'], 
                 marker=algo['marker'],
                 linestyle='-',
                 markersize=6,
                 linewidth=2,
                 label=algo['label'])

    # Plot baseline curves
    plt.plot(random_x[:len(random_median)], random_median, 
             color='#8c564b', 
             marker='x',
             linestyle='--',
             markersize=4,
             linewidth=2,
             label='Random Sampling')
    
    # Plot perfect efficiency line
    plt.plot(perfect_x, perfect_y, 
             color='#d62728', 
             linestyle='--', 
             linewidth=2,
             label='Perfect Efficiency')
    
    plt.axhline(len(all_valid_ids), 
                color='#17becf', 
                linestyle='--', 
                linewidth=2,
                label=f'Total Valid Samples')

    # Configure axes
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    
    plt.xlabel('Annotation Resources Consumed')
    plt.ylabel('Valid Samples Identified')
    plt.title(f'{args.custom_annotation}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=len(all_valid_ids)*1.1)
    
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300)
    plt.close()

    print(f"\nResults saved to: {args.output_image}")

if __name__ == '__main__':
    main()
                       
#python active_learning_analysis.py --custom-annotation "Property: Band_gap  Batch_size: 50" --history-grad sampled_history_bandgap_50_explore.csv --history-greed sampled_history_bandgap_50_greedy.csv --history-active sampled_history_bandgap_50_ourwork37.csv --history-modulus sampled_history_bandgap_50_modulus.csv --history-dropout sampled_history_bandgap_50_dropout.csv --true-features id_prop_bandgap.csv --target-feature 1.5 --tolerance 0.3 --output-image active_learning_dropout.png
#python active_learning_analysis.py --custom-annotation "Property: Band_gap  Batch_size: 50" --history-grad sampled_history_bandgap_50_explore.csv --history-greed sampled_history_bandgap_50_greedy.csv --history-active sampled_history_bandgap_50_ucb.csv --history-modulus sampled_history_bandgap_50_modulus.csv --history-dropout sampled_history_bandgap_50_dropout.csv --true-features id_prop_bandgap.csv --target-feature 1.5 --tolerance 0.3 --output-image active_learning_ucb.png