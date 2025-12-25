import os
import json
import glob
import numpy as np


def analyze_xor_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, "..", "experiments_v2", "xor")

    results = []

    for filepath in glob.glob(os.path.join(experiments_dir, "xor_sy*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)

        params = data['parameters']
        head = params['head_length']
        mutation = params['mutation_rate']

        gens = [r['generations'] for r in data['results'] if r['perfect_found']]

        gens_under_1000 = [g for g in gens if g <= 1000]

        results.append({
            'head_length': head,
            'mutation_rate': mutation,
            'success_rate': len(gens) / len(data['results']),
            'under_1000_rate': len(gens_under_1000) / len(data['results']),
            'mean_gens': np.mean(gens) if gens else None,
            'median_gens': np.median(gens) if gens else None,
            'min_gens': np.min(gens) if gens else None,
            'max_gens': np.max(gens) if gens else None,
        })

    results.sort(key=lambda x: (x['head_length'], x['mutation_rate']))

    print(f"{'HEAD':<6} {'MUT':<6} {'SUCCESS':<8} {'<1000':<8} {'MEAN':<8} {'MEDIAN':<8} {'MIN':<6} {'MAX':<6}")
    print("-" * 62)
    for r in results:
        mean = f"{r['mean_gens']:.1f}" if r['mean_gens'] else "N/A"
        median = f"{r['median_gens']:.1f}" if r['median_gens'] else "N/A"
        min_g = f"{r['min_gens']}" if r['min_gens'] else "N/A"
        max_g = f"{r['max_gens']}" if r['max_gens'] else "N/A"
        under_1000 = f"{r['under_1000_rate'] * 100:.0f}"
        print(f"{r['head_length']:<6} {r['mutation_rate']:<6} {r['success_rate'] * 100:<8.0f} {under_1000:<8} {mean:<8} {median:<8} {min_g:<6} {max_g:<6}")

if __name__ == '__main__':
    analyze_xor_results()
