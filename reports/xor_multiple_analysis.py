import json
import numpy as np
from collections import defaultdict, Counter
import re


def parse_and_build_tree(expr_str):
    """Parse expression and simulate GEP tree building to get exact active nodes."""
    # Extract head symbols
    head_match = re.search(r'Head:\[(.*?)\]', expr_str)
    tail_match = re.search(r'Tail:\[(.*?)\]', expr_str)

    if not head_match or not tail_match:
        return None

    head_str = head_match.group(1)
    tail_str = tail_match.group(1)

    head_symbols = [s.strip().strip("'") for s in head_str.split(',')]
    tail_symbols = [s.strip().strip("'") for s in tail_str.split(',')]

    expression = head_symbols + tail_symbols

    # Function arities
    func_arity = {
        'add': 2, 'add3': 3, 'subtract': 2, 'multiply': 2,
        'relu': 2, 'relu3': 3, 'tanh': 2, 'tanh3': 3,
        'sigmoid': 2, 'sigmoid3': 3,
        'not_f': 1, 'and_f': 2, 'or_f': 2
    }

    # Build tree using GEP sequential parsing
    active_nodes = []  # List of (position_in_tree, symbol, expr_index)

    # Start with root
    if expression[0] in func_arity:
        active_nodes.append((0, expression[0], 0))
        queue = [(0, func_arity[expression[0]])]
    else:
        # Invalid tree
        return None

    expr_idx = 1
    node_position = 1

    # Sequential parsing
    while queue and expr_idx < len(expression):
        parent_pos, remaining = queue.pop(0)

        for _ in range(remaining):
            if expr_idx >= len(expression):
                break

            symbol = expression[expr_idx]
            active_nodes.append((node_position, symbol, expr_idx))

            if symbol in func_arity:
                queue.append((node_position, func_arity[symbol]))

            expr_idx += 1
            node_position += 1

    return active_nodes


def analyze_node_distributions(json_file):
    """Analyze exact node distribution at each position."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = data['results']
    perfect_solutions = [r for r in results if r['perfect_found']]

    if not perfect_solutions:
        print("No perfect solutions found!")
        return

    # Collect distributions
    position_distributions = defaultdict(list)  # position -> list of symbols

    for result in perfect_solutions:
        active_nodes = parse_and_build_tree(result['best_individual']['expression'])
        if active_nodes:
            for pos, symbol, _ in active_nodes:
                position_distributions[pos].append(symbol)

    print("=" * 70)
    print("EXACT NODE DISTRIBUTION BY TREE POSITION")
    print("=" * 70)
    print(f"Analyzing {len(perfect_solutions)} perfect solutions\n")

    # Print distribution for each position
    max_position = max(position_distributions.keys())

    for pos in range(max_position + 1):
        if pos not in position_distributions:
            continue

        symbols = position_distributions[pos]
        symbol_counts = Counter(symbols)
        total = len(symbols)

        print(f"\nPosition {pos}:")
        print("-" * 40)

        # Sort by frequency
        for symbol, count in symbol_counts.most_common():
            percentage = (count / total) * 100
            bar = '#' * int(percentage / 2)  # Bar chart
            print(f"  {symbol:10s}: {bar:25s} {count:3d} ({percentage:5.1f}%)")

    # Summary statistics
    print("\n\nSUMMARY STATISTICS")
    print("=" * 50)

    # Tree sizes
    tree_sizes = [len(parse_and_build_tree(r['best_individual']['expression']) or [])
                  for r in perfect_solutions]
    tree_sizes = [s for s in tree_sizes if s > 0]

    print(f"Tree Size Distribution:")
    size_counter = Counter(tree_sizes)
    for size, count in sorted(size_counter.items()):
        print(f"  {size} nodes: {count} solutions")

    # Common patterns
    print("\n\nCOMMON PATTERNS")
    print("-" * 30)

    # Check for common root-child combinations
    root_child_patterns = defaultdict(int)

    for result in perfect_solutions:
        active_nodes = parse_and_build_tree(result['best_individual']['expression'])
        if active_nodes and len(active_nodes) >= 3:
            root = active_nodes[0][1]
            child1 = active_nodes[1][1] if len(active_nodes) > 1 else None
            child2 = active_nodes[2][1] if len(active_nodes) > 2 else None

            if child1 and child2:
                pattern = f"{root}({child1}, {child2})"
                root_child_patterns[pattern] += 1

    print("Root-Children Patterns:")
    for pattern, count in sorted(root_child_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pattern}: {count}")


if __name__ == "__main__":
    analyze_node_distributions(r"../experiments/xor_multiple_results_20250827_061752.json")