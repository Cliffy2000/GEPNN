import torch
from typing import List, Dict, Tuple, Any
from core.individual import GepnnIndividual
from primitives.functions import get_functions


class GepnnNetwork:
    """Converts GEP individual to executable neural network."""

    def __init__(self, individual: GepnnIndividual):
        self.individual = individual
        self.expression = individual.get_expression()
        self.weights = individual.weights
        self.biases = individual.biases

        # Get function arity mapping
        self.function_arity = {func.__name__: arity for func, arity in get_functions()}

        # Parse the expression into tree structure
        self.tree_nodes, self.tree_levels = self._parse_karva_to_tree()

        # Assign parameters
        self.weight_mapping, self.bias_mapping = self._assign_parameters()

    def _parse_karva_to_tree(self) -> Tuple[List[Dict], Dict[int, List[int]]]:
        """Parse Karva notation into tree structure with level information.

        Returns:
            tree_nodes: List of node dictionaries with properties
            tree_levels: Dict mapping level -> list of node indices
        """
        tree_nodes = []
        tree_levels = {0: []}

        # First pass: identify nodes and their properties
        for i, item in enumerate(self.expression):
            node = {
                'index': i,
                'item': item,
                'is_function': False,
                'arity': 0,
                'children': [],
                'level': -1
            }

            # Check if it's a function
            item_name = getattr(item, '__name__', getattr(item, 'name', str(item)))
            if item_name in self.function_arity:
                node['is_function'] = True
                node['arity'] = self.function_arity[item_name]

            tree_nodes.append(node)

        # Second pass: build tree structure using level-order
        current_level = 0
        queue = [0]  # Start with root
        tree_levels[0] = [0]
        tree_nodes[0]['level'] = 0

        expr_idx = 1  # Next position in expression to use

        while queue and expr_idx < len(tree_nodes):
            next_queue = []

            for parent_idx in queue:
                parent = tree_nodes[parent_idx]

                # Add children based on arity
                for _ in range(parent['arity']):
                    if expr_idx >= len(tree_nodes):
                        break

                    # Link parent to child
                    parent['children'].append(expr_idx)
                    tree_nodes[expr_idx]['level'] = current_level + 1

                    # Add to appropriate level
                    if current_level + 1 not in tree_levels:
                        tree_levels[current_level + 1] = []
                    tree_levels[current_level + 1].append(expr_idx)

                    # Add to next queue if it's a function
                    if tree_nodes[expr_idx]['is_function']:
                        next_queue.append(expr_idx)

                    expr_idx += 1

            queue = next_queue
            current_level += 1

        return tree_nodes, tree_levels

    def _assign_parameters(self) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
        """Assign weights to edges and biases to nodes using level-order traversal.

        Returns:
            weight_mapping: Dict mapping (parent_idx, child_idx) -> weight_idx
            bias_mapping: Dict mapping node_idx -> bias_idx
        """
        weight_mapping = {}
        bias_mapping = {}

        weight_idx = 0
        bias_idx = 0

        # Process nodes level by level
        for level in sorted(self.tree_levels.keys()):
            for node_idx in self.tree_levels[level]:
                node = self.tree_nodes[node_idx]

                # Assign bias if it's a function node
                if node['is_function']:
                    bias_mapping[node_idx] = bias_idx
                    bias_idx += 1

                # Assign weights to edges leading to children
                for child_idx in node['children']:
                    weight_mapping[(node_idx, child_idx)] = weight_idx
                    weight_idx += 1

        return weight_mapping, bias_mapping

    def compile(self) -> callable:
        """Compile the network into a callable function, similar to geppy's compile_.

        Returns:
            A function that takes inputs and returns the network output
        """

        def network_function(*args):
            # Map positional arguments to input names
            input_names = [f'x{i}' for i in range(len(args))]
            inputs = {name: torch.tensor(float(arg)) if isinstance(arg, (int, float)) else arg
                      for name, arg in zip(input_names, args)}

            return self._evaluate(inputs)

        return network_function

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute forward pass through the network.

        Args:
            inputs: Dict mapping input names (x0, x1, etc.) to tensors

        Returns:
            Output tensor from the network
        """
        return self._evaluate(inputs)

    def _evaluate(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Internal evaluation method used by both compile and forward.

        Args:
            inputs: Dict mapping input names to tensors

        Returns:
            Output tensor from the network
        """
        # Cache for computed node values
        node_values = {}

        # Compute values bottom-up (reverse level order)
        max_level = max(self.tree_levels.keys()) if self.tree_levels else 0

        for level in range(max_level, -1, -1):
            for node_idx in self.tree_levels.get(level, []):
                node = self.tree_nodes[node_idx]
                item = node['item']

                # Get item name
                item_name = getattr(item, '__name__', getattr(item, 'name', str(item)))

                if node['is_function']:
                    # Collect weighted inputs from children
                    child_values = []
                    for child_idx in node['children']:
                        child_value = node_values[child_idx]

                        # Apply weight to edge
                        edge_key = (node_idx, child_idx)
                        if edge_key in self.weight_mapping:
                            weight = self.weights[self.weight_mapping[edge_key]]
                            child_value = child_value * weight

                        child_values.append(child_value)

                    # Apply function
                    result = item(*child_values)

                    # Add bias
                    if node_idx in self.bias_mapping:
                        bias = self.biases[self.bias_mapping[node_idx]]
                        result = result + bias

                    node_values[node_idx] = result

                else:
                    # Terminal node - get input value
                    if item_name in inputs:
                        node_values[node_idx] = inputs[item_name]
                    else:
                        # Handle case where terminal is a constant or missing
                        node_values[node_idx] = torch.tensor(0.0)

        # Return root node value
        return node_values[0] if 0 in node_values else torch.tensor(0.0)

    def get_active_parameters(self) -> Tuple[int, int]:
        """Get the number of weights and biases actually used in the network."""
        return len(self.weight_mapping), len(self.bias_mapping)

    def visualize_structure(self) -> str:
        """Generate a text representation of the network structure."""
        lines = []

        for level in sorted(self.tree_levels.keys()):
            lines.append(f"Level {level}:")
            for node_idx in self.tree_levels[level]:
                node = self.tree_nodes[node_idx]
                item_name = getattr(node['item'], '__name__',
                                    getattr(node['item'], 'name', str(node['item'])))

                if node['is_function']:
                    bias_idx = self.bias_mapping.get(node_idx, -1)
                    lines.append(f"  Node {node_idx}: {item_name} (bias_idx={bias_idx})")

                    for i, child_idx in enumerate(node['children']):
                        weight_idx = self.weight_mapping.get((node_idx, child_idx), -1)
                        child = self.tree_nodes[child_idx]
                        child_name = getattr(child['item'], '__name__',
                                             getattr(child['item'], 'name', str(child['item'])))
                        lines.append(f"    -> Node {child_idx}: {child_name} (weight_idx={weight_idx})")
                else:
                    lines.append(f"  Node {node_idx}: {item_name} (terminal)")

        active_weights, active_biases = self.get_active_parameters()
        lines.append(f"\nActive parameters: {active_weights} weights, {active_biases} biases")
        lines.append(f"Total available: {len(self.weights)} weights, {len(self.biases)} biases")

        return "\n".join(lines)