import torch
import collections
from primitives.functions import get_functions
from primitives.terminals import InputTerminal, IndexTerminal
from utils.node import Node


class Network:
    """
    This implementation treats the index terminal as an edge. As a result, the index terminal would not be considered in the node count.
    The other approach would be to treat the index terminals as actual nodes in the tree, they will be leaves in the tree, and affect index counting.
    """
    def __init__(self, individual):
        # TODO: move this to global scope for optimization
        funcs, arities = zip(*get_functions())
        self.func_arity = dict(zip(funcs, arities))

        self.expression = individual.expression
        self.head_length = individual.head_length
        self.num_inputs = individual.num_inputs
        self.weights = list(individual.weights)  # NOTE: using deeper copy just in case
        self.biases = list(individual.biases)

        self.nodes = []
        self.root = self._build_tree()

    def _build_tree(self):
        """Build tree using standard GEP parsing with IndexTerminals as edges."""
        if not self.expression:
            return None

        # Check if first element is IndexTerminal - invalid expression
        if isinstance(self.expression[0], IndexTerminal):
            # Create a dummy node that returns 0
            dummy = Node(lambda: torch.tensor(0.0, dtype=torch.float32))
            dummy.node_index = 0
            self.nodes.append(dummy)
            return dummy

        # Create ALL nodes first (except IndexTerminals)
        for i, symbol in enumerate(self.expression):
            if not isinstance(symbol, IndexTerminal):
                node = Node(symbol)
                node.node_index = len(self.nodes)
                self.nodes.append(node)

        # Assign biases to function nodes
        bias_idx = 0
        for node in self.nodes:
            if node.symbol in self.func_arity:
                node.bias = self.biases[bias_idx] if bias_idx < len(self.biases) else 0.0
                bias_idx += 1

        # Now build connections using sequential GEP parsing
        root = self.nodes[0]
        weight_idx = 0

        # Track which expression positions map to which nodes
        expr_to_node = {}
        node_idx = 0
        for i, symbol in enumerate(self.expression):
            if not isinstance(symbol, IndexTerminal):
                expr_to_node[i] = node_idx
                node_idx += 1

        # Initialize reference storage
        for node in self.nodes:
            node.ref_children = []
            node.ref_weights = []

        # Queue: (parent_node, remaining_arity)
        queue = collections.deque()
        if root.symbol in self.func_arity:
            queue.append((root, self.func_arity[root.symbol]))

        # Sequential parsing: process expression positions one by one
        expr_idx = 1

        while queue and expr_idx < len(self.expression):
            parent, remaining_arity = queue.popleft()

            if remaining_arity == 0:
                continue

            # Process one position for this parent
            symbol = self.expression[expr_idx]

            if isinstance(symbol, IndexTerminal):
                # Reference edge - add directly to parent
                weight = self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0
                weight_idx += 1

                # Add reference if target exists
                if symbol.index < len(self.nodes):
                    parent.ref_children.append(self.nodes[symbol.index])
                    parent.ref_weights.append(weight)
            else:
                # Normal child node
                child_node_idx = expr_to_node[expr_idx]
                child = self.nodes[child_node_idx]
                child.weight = self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0
                weight_idx += 1
                parent.children.append(child)

                # If child is a function, add it to end of queue
                if child.symbol in self.func_arity:
                    queue.append((child, self.func_arity[child.symbol]))

            # Decrement remaining arity and put parent back if needed
            remaining_arity -= 1
            if remaining_arity > 0:
                queue.appendleft((parent, remaining_arity))

            # Move to next position
            expr_idx += 1

        return root

    def forward(self, inputs):
        """Execute forward pass through the network."""
        # Convert inputs to consistent format
        if isinstance(inputs, dict):
            # Validate all required inputs are present
            for i in range(self.num_inputs):  # Need to add self.num_inputs to __init__
                if f'x{i}' not in inputs:
                    raise ValueError(f"Missing input x{i}")
            # Only convert non-tensors
            input_tensors = {}
            for k, v in inputs.items():
                input_tensors[k] = v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
        else:
            # Assume list/array input
            if len(inputs) < self.num_inputs:
                raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
            input_tensors = inputs if torch.is_tensor(inputs) else torch.tensor(inputs, dtype=torch.float32)

        # Reset all node values
        for node in self.nodes:
            node.value = None

        output = self._evaluate(self.root, input_tensors)

        # Update previous values for next timestep
        for node in self.nodes:
            if node.value is not None:
                node.prev_value = node.value.item() if torch.is_tensor(node.value) else node.value

        return output

    def _evaluate(self, node, inputs):
        """Recursively evaluate a node."""
        # Check if already computed
        if node.value is not None:
            return node.value

        symbol = node.symbol

        # Handle terminals
        if isinstance(symbol, InputTerminal):
            if isinstance(inputs, dict):
                value = inputs.get(f'x{symbol.index}', 0.0)
            else:
                value = inputs[symbol.index] if symbol.index < len(inputs) else 0.0
            # Ensure tensor output
            node.value = torch.tensor(value, dtype=torch.float32) if not torch.is_tensor(value) else value
            return node.value

        # Handle functions
        if symbol in self.func_arity:
            child_values = []

            # Evaluate all children (normal and reference) in order
            # First, normal children
            for child in node.children:
                child_value = self._evaluate(child, inputs)
                weighted_value = child_value * child.weight
                child_values.append(weighted_value)

            # Then, reference children
            for ref_child, ref_weight in zip(node.ref_children, node.ref_weights):
                # Check for recurrence
                if ref_child.value is None and ref_child.node_index <= node.node_index:
                    # Recurrent connection - use previous value
                    prev_val = ref_child.prev_value if ref_child.prev_value != 0.0 else 0.0
                    ref_value = torch.tensor(prev_val, dtype=torch.float32)
                else:
                    # Forward reference or already computed
                    ref_value = self._evaluate(ref_child, inputs)

                weighted_value = ref_value * ref_weight
                child_values.append(weighted_value)

            # Apply function
            if child_values:
                result = symbol(*child_values)
                if node.bias is not None:
                    result = result + node.bias
                if not torch.is_tensor(result):
                    result = torch.tensor(result, dtype=torch.float32)
            else:
                result = torch.tensor(0.0, dtype=torch.float32)

            node.value = result
            return result

        # Unknown symbol type - shouldn't happen
        node.value = torch.tensor(0.0, dtype=torch.float32)
        return node.value

    def get_active_parameters(self):
        """Get count of weights and biases actually used in the network."""
        # Count weights: one per child + all reference weights
        active_weights = 0
        active_biases = 0

        for node in self.nodes:
            # Count normal edge weights
            active_weights += len(node.children)

            # Count reference edge weights
            active_weights += len(node.ref_weights)

            # Count biases (only functions have them)
            if node.bias is not None:
                active_biases += 1

        return active_weights, active_biases

    def print_structure(self):
        """Print the network structure for debugging."""
        print(f"Network has {len(self.nodes)} nodes:")

        for node in self.nodes:
            # Basic node info
            symbol_name = getattr(node.symbol, '__name__',
                                  getattr(node.symbol, 'name', str(node.symbol)))
            print(f"\nNode {node.node_index}: {symbol_name}")

            if node.bias is not None:
                print(f"  Bias: {node.bias:.3f}")

            # Show connections
            if node.children:
                child_info = []
                for child in node.children:
                    child_symbol = getattr(child.symbol, '__name__',
                                           getattr(child.symbol, 'name', str(child.symbol)))
                    child_info.append(f"{child.node_index}:{child_symbol} (w={child.weight:.3f})")
                print(f"  Children: {', '.join(child_info)}")

            if node.ref_children:
                ref_info = []
                for ref_child, weight in zip(node.ref_children, node.ref_weights):
                    ref_symbol = getattr(ref_child.symbol, '__name__',
                                         getattr(ref_child.symbol, 'name', str(ref_child.symbol)))
                    ref_info.append(f"{ref_child.node_index}:{ref_symbol} (w={weight:.3f})")
                print(f"  References: {', '.join(ref_info)}")