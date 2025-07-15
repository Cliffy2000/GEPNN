import torch
import collections
from primitives.functions import get_functions
from primitives.terminals import InputTerminal, IndexTerminal
from utils.node import Node


class Network:
    def __init__(self, individual):
        # Get function arities
        funcs, arities = zip(*get_functions())
        self.func_arity = dict(zip(funcs, arities))

        # Get expression and parameters
        self.expression = individual.get_expression()
        self.weights = individual.weights
        self.biases = individual.biases
        self.head_length = individual.head_length

        # Build tree structure
        self.nodes = []
        self.root = self._build_tree()

    def _build_tree(self):
        """Build tree using standard GEP parsing.
        Stop when tree is complete. IndexTerminals don't consume positions.
        """
        if not self.expression:
            return None

        # Create root
        root_symbol = self.expression[0]
        root = Node(root_symbol, position=0)
        self.nodes.append(root)

        # Track assignments
        weight_idx = 0
        bias_idx = 0

        # Assign bias to root if it's a function
        if root_symbol in self.func_arity:
            root.bias = self.biases[bias_idx] if bias_idx < len(self.biases) else 0.0
            bias_idx += 1

        # Queue: (parent_node, remaining_children)
        queue = collections.deque()
        if root_symbol in self.func_arity:
            queue.append((root, self.func_arity[root_symbol]))

        # Parse expression
        expr_idx = 1

        while queue and expr_idx < len(self.expression):
            parent, remaining = queue.popleft()

            if remaining == 0:
                continue

            # Process one child
            symbol = self.expression[expr_idx]

            if isinstance(symbol, IndexTerminal):
                # Reference to existing node - doesn't consume a tree position
                ref_idx = symbol.index

                # Store the reference for later resolution if node doesn't exist yet
                if not hasattr(parent, 'pending_refs'):
                    parent.pending_refs = []
                parent.pending_refs.append((ref_idx, weight_idx))
                weight_idx += 1

                # Put parent back with one less child needed
                if remaining > 1:
                    queue.appendleft((parent, remaining - 1))

                # Move to next expression position
                expr_idx += 1

            else:
                # Create new node for function or terminal
                child = Node(symbol, position=len(self.nodes))
                self.nodes.append(child)

                # Add as normal child with weight
                child.weight = self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0
                weight_idx += 1
                parent.add_child(child)

                # If child is a function, assign bias and add to queue
                if symbol in self.func_arity:
                    child.bias = self.biases[bias_idx] if bias_idx < len(self.biases) else 0.0
                    bias_idx += 1
                    queue.append((child, self.func_arity[symbol]))

                # Put parent back if more children needed
                if remaining > 1:
                    queue.appendleft((parent, remaining - 1))

                # Move to next expression position
                expr_idx += 1

        # Post-process: Resolve all pending references
        for node in self.nodes:
            if hasattr(node, 'pending_refs'):
                if not hasattr(node, 'ref_children'):
                    node.ref_children = []
                    node.ref_weights = []

                for ref_idx, weight_idx in node.pending_refs:
                    if 0 <= ref_idx < len(self.nodes):
                        ref_node = self.nodes[ref_idx]
                        node.ref_children.append(ref_node)
                        node.ref_weights.append(self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0)

                # Clean up
                delattr(node, 'pending_refs')

        return root

    def forward(self, inputs):
        """Execute forward pass through the network."""
        # Convert inputs to tensor format if needed
        if isinstance(inputs, dict):
            input_tensors = {k: v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
                             for k, v in inputs.items()}
        else:
            input_tensors = inputs if torch.is_tensor(inputs) else torch.tensor(inputs, dtype=torch.float32)

        # Reset all node values
        for node in self.nodes:
            node.reset()

        # Evaluate the tree
        output = self._evaluate(self.root, input_tensors)

        # Update previous values for recurrence
        for node in self.nodes:
            node.update_prev()

        return output

    def _evaluate(self, node, inputs):
        """Recursively evaluate a node."""
        # Check if already computed
        if node.value is not None:
            return node.value

        symbol = node.symbol

        # Handle terminals
        if isinstance(symbol, InputTerminal):
            # Get input value
            if isinstance(inputs, dict):
                value = inputs.get(f'x{symbol.index}', torch.tensor(0.0))
            else:
                value = inputs[symbol.index] if symbol.index < len(inputs) else torch.tensor(0.0)
            node.value = value
            return value

        # Handle functions
        elif symbol in self.func_arity:
            # Evaluate all children (both normal and reference)
            child_values = []

            # Normal children
            for child in node.children:
                child_value = self._evaluate(child, inputs)
                weighted_value = child_value * child.weight
                child_values.append(weighted_value)

            # Reference children (from IndexTerminals)
            if hasattr(node, 'ref_children'):
                for ref_child, ref_weight in zip(node.ref_children, node.ref_weights):
                    # Check if reference creates a cycle
                    if ref_child.value is None and ref_child.position <= node.position:
                        # Recurrent connection - use previous value
                        if torch.is_tensor(ref_child.prev_value):
                            ref_value = ref_child.prev_value.detach().clone()
                        else:
                            ref_value = torch.tensor(ref_child.prev_value, dtype=torch.float32)
                    else:
                        # Forward reference or already computed
                        ref_value = self._evaluate(ref_child, inputs)
                    weighted_value = ref_value * ref_weight
                    child_values.append(weighted_value)

            # Apply function
            if child_values:
                result = symbol(*child_values)
                # Add bias
                if node.bias is not None:
                    result = result + node.bias
            else:
                result = torch.tensor(0.0)

            node.value = result
            return result

        else:
            # Unknown symbol type
            node.value = torch.tensor(0.0)
            return node.value

    def get_active_parameters(self):
        """Get count of weights and biases actually used in the network."""
        active_weights = sum(1 for node in self.nodes if node.weight is not None)
        for node in self.nodes:
            if hasattr(node, 'ref_weights'):
                active_weights += len(node.ref_weights)
        active_biases = sum(1 for node in self.nodes if node.bias is not None)
        return active_weights, active_biases

    def print_structure(self):
        """Print the network structure for debugging."""
        print(f"Network has {len(self.nodes)} active nodes:")
        for i, node in enumerate(self.nodes):
            print(f"  Node {i}: {node.symbol}")
            if node.children:
                print(f"    Normal children: {[self.nodes.index(c) for c in node.children]}")
            if hasattr(node, 'ref_children'):
                print(f"    Reference children: {[self.nodes.index(c) for c in node.ref_children]}")