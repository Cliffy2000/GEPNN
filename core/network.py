import torch
import collections
from primitives.functions import get_functions
from primitives.terminals import InputTerminal, IndexTerminal
from utils.node import Node

'''
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

'''

class Network:
    def __init__(self, individual):
        # TODO: move this to global scope for optimization
        funcs, arities = zip(*get_functions())
        self.func_arity = dict(zip(funcs, arities))

        self.expression = individual.expression
        self.head_length = individual.head_length
        self.num_inputs = individual.num_inputs
        self.weights = list(individual.weights)     # NOTE: using deeper copy just in case
        self.biases = list(individual.biases)

        self.nodes = []
        self.root = self._build_tree()

    def _build_tree(self):
        """Build tree using standard GEP parsing with IndexTerminals as edges."""
        if not self.expression:
            return None

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

        # Now build connections
        root = self.nodes[0]
        weight_idx = 0

        # Track which expression positions map to which nodes
        expr_to_node = {}
        node_idx = 0
        for i, symbol in enumerate(self.expression):
            if not isinstance(symbol, IndexTerminal):
                expr_to_node[i] = node_idx
                node_idx += 1

        # Queue: (parent_node, expr_positions_to_process)
        queue = collections.deque()
        if root.symbol in self.func_arity:
            arity = self.func_arity[root.symbol]
            child_positions = list(range(1, min(1 + arity, len(self.expression))))
            if child_positions:
                queue.append((root, child_positions))

        # Track pending references
        pending_refs = collections.defaultdict(list)

        while queue:
            parent, positions = queue.popleft()

            for pos in positions:
                if pos >= len(self.expression):
                    break

                symbol = self.expression[pos]

                if isinstance(symbol, IndexTerminal):
                    # Reference edge - store for later
                    weight = self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0
                    weight_idx += 1
                    pending_refs[parent].append((symbol.index, weight))
                else:
                    # Normal child
                    child_node_idx = expr_to_node[pos]
                    child = self.nodes[child_node_idx]
                    child.weight = self.weights[weight_idx] if weight_idx < len(self.weights) else 1.0
                    weight_idx += 1
                    parent.children.append(child)

                    # If child is a function, find its children
                    if child.symbol in self.func_arity:
                        arity = self.func_arity[child.symbol]
                        # Find next positions after current batch
                        next_start = max(positions) + 1
                        child_positions = []

                        # Scan forward to find positions for this child
                        scan_pos = next_start
                        found = 0
                        while found < arity and scan_pos < len(self.expression):
                            child_positions.append(scan_pos)
                            # Skip the position but count it as consuming a child slot
                            found += 1
                            scan_pos += 1

                        if child_positions:
                            queue.append((child, child_positions))

        # Resolve all pending references
        for parent, refs in pending_refs.items():
            for target_idx, weight in refs:
                if target_idx < len(self.nodes):
                    parent.ref_children.append(self.nodes[target_idx])
                    parent.ref_weights.append(weight)

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
