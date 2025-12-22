import torch
import numpy as np
import collections
from core.individual import Individual
from primitives.functions import get_functions
from primitives.terminals import InputTerminal, IndexTerminal


_FUNC_ARITY = None


def _get_func_arity():
    global _FUNC_ARITY
    if _FUNC_ARITY is None:
        funcs, arities = zip(*get_functions())
        _FUNC_ARITY = dict(zip(funcs, arities))
    return _FUNC_ARITY


class Network:
    """
    Updated Class that stors node information in separate arrays and uses numpy for ops
    """
    def __init__(self, individual: Individual):
        self.func_arity = _get_func_arity()
        self.expression = individual.expression
        self.num_inputs = individual.num_inputs

        self.n = len(self.expression)    # length of expression

        # Pretend 1.0 weight, this would align node index and the index of the used weight
        # The 1.0 is not used, and would not take effect even if it did
        self.weights = np.array([1.0] + list(individual.weights), dtype=np.float32)

        self.children = [[] for _ in range(self.n)]
        self.child_weights = [[] for _ in range(self.n)]

        self.is_function = np.zeros(self.n, dtype=bool)
        self.is_input = np.zeros(self.n, dtype=bool)
        self.is_index = np.zeros(self.n, dtype=bool)

        # assigns
        self.biases = np.zeros(self.n, dtype=np.float32)
        bias_idx = 0
        for i, sym in enumerate(self.expression):
            if sym in self.func_arity:
                self.is_function[i] = True
                self.biases[i] = individual.biases[bias_idx] if bias_idx < len(individual.biases) else 0.0
                bias_idx += 1
            elif isinstance(sym, InputTerminal):
                self.is_input[i] = True
            elif isinstance(sym, IndexTerminal):
                self.is_index[i] = True

        self.values = None
        self.prev_values = None

        self._build_tree()

    def _build_tree(self):
        if not self.expression or isinstance(self.expression[0], IndexTerminal):
            return

        queue = collections.deque()

        if self.is_function[0]:
            queue.append((0, self.func_arity[self.expression[0]]))

        expr_idx = 1
        while queue and expr_idx < len(self.expression):
            parent_idx, remaining = queue.popleft()

            self.children[parent_idx].append(expr_idx)
            self.child_weights[parent_idx].append(self.weights[expr_idx])

            if self.is_function[expr_idx]:
                queue.append((expr_idx, self.func_arity[self.expression[expr_idx]]))

            expr_idx += 1

            if remaining > 1:
                queue.appendleft((parent_idx, remaining - 1))

    '''
    def forward(self, inputs_batch):
        """
        Data of every instance in batch at single timestep
        :param inputs_batch: shape (batch_size, num_inputs)
        """
        batch_size = inputs_batch.shape[0]

        if self.prev_values is None:
            self.prev_values = np.zeros((self.n, batch_size), dtype=np.float32)

        self.values = np.zeros((self.n, batch_size), dtype=np.float32)
        self.evaluated = np.zeros(self.n, dtype=bool)  # Track evaluation status

        def evaluate(idx):
            # Skip if already evaluated
            if self.evaluated[idx]:
                return

            if self.is_input[idx]:
                input_idx = self.expression[idx].index
                self.values[idx] = inputs_batch[:, input_idx]

            elif self.is_index[idx]:
                target = self.expression[idx].index
                if target > idx:  # recurrent
                    self.values[idx] = self.prev_values[target]
                elif target < idx:  # skip connection
                    evaluate(target)  # Ensure target is evaluated
                    self.values[idx] = self.values[target]
                else:  # self loop
                    self.values[idx] = self.prev_values[idx]

            elif self.is_function[idx]:
                # Evaluate children first
                for child_idx in self.children[idx]:
                    evaluate(child_idx)

                # Now evaluate function
                if self.children[idx]:
                    child_vals = [self.values[c] * w for c, w in zip(self.children[idx], self.child_weights[idx])]
                    self.values[idx] = self.expression[idx](*child_vals) + self.biases[idx]
                else:
                    self.values[idx] = self.biases[idx]

            self.evaluated[idx] = True

        # Start from root
        evaluate(0)

        self.prev_values = self.values.copy()
        return self.values[0]
    '''

    def forward(self, inputs_batch):
        batch_size = inputs_batch.shape[0]

        if self.prev_values is None:
            self.prev_values = np.zeros((self.n, batch_size), dtype=np.float32)

        self.values = np.zeros((self.n, batch_size), dtype=np.float32)

        # First pass: Fill in all terminals
        for i in range(self.n):
            if self.is_input[i]:
                input_idx = self.expression[i].index
                self.values[i] = inputs_batch[:, input_idx]
            elif self.is_index[i]:
                target = self.expression[i].index
                if target < i:  # Recurrent (tail -> head from previous timestep)
                    self.values[i] = self.prev_values[target]
                elif target > i:  # Skip (head -> later head, evaluated this timestep)
                    pass  # Handle in second pass
                else:  # Self-loop
                    self.values[i] = self.prev_values[i]

        # Second pass: Process functions (reverse order) and skip connections
        for i in range(self.n - 1, -1, -1):
            if self.is_function[i]:
                if self.children[i]:
                    child_vals = [self.values[c] * w for c, w in
                                  zip(self.children[i], self.child_weights[i])]
                    self.values[i] = self.expression[i](*child_vals) + self.biases[i]
                else:
                    self.values[i] = self.biases[i]
            elif self.is_index[i]:
                target = self.expression[i].index
                if target > i:  # Skip connection only
                    self.values[i] = self.values[target]
                # Don't touch target < i (already set as recurrent)

        self.prev_values = self.values.copy()
        return self.values[0]