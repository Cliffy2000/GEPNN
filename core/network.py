import collections
from primitives.functions import get_functions
from primitives.terminals import InputTerminal, IndexTerminal
from utils.node import Node

class Network:
    def __init__(self, individual):
        funcs, arities = zip(*get_functions())
        self.func_arity = dict(zip(funcs, arities))
        expr = individual.get_expression()
        self.weights = list(individual.weights)
        self.biases  = list(individual.biases)
        self.head_length = individual.head_length
        self.root, self.nodes = self._build_tree(expr)

    def _build_tree(self, symbols):
        w_iter = iter(self.weights)
        b_iter = iter(self.biases)
        root_sym = symbols[0]
        root = Node(root_sym, next(w_iter), next(b_iter), position=0)
        nodes = [root]
        queue = collections.deque([(root, self.func_arity.get(root_sym, 0))])
        i = 1
        while queue and i < len(symbols):
            parent, rem = queue.popleft()
            if rem == 0:
                continue
            sym = symbols[i]
            if sym in self.func_arity:
                node = Node(sym, next(w_iter), next(b_iter), position=i)
                queue.appendleft((parent, rem-1))
                queue.append((node, self.func_arity[sym]))
            else:
                node = Node(sym, position=i)
                queue.appendleft((parent, rem-1))
            parent.add_child(node)
            nodes.append(node)
            i += 1
        return root, nodes

    def evaluate(self, input_vector):
        def _eval(node):
            if node.value is not None:
                return node.value
            if not node.children:
                if isinstance(node.symbol, InputTerminal):
                    val = input_vector[node.symbol.index]
                elif isinstance(node.symbol, IndexTerminal):
                    ref = self.nodes[node.symbol.index]
                    val = ref.value if ref.value is not None else ref.prev_value
                else:
                    val = node.symbol
            else:
                args = [_eval(c) for c in node.children]
                val = node.symbol(*args) * node.weight + node.bias
            node.value = val
            return val

        output = _eval(self.root)
        for n in self.nodes:
            n.update_prev()
            n.reset()
        return output