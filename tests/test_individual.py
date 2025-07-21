import unittest
import torch
from core.individual import Individual
from primitives.functions import add, multiply, sigmoid
from primitives.terminals import InputTerminal, IndexTerminal


class TestIndividual(unittest.TestCase):

    def test_gene_property_consistency(self):
        """Test that gene property always reflects current state."""
        ind = Individual(head_length=3, num_inputs=2, num_weights=4, num_biases=2)

        # Get initial gene
        gene1 = ind.gene.copy()

        # Modify internal state
        ind.weights[0] = 999.0
        ind.head[0] = multiply

        # Gene should reflect changes without manual sync
        gene2 = ind.gene
        self.assertNotEqual(gene1, gene2)
        self.assertEqual(999.0, gene2[len(ind.expression)])

    def test_chromosome_parsing(self):
        """Test creating individual from specific chromosome."""
        # Create a known chromosome
        chromosome = [
            # Head
            add, InputTerminal(0), IndexTerminal(0),
            # Tail (7 elements for head_length=3, max_arity=3)
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and biases
            1.0, 2.0, 3.0, 4.0,  # weights
            10.0, 20.0  # biases
        ]

        ind = Individual(
            head_length=3,
            num_inputs=2,
            num_weights=4,
            num_biases=2,
            chromosome=chromosome
        )

        # Verify parsing
        self.assertEqual(add, ind.head[0])
        self.assertIsInstance(ind.head[1], InputTerminal)
        self.assertIsInstance(ind.head[2], IndexTerminal)
        self.assertEqual(0, ind.head[2].index)  # @0
        self.assertEqual([1.0, 2.0, 3.0, 4.0], ind.weights)
        self.assertEqual([10.0, 20.0], ind.biases)

    def test_tail_length_calculation(self):
        """Test that tail length is correctly calculated."""
        # With max_arity=3 in our function set
        for head_length in [3, 5, 7]:
            ind = Individual(head_length=head_length, num_inputs=2,
                             num_weights=10, num_biases=5)
            expected_tail = head_length * (3 - 1) + 1
            self.assertEqual(expected_tail, ind.tail_length)
            self.assertEqual(expected_tail, len(ind.tail))

    def test_deep_copy(self):
        """Test that copy creates independent individual."""
        ind1 = Individual(head_length=3, num_inputs=2, num_weights=4, num_biases=2)
        ind1.fitness = 100.0

        ind2 = ind1.copy()

        # Should be different objects
        self.assertIsNot(ind1, ind2)
        self.assertIsNot(ind1.head, ind2.head)
        self.assertIsNot(ind1.weights, ind2.weights)

        # But same values
        self.assertEqual(ind1.fitness, ind2.fitness)
        self.assertEqual(ind1.gene, ind2.gene)

        # Modifications don't affect original
        ind2.weights[0] = 999.0
        self.assertNotEqual(ind1.weights[0], ind2.weights[0])

    def test_index_terminal_range(self):
        """Test that IndexTerminals are created within valid range."""
        head_length = 5
        ind = Individual(head_length=head_length, num_inputs=2,
                         num_weights=10, num_biases=5)

        # Check all IndexTerminals in head
        for symbol in ind.head:
            if isinstance(symbol, IndexTerminal):
                self.assertGreaterEqual(symbol.index, 0)
                self.assertLess(symbol.index, head_length)

    def test_no_index_terminals_in_tail(self):
        """Test that tail contains only InputTerminals."""
        ind = Individual(head_length=5, num_inputs=3, num_weights=10, num_biases=5)

        for symbol in ind.tail:
            self.assertIsInstance(symbol, InputTerminal)
            self.assertNotIsInstance(symbol, IndexTerminal)


if __name__ == '__main__':
    unittest.main()