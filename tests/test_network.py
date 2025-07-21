import unittest
import torch
from core.individual import Individual
from core.network import Network
from primitives.functions import add, multiply, sigmoid
from primitives.terminals import InputTerminal, IndexTerminal


class TestNetwork(unittest.TestCase):

    def test_simple_feedforward(self):
        """Test basic feedforward computation."""
        # Create specific individual: add(x0, x1)
        chromosome = [
            add, InputTerminal(0), InputTerminal(1),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and bias
            2.0, 3.0,  # weights
            10.0  # bias
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # Test computation
        result = network.forward({'x0': 5.0, 'x1': 7.0})
        expected = (5.0 * 2.0) + (7.0 * 3.0) + 10.0  # 41.0
        self.assertEqual(expected, result.item())

    def test_index_terminal_reference(self):
        """Test that IndexTerminals create proper references."""
        # Create: add(x0, @0) - self reference
        chromosome = [
            add, InputTerminal(0), IndexTerminal(0),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and bias
            2.0, 3.0,  # weights
            10.0  # bias
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # First forward pass - @0 should use prev_value (0.0)
        result1 = network.forward({'x0': 1.0, 'x1': 0.0})
        expected1 = (1.0 * 2.0) + (0.0 * 3.0) + 10.0  # 12.0
        self.assertEqual(expected1, result1.item())

        # Second forward pass - @0 should use previous output
        result2 = network.forward({'x0': 1.0, 'x1': 0.0})
        expected2 = (1.0 * 2.0) + (12.0 * 3.0) + 10.0  # 48.0
        self.assertEqual(expected2, result2.item())

    def test_forward_reference(self):
        """Test forward references work correctly."""
        # Create: add(add(x0, x1), @2) where @2 refers to node 2 (x0)
        chromosome = [
            add, add, InputTerminal(0), InputTerminal(1), IndexTerminal(2),
            # Tail (11 elements)
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and biases
            1.0, 1.0, 1.0, 1.0,  # weights
            0.0, 0.0  # biases
        ]

        ind = Individual(head_length=5, num_inputs=2, num_weights=4,
                         num_biases=2, chromosome=chromosome)
        network = Network(ind)

        # With all weights=1 and biases=0:
        # inner add: x0 + x1 = 2 + 3 = 5
        # outer add: inner + x0 = 5 + 2 = 7
        result = network.forward({'x0': 2.0, 'x1': 3.0})
        self.assertEqual(7.0, result.item())

    def test_node_count_with_index_terminals(self):
        """Test that IndexTerminals don't create nodes."""
        # Expression with 2 functions, 2 inputs, 1 index terminal
        chromosome = [
            add, add, InputTerminal(0), InputTerminal(1), IndexTerminal(2),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and biases
            1.0, 1.0, 1.0, 1.0,
            0.0, 0.0
        ]

        ind = Individual(head_length=5, num_inputs=2, num_weights=4,
                         num_biases=2, chromosome=chromosome)
        network = Network(ind)

        # Should have 4 nodes: add, add, x0, x1 (not 5)
        self.assertEqual(15, len(network.nodes))

    def test_input_validation(self):
        """Test that missing inputs raise appropriate errors."""
        ind = Individual(head_length=3, num_inputs=3, num_weights=4, num_biases=2)
        network = Network(ind)

        # Missing input should raise error
        with self.assertRaises(ValueError) as context:
            network.forward({'x0': 1.0, 'x1': 2.0})  # Missing x2
        self.assertIn("Missing input x2", str(context.exception))

        # List with insufficient inputs
        with self.assertRaises(ValueError) as context:
            network.forward([1.0, 2.0])  # Need 3 inputs
        self.assertIn("Expected 3 inputs", str(context.exception))

    def test_weight_bias_assignment(self):
        """Test correct assignment of weights and biases."""
        # Create known structure
        chromosome = [
            multiply, add, InputTerminal(0), InputTerminal(1), InputTerminal(2),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and biases
            2.0, 3.0, 4.0, 5.0, 6.0,  # weights
            10.0, 20.0  # biases for multiply and add
        ]

        ind = Individual(head_length=5, num_inputs=3, num_weights=5,
                         num_biases=2, chromosome=chromosome)
        network = Network(ind)

        # Check active parameters
        active_weights, active_biases = network.get_active_parameters()
        self.assertEqual(2, active_biases)  # Two function nodes

        # Verify computation uses correct weights/biases
        result = network.forward({'x0': 1.0, 'x1': 1.0, 'x2': 1.0})
        # add: (x0*4 + x1*5) + 20 = (1*4 + 1*5) + 20 = 29
        # multiply: (add*2 * x2*3) + 10 = (29*2 * 1*3) + 10 = 184
        self.assertEqual(184.0, result.item())

    def test_simple_recurrence(self):
        """Test basic recurrent connection."""
        # Create: add(x0, @0) - output feeds back to itself
        chromosome = [
            add, InputTerminal(0), IndexTerminal(0),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and bias
            1.0, 0.5,  # weights: x0 weight=1.0, @0 weight=0.5
            0.0  # bias
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # Time step 1: @0 uses prev_value=0
        # add(1*1.0, 0*0.5) + 0 = 1.0
        result1 = network.forward({'x0': 1.0, 'x1': 0.0})
        self.assertEqual(1.0, result1.item())

        # Time step 2: @0 uses prev_value=1.0
        # add(1*1.0, 1.0*0.5) + 0 = 1.5
        result2 = network.forward({'x0': 1.0, 'x1': 0.0})
        self.assertEqual(1.5, result2.item())

        # Time step 3: @0 uses prev_value=1.5
        # add(1*1.0, 1.5*0.5) + 0 = 1.75
        result3 = network.forward({'x0': 1.0, 'x1': 0.0})
        self.assertEqual(1.75, result3.item())


    def test_simple_recurrent_network(self):
        """Test a simple recurrent network with manually verifiable calculations."""
        # Create: add(x0, @0) - output feeds back to itself
        # This creates a simple accumulator
        chromosome = [
            add, InputTerminal(0), IndexTerminal(0),
            # Tail
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and bias
            1.0, 2.0,  # weights: x0 weight=1.0, @0 weight=2.0
            5.0  # bias = 5.0
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # First timestep: input x0=3
        # @0 uses prev_value=0 (initial state)
        # output = (3*1) + (0*2) + 5 = 8
        result1 = network.forward({'x0': 3.0, 'x1': 0.0})
        self.assertEqual(8.0, result1.item())

        # Second timestep: input x0=3
        # @0 uses previous output = 8
        # output = (3*1) + (8*2) + 5 = 3 + 16 + 5 = 24
        result2 = network.forward({'x0': 3.0, 'x1': 0.0})
        self.assertEqual(24.0, result2.item())

        # Third timestep: input x0=1
        # @0 uses previous output = 24
        # output = (1*1) + (24*2) + 5 = 1 + 48 + 5 = 54
        result3 = network.forward({'x0': 1.0, 'x1': 0.0})
        self.assertEqual(54.0, result3.item())

    def test_forward_reference_feedforward(self):
        """Test forward reference where IndexTerminal references a not-yet-created node."""
        # Create expression where root references @3, which will be created later
        # Expression parsing:
        # 0: add (root) - needs 2 children
        # 1: x0 (first child of root)
        # 2: @3 (reference to future node 3)
        # 3: multiply (created to satisfy @3 reference) - needs 2 children
        # 4: x1 (first child of multiply)
        # 5: x2 (second child of multiply)

        chromosome = [
            # Head
            add, InputTerminal(0), IndexTerminal(3), multiply, InputTerminal(1),
            # Tail (11 elements for head_length=5)
            InputTerminal(2), InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(0), InputTerminal(0),
            # Weights
            1.0, 2.0, 3.0, 4.0,
            # Biases
            5.0, 10.0
        ]

        ind = Individual(head_length=5, num_inputs=3, num_weights=4,
                         num_biases=2, chromosome=chromosome)
        network = Network(ind)

        # Should create 6 nodes total
        self.assertEqual(15, len(network.nodes))

        # Verify node structure
        self.assertEqual(add, network.nodes[0].symbol)  # root
        self.assertEqual('x0', network.nodes[1].symbol.name)
        self.assertEqual(multiply, network.nodes[3].symbol)
        self.assertEqual('x1', network.nodes[4].symbol.name)
        self.assertEqual('x2', network.nodes[5].symbol.name)

        # Test computation with x0=2, x1=3, x2=4
        # multiply: (x1*3 + x2*4) + 10 = (3*3 + 4*4) + 10 = 9 + 16 + 10 = 35
        # add: (x0*1 + @3*2) + 5 = (2*1 + 35*2) + 5 = 2 + 70 + 5 = 77
        result = network.forward({'x0': 2.0, 'x1': 3.0, 'x2': 4.0})
        self.assertEqual(77.0, result.item())

        # Verify the forward reference was resolved
        self.assertEqual(1, len(network.root.ref_children))
        self.assertEqual(network.nodes[3], network.root.ref_children[0])

if __name__ == '__main__':
    unittest.main()