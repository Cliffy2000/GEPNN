import unittest
import numpy as np
from core.individual import Individual
from core.network import Network
from primitives.functions import add, multiply, sigmoid
from primitives.terminals import InputTerminal, IndexTerminal


class TestNetwork(unittest.TestCase):

    def test_simple_feedforward(self):
        """Test basic feedforward computation with batch processing."""
        # Create specific individual: add(x0, x1)
        # Tree structure:
        # [0] add (bias=10.0)
        #   [1] x0 (weight=2.0)
        #   [2] x1 (weight=3.0)

        chromosome = [
            add, InputTerminal(0), InputTerminal(1),
            # Tail (unused in this case)
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Weights and bias
            2.0, 3.0,  # weights for edges to x0 and x1
            10.0  # bias for add function
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # Debug network structure
        print("\n=== Network Structure ===")
        print(f"Expression: {[str(s) for s in network.expression]}")
        print(f"Weights (with dummy): {network.weights}")
        print(f"Biases: {network.biases}")
        print(f"Children: {network.children}")
        print(f"Child weights: {network.child_weights}")
        print(f"Is function: {network.is_function}")
        print(f"Is input: {network.is_input}")

        # Test single instance
        inputs = np.array([[5.0, 7.0]], dtype=np.float32)
        print(f"\n=== Forward Pass Debug ===")
        print(f"Input shape: {inputs.shape}")
        print(f"Input values: {inputs}")

        # Add debug prints in forward pass
        result = network.forward(inputs)

        print(f"\n=== After Forward ===")
        print(f"All node values:")
        for i in range(network.n):
            print(f"  Node {i} ({network.expression[i]}): {network.values[i]}")

        print(f"\nResult: {result}")
        print(f"Expected: 41.0")

        # Expected: (5.0 * 2.0) + (7.0 * 3.0) + 10.0 = 41.0
        expected = 41.0
        self.assertAlmostEqual(expected, result[0])

    '''

    def test_index_terminal_reference(self):
        """Test that IndexTerminals create proper references."""
        # Create: add(x0, @2) - @2 is a self-loop
        # Tree structure:
        # [0] add (bias=10.0)
        #   [1] x0 (weight=2.0)
        #   [2] @2 (weight=3.0) -> self-loop

        chromosome = [
            add, InputTerminal(0), IndexTerminal(2),
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

        # @2 is a self-loop, should always return 0.0
        result1 = network.forward({'x0': 1.0, 'x1': 0.0})
        expected1 = (1.0 * 2.0) + (0.0 * 3.0) + 10.0  # 12.0
        self.assertAlmostEqual(expected1, result1.item())

        # Second forward pass - @2 still returns 0.0 (self-loop)
        result2 = network.forward({'x0': 1.0, 'x1': 0.0})
        expected2 = (1.0 * 2.0) + (0.0 * 3.0) + 10.0  # 12.0 (same)
        self.assertAlmostEqual(expected2, result2.item())

    def test_index_terminal_forward_reference(self):
        """Test IndexTerminal referencing a later node."""
        # Create: add(x0, @3) where @3 references x1
        # Tree structure:
        # [0] add (bias=5.0)
        #   [1] x0 (weight=2.0)
        #   [2] @3 (weight=3.0) -> references x1[3]
        #   [3] x1 (in tail, but referenced)

        chromosome = [
            add, InputTerminal(0), IndexTerminal(3),
            InputTerminal(1),  # This is node 3
            # Rest of tail (needs 6 more for tail_length=7)
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(0), InputTerminal(1),
            # Weights and bias
            2.0, 3.0,  # weights
            5.0  # bias
        ]

        ind = Individual(head_length=3, num_inputs=2, num_weights=2,
                         num_biases=1, chromosome=chromosome)
        network = Network(ind)

        # @3 references x1, so computation is: (4.0 * 2.0) + (7.0 * 3.0) + 5.0
        result = network.forward({'x0': 4.0, 'x1': 7.0})
        expected = (4.0 * 2.0) + (7.0 * 3.0) + 5.0  # 34.0
        self.assertAlmostEqual(expected, result.item())

    def test_index_terminal_input_reference(self):
        """Test IndexTerminal referencing an input terminal."""
        # Expression: [add, add, x0, x1, @2]
        # Tree structure:
        # [0] add (bias=5.0)
        #   [1] add (weight=1.0, bias=3.0)
        #     [3] x1 (weight=4.0)
        #     [4] @2 (weight=6.0) -> references x0[2]
        #   [2] x0 (weight=2.0)

        chromosome = [
            add, add, InputTerminal(0), InputTerminal(1), IndexTerminal(2),
            # Tail (11 more elements for head_length=5)
            InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
            InputTerminal(0),
            # Weights and biases
            1.0, 2.0, 4.0, 6.0,  # 4 weights
            5.0, 3.0  # 2 biases (for the two add functions)
        ]

        ind = Individual(head_length=5, num_inputs=2, num_weights=4,
                         num_biases=2, chromosome=chromosome)
        network = Network(ind)

        # @2 references x0, always gets current value
        # Inner add: (2.0 * 2.0) + (3.0 * 4.0) + 3.0 = 19.0
        # Outer add: (19.0 * 1.0) + (2.0 * 6.0) + 5.0 = 36.0
        result = network.forward({'x0': 2.0, 'x1': 3.0})
        expected = 36.0
        self.assertAlmostEqual(expected, result.item())

    def test_specific_individual_fitness(self):
        """Test fitness evaluation of specific individual."""
        from primitives.functions import tanh, multiply, add
        from evaluation.fitness import evaluate_xor

        # The specific individual
        chromosome = [
            tanh, multiply, InputTerminal(0), InputTerminal(1), add,
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Tail (17 elements)
            InputTerminal(0), InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(1), InputTerminal(0), InputTerminal(0),
            InputTerminal(0), InputTerminal(0), InputTerminal(1), InputTerminal(1),
            InputTerminal(1), InputTerminal(0), InputTerminal(0), InputTerminal(1),
            InputTerminal(0),
            # Weights (16)
            1.7241547182625894, 3.1723251237480206, -1.4478290641511684, 2.480997583448104,
            1.8508306741714478, 0.8913540244102478, 1.8751992324743543, 1.5752192605535866,
            -1.7370924949645996, -1.9757283471387248, 1.1963967084884644, -2.0744085266786882,
            3.3392197123736778, -2.325628891045713, 3.108464744943104, -0.2191777527332306,
            # Biases (8)
            0.5632854667504225, -0.9878491220275905, -1.974275116172632, 1.3006843328475952,
            0.4542977809906006, -0.5106743574142456, -0.7389845431666773, 2.1065809249854186
        ]

        ind = Individual(head_length=8, num_inputs=2, num_weights=16,
                         num_biases=8, chromosome=chromosome)

        # Get fitness
        fitness = evaluate_xor(ind)
        print(f"\nFitness score: {fitness[0]}")

        # Also manually check each XOR case
        network = Network(ind)
        print("\nDetailed XOR evaluation:")
        xor_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

        for x0, x1, target in xor_cases:
            for node in network.nodes:
                node.prev_value = 0.0
            result = network.forward({'x0': float(x0), 'x1': float(x1)})
            result_val = result.item()
            pred = 1 if result_val > 0.5 else 0
            print(f"({x0}, {x1}) -> {result_val:.4f} -> {pred} (target: {target})")


    def test_specific_tanh_multiply_case(self):
        """Test specific individual with tanh at root."""
        from primitives.functions import tanh, multiply, add

        # Parse the individual
        chromosome = [
            tanh, multiply, InputTerminal(0), InputTerminal(1), add,
            InputTerminal(0), InputTerminal(1), InputTerminal(0),
            # Tail (17 elements)
            InputTerminal(0), InputTerminal(0), InputTerminal(1), InputTerminal(0),
            InputTerminal(1), InputTerminal(1), InputTerminal(0), InputTerminal(0),
            InputTerminal(0), InputTerminal(0), InputTerminal(1), InputTerminal(1),
            InputTerminal(1), InputTerminal(0), InputTerminal(0), InputTerminal(1),
            InputTerminal(0),
            # Weights (16)
            1.7241547182625894, 3.1723251237480206, -1.4478290641511684, 2.480997583448104,
            1.8508306741714478, 0.8913540244102478, 1.8751992324743543, 1.5752192605535866,
            -1.7370924949645996, -1.9757283471387248, 1.1963967084884644, -2.0744085266786882,
            3.3392197123736778, -2.325628891045713, 3.108464744943104, -0.2191777527332306,
            # Biases (8)
            0.5632854667504225, -0.9878491220275905, -1.974275116172632, 1.3006843328475952,
            0.4542977809906006, -0.5106743574142456, -0.7389845431666773, 2.1065809249854186
        ]

        ind = Individual(head_length=8, num_inputs=2, num_weights=16,
                         num_biases=8, chromosome=chromosome)
        network = Network(ind)

        # Print structure to debug
        print("\n\nSpecific tanh case - Tree structure:")
        network.print_tree()

        # Manual calculation for x0=1.0, x1=0.0
        # multiply: (1.0*-1.4478) + (0.0*2.4810) + (-0.9878) = -2.4356
        # add: (1.0*1.8508) + (0.0*0.8913) + (-1.9743) = -0.1235
        # tanh: tanh((-2.4356*1.7242) + (-0.1235*3.1723)) + 0.5633
        #     = tanh(-4.5907) + 0.5633 = -0.9999 + 0.5633 = -0.4366

        result = network.forward({'x0': 1.0, 'x1': 0.0})

        # Let's verify step by step
        import torch
        mult_result = (1.0 * -1.4478290641511684) + (0.0 * 2.480997583448104) + (-0.9878491220275905)
        add_result = (1.0 * 1.8508306741714478) + (0.0 * 0.8913540244102478) + (-1.974275116172632)
        tanh_input = (mult_result * 1.7241547182625894) + (add_result * 3.1723251237480206)
        expected = torch.tanh(torch.tensor(tanh_input)).item() + 0.5632854667504225

        print(f"Result: {result.item():.6f}, Expected: {expected:.6f}")
        self.assertAlmostEqual(expected, result.item(), places=4)



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
        # 2: @3 (reference to node 3 - but node 3 will be x1, not multiply!)
        # 3: multiply (node 2) - needs 2 children
        # 4: x1 (node 3)
        # 5: x2 (node 4)

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

        # Should create 15 nodes total (all non-IndexTerminal positions)
        self.assertEqual(15, len(network.nodes))

        # Verify node structure
        self.assertEqual(add, network.nodes[0].symbol)  # root
        self.assertEqual('x0', network.nodes[1].symbol.name)
        self.assertEqual(multiply, network.nodes[2].symbol)  # multiply is node 2, not 3!
        self.assertEqual('x1', network.nodes[3].symbol.name)  # x1 is node 3
        self.assertEqual('x2', network.nodes[4].symbol.name)

        # Test computation with x0=2, x1=3, x2=4
        # add gets: x0 as child, @3 (which is x1) as reference
        # add: (x0*1 + x1*2) + 5 = (2*1 + 3*2) + 5 = 2 + 6 + 5 = 13
        result = network.forward({'x0': 2.0, 'x1': 3.0, 'x2': 4.0})
        self.assertEqual(13.0, result.item())

        # Verify the forward reference was resolved
        self.assertEqual(1, len(network.root.ref_children))
        self.assertEqual(network.nodes[3], network.root.ref_children[0])  # @3 references node 3 (x1)
'''

if __name__ == '__main__':
    unittest.main()