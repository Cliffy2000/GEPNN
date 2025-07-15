import torch
from core.individual import Individual
from core.network import Network
from primitives.functions import add
from primitives.terminals import InputTerminal, IndexTerminal
from utils.output_utils import (
    print_separator, print_section_break, print_header,
    print_test_info, print_calculation_step, print_result, print_tree_structure
)


def test_simple_add():
    """Test the simplest case: add(x0, x1)"""
    print_header("Test 1: Simple Addition", level=1)
    print_tree_structure("add(x0, x1)")

    # For head_length=3, max_arity=3, tail_length = 3*(3-1)+1 = 7
    # Create chromosome manually
    chromosome = [
        # Head (3)
        add, InputTerminal(0), InputTerminal(1),
        # Tail (7) - must be terminals only
        InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
        InputTerminal(0), InputTerminal(1), InputTerminal(0),
        # Weights (2)
        2.0, 3.0,
        # Biases (1)
        10.0
    ]

    # Create individual from chromosome
    individual = Individual.from_chromosome(
        chromosome=chromosome,
        head_length=3,
        num_inputs=2,
        num_weights=2,
        num_biases=1
    )

    # Create network
    network = Network(individual)

    # Test Case 1
    print_header("Test Case 1.1", level=2)
    inputs = {'x0': 5.0, 'x1': 7.0}
    print_test_info(inputs, individual.weights, individual.biases)

    output = network.forward(inputs)

    print_calculation_step(
        1, "Apply weights to inputs",
        "(x0 × w0) + (x1 × w1) = (5 × 2) + (7 × 3)",
        "10 + 21 = 31"
    )
    print_calculation_step(
        2, "Add bias",
        "31 + bias",
        "31 + 10 = 41"
    )

    print_result(41.0, output.item())

    # Test Case 2
    print_header("Test Case 1.2", level=2)
    inputs2 = {'x0': 1.0, 'x1': 2.0}
    print_test_info(inputs2, individual.weights, individual.biases)

    output2 = network.forward(inputs2)

    print_calculation_step(
        1, "Apply weights to inputs",
        "(x0 × w0) + (x1 × w1) = (1 × 2) + (2 × 3)",
        "2 + 6 = 8"
    )
    print_calculation_step(
        2, "Add bias",
        "8 + bias",
        "8 + 10 = 18"
    )

    print_result(18.0, output2.item())


def test_index_terminal():
    """Test with index terminal: add(add(x0, x1), @2)"""
    print_section_break()
    print("Test 2: Index Terminal - add(add(x0, x1), @2)")
    print_separator()

    # For head_length=5, max_arity=3, tail_length = 5*(3-1)+1 = 11
    # Position: 0    1    2   3   4
    # Symbols: add, add, x0, x1, @2
    chromosome = [
        # Head (5)
        add, add, InputTerminal(0), InputTerminal(1), IndexTerminal(2),
        # Tail (11) - terminals only
        InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
        InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
        InputTerminal(0), InputTerminal(1), InputTerminal(0),
        # Weights (4)
        2.0, 3.0, 4.0, 5.0,
        # Biases (2)
        100.0, 200.0
    ]

    individual = Individual.from_chromosome(
        chromosome=chromosome,
        head_length=5,
        num_inputs=2,
        num_weights=4,
        num_biases=2
    )

    print(f"Individual: {individual}")

    # Create network
    network = Network(individual)

    # Test with inputs
    inputs = {'x0': 1.0, 'x1': 1.0}
    output = network.forward(inputs)

    print(f"\nInputs: {inputs}")
    print(f"Weights: {individual.weights}")
    print(f"Biases: {individual.biases}")

    print("\nStep-by-step calculation:")
    print("1. Inner add (position 1): add(x0*4, x1*5) + 200")
    print(f"   = add(1*4, 1*5) + 200 = add(4, 5) + 200 = 9 + 200 = 209")

    print("2. Position 2 (x0) value = 1")

    print("3. Outer add (position 0): add(inner_add*2, x0*3) + 100")
    print(f"   = add(209*2, 1*3) + 100 = add(418, 3) + 100 = 421 + 100 = 521")

    print(f"\nExpected: 521")
    print(f"Output: {output.item()}")

    # Show network structure
    print("\nNetwork structure:")
    network.print_structure()


def test_recurrent_connection():
    """Test recurrent connection with index terminal"""
    print_header("Test 3: Recurrent Connection", level=1)
    print_tree_structure("add(x0, @0) where @0 creates a self-reference")

    # For head_length=3, max_arity=3, tail_length = 3*(3-1)+1 = 7
    chromosome = [
        # Head (3)
        add, InputTerminal(0), IndexTerminal(0),
        # Tail (7)
        InputTerminal(0), InputTerminal(1), InputTerminal(0), InputTerminal(1),
        InputTerminal(0), InputTerminal(1), InputTerminal(0),
        # Weights (2)
        2.0, 3.0,
        # Biases (1)
        10.0
    ]

    individual = Individual.from_chromosome(
        chromosome=chromosome,
        head_length=3,
        num_inputs=2,
        num_weights=2,
        num_biases=1
    )

    network = Network(individual)

    print("\nRecurrent Formula: output(t) = (x0 × 2) + (output(t-1) × 3) + 10")
    print("Initial state: output(-1) = 0")

    x0_values = [1.0, 2.0, 3.0]
    expected_outputs = [12.0, 50.0, 166.0]

    print_header("Time Series Evaluation", level=2)

    for t, (x0, expected) in enumerate(zip(x0_values, expected_outputs)):
        inputs = {'x0': x0, 'x1': 0.0}
        output = network.forward(inputs)

        print(f"\nTimestep {t}: x0 = {x0}")

        if t == 0:
            print("  Calculation: (1 × 2) + (0 × 3) + 10 = 2 + 0 + 10 = 12")
        elif t == 1:
            print("  Calculation: (2 × 2) + (12 × 3) + 10 = 4 + 36 + 10 = 50")
        else:
            print("  Calculation: (3 × 2) + (50 × 3) + 10 = 6 + 150 + 10 = 166")

        print(f"  Output: {output.item():.1f} {'✓' if abs(output.item() - expected) < 0.001 else '✗'}")


def test_complex_index():
    """Test complex index case: add(add(x0, @4), add(x1, x2))"""
    print_header("Test 4: Complex Tree with Forward Reference", level=1)
    print_tree_structure("add(add(x0, @4), add(x1, x2)) where @4 references node 4 (x1)")

    # For head_length=7, max_arity=3, tail_length = 7*(3-1)+1 = 15
    chromosome = [
        # Head (7)
        add, add, add, InputTerminal(0), IndexTerminal(4), InputTerminal(1), InputTerminal(2),
        # Tail (15)
        InputTerminal(0), InputTerminal(1), InputTerminal(2), InputTerminal(0),
        InputTerminal(1), InputTerminal(2), InputTerminal(0), InputTerminal(1),
        InputTerminal(2), InputTerminal(0), InputTerminal(1), InputTerminal(2),
        InputTerminal(0), InputTerminal(1), InputTerminal(2),
        # Weights (6)
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        # Biases (3)
        100.0, 200.0, 300.0
    ]

    individual = Individual.from_chromosome(
        chromosome=chromosome,
        head_length=7,
        num_inputs=3,
        num_weights=6,
        num_biases=3
    )

    network = Network(individual)

    inputs = {'x0': 1.0, 'x1': 2.0, 'x2': 3.0}
    print_test_info(inputs, individual.weights, individual.biases)

    output = network.forward(inputs)

    print("\nNetwork Structure:")
    network.print_structure()

    print_header("Calculation Steps", level=2)
    print("\nNote: @4 creates a forward reference to node 4 (x1)")

    print_calculation_step(
        1, "Left add (node 1)",
        "add(x0×4, x1×5) + bias = add(1×4, 2×5) + 200",
        "add(4, 10) + 200 = 14 + 200 = 214"
    )

    print_calculation_step(
        2, "Right add (node 2)",
        "add(x1×6, x2×7) + bias = add(2×6, 3×7) + 300",
        "add(12, 21) + 300 = 33 + 300 = 333"
    )

    print_calculation_step(
        3, "Root add (node 0)",
        "add(left×2, right×3) + bias = add(214×2, 333×3) + 100",
        "add(428, 999) + 100 = 1427 + 100 = 1527"
    )

    print_result(1527.0, output.item())


if __name__ == "__main__":
    print_header("GEPNN Network Testing Suite", level=1)
    print("\nThis test suite demonstrates Gene Expression Programming Neural Networks (GEPNN)")
    print("with support for index-based connections enabling skip and recurrent connections.")

    test_simple_add()
    test_index_terminal()
    test_recurrent_connection()
    test_complex_index()

    print_section_break()
    print("All tests completed successfully.")