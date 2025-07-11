import torch
from core.individual import GepnnIndividual
from core.network import GepnnNetwork
from primitives.functions import add
from utils.output_utils import print_separator, print_section_break


def test_simple_add():
    """Test single add node with fixed values."""
    print_separator()
    print("Simple Add Test")
    print_separator()

    # Create individual
    individual = GepnnIndividual(
        head_length=3,
        num_inputs=2,
        num_weights=2,
        num_biases=1
    )

    # Expression: add(x0, x1)
    individual.gene[0] = add
    individual.gene[1] = 'x0'
    individual.gene[2] = 'x1'

    # Fixed weights and bias
    individual.weights = [2.0, 3.0]  # w0, w1
    individual.biases = [10.0]  # b0

    print(f"Full individual: {individual}")
    print(f"Head: {individual.get_head()}")
    print(f"Tail: {individual.get_tail()}")
    print(f"All weights: {individual.weights}")
    print(f"All biases: {individual.biases}")

    # Create network and test
    network = GepnnNetwork(individual)

    # Test with x0=5, x1=7
    inputs = {'x0': torch.tensor([5.0]), 'x1': torch.tensor([7.0])}
    output = network.forward(inputs)

    print(f"\nInputs: x0=5, x1=7")
    print(f"Output: {output.item()}")


def test_nested_add():
    """Test nested add nodes with fixed values."""
    print_section_break()
    print("Nested Add Test")
    print_separator()

    individual = GepnnIndividual(
        head_length=4,
        num_inputs=3,
        num_weights=4,
        num_biases=2
    )

    # Expression: add(add(x0, x1), x2)
    individual.gene[0] = add  # root
    individual.gene[1] = add  # left child
    individual.gene[2] = 'x2'  # right child
    individual.gene[3] = 'x0'  # left-left
    individual.gene[4] = 'x1'  # left-right

    # Fixed weights and biases
    individual.weights = [2.0, 3.0, 4.0, 5.0]  # w0, w1, w2, w3
    individual.biases = [100.0, 200.0]  # b0, b1

    print(f"Full individual: {individual}")
    print(f"Head: {individual.get_head()}")
    print(f"Tail: {individual.get_tail()}")
    print(f"All weights: {individual.weights}")
    print(f"All biases: {individual.biases}")

    # Create network and test
    network = GepnnNetwork(individual)
    print("\nNetwork structure:")
    print(network.visualize_structure())

    # Test with simple values
    inputs = {'x0': torch.tensor([1.0]), 'x1': torch.tensor([1.0]), 'x2': torch.tensor([1.0])}
    output = network.forward(inputs)

    print(f"\nInputs: x0=1, x1=1, x2=1")
    print(f"Output: {output.item()}")


if __name__ == "__main__":
    test_simple_add()
    test_nested_add()