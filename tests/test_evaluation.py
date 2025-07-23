import unittest
import torch
import random
from core.individual import Individual
from core.network import Network

from tqdm import tqdm


class TestRandomNetworkEvaluation(unittest.TestCase):

    def test_random_network_evaluation(self):
        """Test that 2000 random individuals can be evaluated without errors."""
        head_length = 10
        num_inputs = 4
        num_weights = 50
        num_biases = 10
        num_individuals = 20000

        successful_evaluations = 0

        for i in tqdm(range(num_individuals), desc="Testing random chromosomes"):
            # Create random individual
            individual = Individual(
                head_length=head_length,
                num_inputs=num_inputs,
                num_weights=num_weights,
                num_biases=num_biases
            )

            # Create network
            try:
                network = Network(individual)
            except Exception as e:
                self.fail(f"Failed to create network for individual {i}: {str(e)}")

            # Generate random inputs
            inputs = {
                f'x{j}': random.uniform(-10.0, 10.0)
                for j in range(num_inputs)
            }

            # Evaluate network
            try:
                output = network.forward(inputs)

                # Verify output is valid
                self.assertIsInstance(output, torch.Tensor)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

                successful_evaluations += 1

            except Exception as e:
                self.fail(f"Failed to evaluate individual {i}: {str(e)}\n"
                          f"Individual: {individual}")

        # All individuals should evaluate successfully
        self.assertEqual(successful_evaluations, num_individuals)
        print(f"\nSuccessfully evaluated {successful_evaluations} random individuals")


if __name__ == '__main__':
    unittest.main()