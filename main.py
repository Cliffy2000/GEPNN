import torch
import numpy as np

from core.genome import GepnnChromosome
from core.network import NetworkCompiler

from primitives.primitives import get_primitive_set

from utils.output_utils import print_section_break


def main():
    print("GEPNN Checkpoint 1: Chromosome & Network test")

    primitive_set = get_primitive_set(num_inputs=2)

    chromosome = GepnnChromosome.create_random(
        primitive_set=primitive_set,
        head_length=5,
        num_genes=3
    )

    print(f"Created chromosome with {len(chromosome.genes)} genes")
    print(f"Chromosome: {chromosome}")

    compiler = NetworkCompiler()
    neural_network = compiler.compile(chromosome)

    test_input = torch.tensor([1.0, 2.0])  # TODO: set actual test input
    output = neural_network(test_input)

    print(f"Input: {test_input}")
    print(f"Output: {output}")
    print("Checkpoint 1 complete")
    print_section_break()


if __name__ == "__main__":
    main()
