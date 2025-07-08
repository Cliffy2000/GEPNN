import geppy as gep
from typing import List


class GepnnChromosome:
    def __init__(self, genes: List[gep.Gene], primitive_set):
        self.genes = genes
        self.primitive_set = primitive_set

    @classmethod
    def create_random(cls, primitive_set, head_length: int, num_genes: int):
        genes = []
        for _ in range(num_genes):
            gene = gep.Gene(pset=primitive_set, head_length=head_length)
            genes.append(gene)
        return cls(genes, primitive_set)

    def __str__(self):
        gene_strs = [str(gene) for gene in self.genes]
        return f"Chromosome({', '.join(gene_strs)})"

    def __len__(self):
        return len(self.genes)