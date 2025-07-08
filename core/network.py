import torch
import torch.nn as nn
import geppy as gep


class CompiledNetwork(nn.Module):
    def __init__(self, compiled_genes):
        super().__init__()
        self.compiled_genes = compiled_genes

    def forward(self, inputs):
        gene_outputs = []
        for gene_func in self.compiled_genes:
            try:
                output = gene_func(*inputs)
                if isinstance(output, (int, float)):
                    output = torch.tensor(float(output))
                gene_outputs.append(output)
            except Exception as e:
                print(f"Gene execution error: {e}")
                gene_outputs.append(torch.tensor(0.0))

        if len(gene_outputs) == 1:
            return gene_outputs[0]
        else:
            return torch.stack(gene_outputs)


class NetworkCompiler:
    def compile(self, chromosome):
        compiled_genes = []

        # Get primitive set from chromosome
        pset = chromosome.primitive_set

        for gene in chromosome.genes:
            print(f"Gene: {gene}")

            try:
                # Create a single-gene chromosome for this gene
                single_gene_chromosome = gep.Chromosome.from_genes([gene])

                # Use geppy's proper compilation method on the chromosome
                gene_func = gep.compile_(single_gene_chromosome, pset)
                print(f"Compiled function: {gene_func}")
                compiled_genes.append(gene_func)

            except Exception as e:
                print(f"Compilation error: {e}")
                compiled_genes.append(lambda *args: torch.tensor(0.0))

        return CompiledNetwork(compiled_genes)