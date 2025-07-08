import geppy as gep
from .functions import get_functions
from .terminals import get_input_names, get_ephemeral_terminals


def get_primitive_set(num_inputs):
    input_names = get_input_names(num_inputs)
    pset = gep.PrimitiveSet('neural_primitives', input_names=input_names)

    functions = get_functions()
    for func, arity in functions:
        pset.add_function(func, arity)

    ephemeral_terminals = get_ephemeral_terminals()
    for name, generator in ephemeral_terminals:
        pset.add_ephemeral_terminal(name, generator)

    return pset