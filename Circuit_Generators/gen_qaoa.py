
import cirq
import networkx as nx
import numpy as np

from recirq.qaoa.circuit_structure import validate_well_structured
from recirq.qaoa.problem_circuits import get_routed_hardware_grid_circuit, \
    get_compiled_hardware_grid_circuit, get_generic_qaoa_circuit
from recirq.qaoa.problems import random_plus_minus_1_weights, HardwareGridProblem

def test_get_generic_qaoa_circuit():
    
    problem_graph = nx.gnp_random_graph(n=6, p=0.5, seed=52)
    nx.set_edge_attributes(problem_graph, 1, name='weight')
    qubits = cirq.GridQubit.rect(2, 3, 10, 10)

    circuit = get_generic_qaoa_circuit(problem_graph=problem_graph,
                                       qubits=qubits,
                                       gammas=[0.1, 0.2],
                                       betas=[0.3, 0.4])
    # Hadamard, then p = 2 problem+driver unitaries
    assert len(circuit) == 1 + (2 * 2)