import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qsimcirq

def add_depolarizing_noise(circuit, p):
    noisy_ops = []
    for moment in circuit:
        for op in moment:
            qubits = op.qubits
            noisy_ops.append(op)
            for qubit in qubits:
                noisy_ops.append(cirq.depolarize(p=p)(qubit))
    return cirq.Circuit(noisy_ops)

def calculate_probability_of_observing_all_zero_states(simulator, circuit):
    results = simulator.simulate(circuit)
    state_vector = results.final_state_vector
    return abs(state_vector[0])**2

if __name__ == "__main__":
    filename = 'Benchmarks/Simple/simple.qasm'
    
    # Load the QASM circuit into Cirq
    with open(filename, 'r') as f:
        qasm_str = f.read()
    circuit = circuit_from_qasm(qasm_str)

    # Add noise to the circuit
    circuit_with_noise = add_depolarizing_noise(circuit, 0.1)

    # Simulate the noisy circuit
    qsim_simulator = qsimcirq.QSimSimulator()
    results = qsim_simulator.simulate(circuit_with_noise)

    # Output the final state vector and the probability of all-zero state
    print("Final State Vector:", results.final_state_vector)
    probability_zero_state = calculate_probability_of_observing_all_zero_states(qsim_simulator, circuit_with_noise)
    print("Probability of observing all-zero state:", probability_zero_state)
