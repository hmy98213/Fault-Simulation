import cirq

# Create a noisy quantum circuit
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(cirq.rx(0.5).on(qubit))

# Create a simulator object
simulator = cirq.MPSSimulator()

# Set the noise model for the simulator
noise_model = cirq.ConstantQubitNoiseModel(qubit_noise=cirq.DepolarizingChannel(p=0.05))
simulator.set_noise(noise_model)

# Simulate the circuit
result = simulator.simulate(circuit)

# Get the results of the simulation
print(result)
