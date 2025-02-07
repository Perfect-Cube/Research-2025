import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt

# Quantum Phase Estimation Simulation
# -------------------------------------
# In many quantum sensing techniques, a small physical deformation can induce a phase shift (φ)
# in an interferometric setup. Here we simulate the estimation of such a phase shift using QPE.
#
# We assume we have a unitary operator U such that:
#   U|ψ> = exp(2πi φ)|ψ>
#
# Our goal is to estimate the phase φ.
#
# We use:
#   - n_count counting qubits (to register the phase estimate)
#   - 1 eigenstate qubit (assumed to be in an eigenstate |ψ> of U)
#
# The algorithm:
#   1. Prepare the eigenstate |ψ> (here we choose |1> for simplicity).
#   2. Apply Hadamard gates to the counting qubits.
#   3. For each counting qubit, apply controlled-U^(2^j) (here U is implemented as a phase gate).
#   4. Perform the inverse Quantum Fourier Transform (QFT) on the counting qubits.
#   5. Measure the counting qubits to obtain an estimate of φ.
#
# The estimated phase is given by the measured bitstring interpreted as an integer divided by 2^n_count.

# Set the true phase value to be estimated (must be in [0,1])
phi = 0.125  # example value

# Number of counting qubits (increases resolution of the estimate)
n_count = 3

# Create a quantum circuit with n_count counting qubits and 1 eigenstate qubit,
# plus n_count classical bits for measurement.
qc = QuantumCircuit(n_count + 1, n_count)

# Step 1: Prepare the eigenstate |ψ>
# Here we assume |ψ> = |1>, which is an eigenstate of a phase gate.
qc.x(n_count)

# Step 2: Apply Hadamard gates to all counting qubits.
for q in range(n_count):
    qc.h(q)

# Step 3: Apply controlled-U operations.
# In our simulation, U is a phase gate RZ(2πφ) acting on the eigenstate qubit.
# We apply controlled-U^(2^j) for j = 0 to n_count-1.
for j in range(n_count):
    repetitions = 2 ** j
    # Apply a controlled phase (cp) gate with angle = 2π * φ * (2^j)
    qc.cp(2 * np.pi * phi * repetitions, j, n_count)

# Step 4: Apply the inverse Quantum Fourier Transform on the counting qubits.
# Qiskit provides a QFT circuit; we specify inverse=True to perform the inverse QFT.
qc.append(QFT(n_count, inverse=True, do_swaps=True).to_instruction(), range(n_count))

# Step 5: Measure the counting qubits.
for i in range(n_count):
    qc.measure(i, i)

# Draw the circuit
print("Quantum Phase Estimation Circuit:")
print(qc.draw())

# Simulate the circuit using Qiskit's Aer simulator.
backend = Aer.get_backend('qasm_simulator')
shots = 1024
result = execute(qc, backend, shots=shots).result()
counts = result.get_counts()

# Display the measurement results.
print("\nMeasurement results (counts):")
print(counts)
plot_histogram(counts)
plt.show()

# Post-process the measurement results to estimate φ.
# The measured bitstring is interpreted as an integer x; then φ_estimate = x / 2^n_count.
# We take the most frequent measurement outcome.
most_common = max(counts, key=counts.get)
phase_estimate = int(most_common, 2) / (2 ** n_count)
print(f"\nEstimated phase: {phase_estimate} (True phase: {phi})")



# How It Works

#     Circuit Setup:
#     The circuit has three counting qubits (for a resolution of 1/8) and one eigenstate qubit prepared in |1⟩.
#     Controlled Operations:
#     Each counting qubit controls a phase rotation (a controlled‑U gate) raised to a power that doubles with each qubit (2⁰, 2¹, 2²).
#     Inverse QFT:
#     The inverse QFT transforms the accumulated phases into a binary number that estimates the phase φ.
#     Measurement:
#     The counting qubits are measured, and the most common output is used to compute the estimated phase.

# This code demonstrates a fundamental quantum technique often used in high-precision sensing and material analysis applications, where even minute phase shifts (caused by, for example, deformations in materials) can be accurately estimated.
