pip install qiskit hyperopt scikit-learn

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Parameters and Setup
# ---------------------------
n_qubits = 4
entanglement_level = 3  # number of CNOT gates in the entangling block

# Define a fixed ordering of possible directed edges.
# (This ordering is one possible choice; the QES paper proposes one encoding.)
possible_edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 3),
    (3, 2), (3, 1),
    (2, 1),
    (3, 0), (2, 0), (1, 0)
]

# For demonstration we fix the “learnable” weights to a constant.
# In a full implementation these would be trained (e.g., via gradient descent).
fixed_weights = np.zeros(n_qubits)

# ---------------------------
# Data Preparation
# ---------------------------
# Use the Iris dataset but filter to only two classes for binary classification.
data = load_iris()
# Select only classes 0 and 1
mask = data.target != 2
X = data.data[mask]
y = data.target[mask]

# Normalize features to [0, π] (so rotation angles are reasonable)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)
# For our circuit we assume one feature per qubit.
X_use = X_scaled[:, :n_qubits]

# Split into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_use, y, test_size=0.2, random_state=42)

# ---------------------------
# Quantum Circuit Construction
# ---------------------------
def build_quantum_circuit(feature_vector, weights, genotype):
    """
    Build a quantum circuit (ansatz) with three blocks:
      1. Feature embedding: Apply RY rotations using the input features.
      2. Entanglement: Apply CNOT gates as given by the candidate genotype.
      3. Final rotation: Apply RY gates using (fixed) weights.
    
    Args:
      feature_vector (array): length n_qubits (angles for RY gates).
      weights (array): length n_qubits (rotation angles in the final block).
      genotype (list): list of indices (length = entanglement_level)
                       selecting edges from possible_edges.
    
    Returns:
      QuantumCircuit object.
    """
    qc = QuantumCircuit(n_qubits)
    
    # Feature embedding block: encode classical data as rotations.
    for i in range(n_qubits):
        qc.ry(feature_vector[i], i)
        
    # Entanglement block: add CNOT gates as determined by genotype.
    for gene in genotype:
        # Ensure gene is an integer index in the list of possible edges.
        idx = int(gene)
        control, target = possible_edges[idx]
        qc.cx(control, target)
    
    # Final parameterized block (here fixed for demonstration).
    for i in range(n_qubits):
        qc.ry(weights[i], i)
    
    qc.measure_all()
    return qc

def run_circuit(qc):
    """
    Run the quantum circuit on a simulator and compute an approximate expectation value.
    For each qubit we compute an expectation value by mapping measured '0' -> +1 and '1' -> -1.
    """
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts(qc)
    expectation = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings with qubit 0 as rightmost bit.
        bits = bitstring[::-1]
        for i, bit in enumerate(bits):
            # Map '0' to +1, '1' to -1.
            z_val = 1 if bit == '0' else -1
            expectation[i] += z_val * count
    expectation = expectation / 1024.0
    return expectation

def predict_from_expectation(expectation):
    """
    Very simple binary classifier:
      - Sum the expectation values from all qubits.
      - Pass the sum through a sigmoid.
      - Predict class 1 if probability > 0.5, else 0.
    """
    s = np.sum(expectation)
    prob = 1 / (1 + np.exp(-s))
    return 1 if prob > 0.5 else 0

# ---------------------------
# Objective Function for SMBO
# ---------------------------
def objective(params):
    """
    The objective function evaluates a candidate entanglement layout.
    Here params is a dictionary with keys 'gene_0', 'gene_1', ..., each representing
    an index in the possible_edges list.
    
    We build the circuit for each training sample, get a prediction, and
    compute the mean squared error as a simple loss.
    """
    # Construct the genotype vector from the parameters.
    genotype = [params[f'gene_{i}'] for i in range(entanglement_level)]
    
    loss = 0.0
    for features, label in zip(X_train, y_train):
        qc = build_quantum_circuit(features, fixed_weights, genotype)
        exp_vals = run_circuit(qc)
        pred = predict_from_expectation(exp_vals)
        loss += (pred - label) ** 2  # squared error for binary classification
    loss /= len(X_train)
    return {'loss': loss, 'status': STATUS_OK}

# ---------------------------
# Define the Search Space
# ---------------------------
# Each gene is an integer index between 0 and (len(possible_edges)-1).
space = {}
for i in range(entanglement_level):
    space[f'gene_{i}'] = hp.quniform(f'gene_{i}', 0, len(possible_edges)-1, 1)

# ---------------------------
# Run the SMBO Search using TPE (via Hyperopt)
# ---------------------------
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
best_genotype = [int(best[f'gene_{i}']) for i in range(entanglement_level)]
print("Best candidate genotype (entangling layout):", best_genotype)

# ---------------------------
# Evaluate on the Validation Set
# ---------------------------
def evaluate_candidate(genotype, X, y):
    correct = 0
    for features, label in zip(X, y):
        qc = build_quantum_circuit(features, fixed_weights, genotype)
        exp_vals = run_circuit(qc)
        pred = predict_from_expectation(exp_vals)
        if pred == label:
            correct += 1
    accuracy = correct / len(y)
    return accuracy

val_accuracy = evaluate_candidate(best_genotype, X_val, y_val)
print("Validation accuracy for the selected entangling layout:", val_accuracy)
