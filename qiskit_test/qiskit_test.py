#!/usr/bin/python3
import numpy as np
import qiskit.qasm2
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
import time
import argparse
from qiskit import QuantumCircuit
import qiskit

def read_complex_numbers(filename):
    with open(filename, 'r') as file:
        line = file.readline()  # Read the whole line since all numbers are on one line
    
    complex_numbers = []
    # Split the line by spaces, process pairs of real and imaginary numbers
    numbers = list(map(float, line.split()))  # Convert all to float
    for i in range(0, len(numbers), 2):
        real = numbers[i]
        imag = numbers[i + 1]
        complex_numbers.append(complex(real, imag))  # Create complex number
    
    return np.array(complex_numbers)

def write_complex_numbers(filename, complex_array):
    with open(filename, 'w') as file:
        for cnum in complex_array:
            file.write(f"{cnum.real} {cnum.imag}\n")  

parser = argparse.ArgumentParser(description='Qiskit AER GPU baseline.')
parser.add_argument('--circuit_name', '-c', type=str, help='Name of the circuit (e.g., dnn, vqe)')
parser.add_argument('--num_qubits', '-n', type=int, help='Number of qubits')
parser.add_argument('--rounds', '-r', type=int, help='Rounds')
# parser.add_argument('--num_batches', '-b', type=int, help='Number of batches')

args = parser.parse_args()
circuit_name = args.circuit_name
num_qubits = args.num_qubits
rounds = args.rounds

circuit_path = "../circuits/"+circuit_name+"_n"+str(num_qubits)+".qasm"
input_path = "../input_batch/n"+str(num_qubits)+".txt"
output_path = "../log/results/state/qiskit_"+circuit_name+"_n"+str(num_qubits)+".txt"

input_state_np = read_complex_numbers(input_path)
input_state_q = qiskit.quantum_info.Statevector(input_state_np)
circuit1 = QuantumCircuit(num_qubits)
circuit1.set_statevector(input_state_q)
circuit2 = qiskit.qasm2.load(circuit_path)
circuit = circuit1.compose(circuit2)
circuit.save_statevector()

sim = AerSimulator(method='statevector', device='GPU', fusion_enable=True, fusion_threshold=5, fusion_verbose=True, batched_shots_gpu=True)



# time1 = time.time()

# circuit.set_statevector(input_state_q)
for i in range(rounds):
    job = sim.run(circuit) 
    result = job.result()
    statevector = result.get_statevector(circuit)
