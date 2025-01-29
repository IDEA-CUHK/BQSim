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
parser.add_argument('--batch_size', '-s', type=int, help='Batch size')
parser.add_argument('--num_batches', '-b', type=int, help='Number of batches')

args = parser.parse_args()
circuit_name = args.circuit_name
num_qubits = args.num_qubits
batch_size = args.batch_size
num_batches = args.num_batches

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



time1 = time.time()
for i in range(num_batches):
    print("Simulating batch #"+str(i))
    for j in range(batch_size):
        # circuit.set_statevector(input_state_q)
        job = sim.run(circuit) 
        result = job.result()
        statevector = result.get_statevector(circuit)


time2 = time.time()
print("Simulation time for "+circuit_name+"_n"+str(num_qubits) + ": "+str(time2-time1))


job = sim.run(circuit) 
result = job.result()
statevector = result.get_statevector(circuit)
# print(np.array(statevector))
write_complex_numbers(output_path, np.array(statevector))
total_mac = 0
cx_flag = False
cy_flag = False
cz_flag = False

for gate in result.results[0].metadata.get("fusion", {})['output_ops'][1:-1]:
    if gate['name'] == 'unitary':
        total_mac += 2**len(gate['qubits'])
        cx_flag = False
        cy_flag = False
        cz_flag = False
    elif gate['name'] == 'diagonal':
        total_mac += 1
        cx_flag = False
        cy_flag = False
        cz_flag = False
    elif gate['name'] == 'cx':
        cy_flag = False
        cz_flag = False
        if cx_flag == False:
            total_mac += 1
            cx_flag = True
    elif gate['name'] == 'cy':
        cx_flag = False
        cz_flag = False
        if cy_flag == False:
            total_mac += 1
            cy_flag = True
    elif gate['name'] == 'cz':
        cx_flag = False
        cy_flag = False
        if cz_flag == False:
            total_mac += 1
            cz_flag = True
    else:
        print("[WARNING] Undetected fused gate type: "+gate['name'])

print("Total #MACs=" +str(total_mac))
print("#Fused gates = "+str(len(result.results[0].metadata.get("fusion", {})['output_ops'][:-1])))
# print(result.results[0].metadata.get("fusion", {})['output_ops'][1:-1])

with open('../log/fused_gates/qiskit_'+circuit_name+'_n'+str(num_qubits)+".txt", 'w') as file:
    file.write(str(len(result.results[0].metadata.get("fusion", {})['output_ops'][1:-1])) + "\n")
    for gate in result.results[0].metadata.get("fusion", {})['output_ops'][1:-1]:
        file.write(str(len(gate['qubits'])) + "\n")
        file.write(str(gate['qubits'][0]))
        for i in range(1, len(gate['qubits'])):
            file.write(" "+str(gate['qubits'][i]))
        file.write("\n")
        tensor_size = (2**len(gate['qubits'])) * (2**len(gate['qubits']))
        file.write(str(tensor_size) + "\n")
        if gate['name'] == 'unitary':
            for i in range(2**len(gate['qubits'])):
                for j in range(2**len(gate['qubits'])):
                    file.write(str(gate['mats'][0][i][j][0])+" "+str(gate['mats'][0][i][j][1])+" ")
            
        elif gate['name'] == 'diagonal':
            for i in range(2**len(gate['qubits'])):
                for j in range(2**len(gate['qubits'])):
                    if i == j:
                        file.write(str(gate['params'][i][0])+" "+str(gate['params'][i][1])+" ")
                    else:
                        file.write("0 0 ")
        elif gate['name'] == 'cx':
            file.write("1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0")
        elif gate['name'] == 'cy':
            file.write("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0")
        elif gate['name'] == 'cz':
            file.write("1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 -1 0")

        file.write("\n")
