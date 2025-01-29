from qulacs import QuantumState

# Generate random initial state
for n in range(3, 4):
    print("generating input for qubit n="+str(n))
    state = QuantumState(n)
    with open('../input_batch/n'+str(n)+'.txt', 'w') as file:
        state.set_Haar_random_state()
        for amp_id in range(2**n):
            file.write(str(state.get_vector()[amp_id].real) + ' ' + str(state.get_vector()[amp_id].imag) + ' ')