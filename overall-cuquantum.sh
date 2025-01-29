#!/bin/bash

cd build/cuquantum_test
# Args: num_qubits, batchsize, num_batches, use_fused_gates: (0-no fusion, 1-qiskit fusion, 2-BQCS-aware fusion), and output_or_not (0 or 1)
./cuquantum dnn 17 256 200 0 0
./cuquantum dnn 19 256 200 0 0
./cuquantum dnn 21 256 200 0 0 
./cuquantum vqe 12 256 200 0 0
./cuquantum vqe 14 256 200 0 0
./cuquantum vqe 16 256 200 0 0
./cuquantum portfolio_vqe 16 256 200 0 0
./cuquantum portfolio_vqe 17 256 200 0 0
./cuquantum portfolio_vqe 18 256 200 0 0
./cuquantum graph_state 16 256 200 0 0
./cuquantum graph_state 18 256 200 0 0
./cuquantum graph_state 20 256 200 0 0
./cuquantum tsp 9 256 200 0 0
./cuquantum tsp 16 256 200 0 0
./cuquantum routing 6 256 200 0 0
./cuquantum routing 12 256 200 0 0