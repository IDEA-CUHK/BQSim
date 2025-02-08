#!/bin/bash
cd build/apps
./BQSim --ps  --batch_size 1 --file ../../circuits/vqe_n12.qasm --num_batch 1 --conversion_type 2 --export_fused_gates
./BQSim --ps  --batch_size 1 --file ../../circuits/tsp_n9.qasm --num_batch 1 --conversion_type 2 --export_fused_gates
./BQSim --ps  --batch_size 1 --file ../../circuits/routing_n6.qasm --num_batch 1 --conversion_type 2 --export_fused_gates

cd ../../qiskit_test
./qiskit_export_gate.py -c vqe -n 12
./qiskit_export_gate.py -c vqe -n 14
./qiskit_export_gate.py -c vqe -n 16
./qiskit_export_gate.py -c portfolio_vqe -n 16
./qiskit_export_gate.py -c graph_state -n 16
./qiskit_export_gate.py -c tsp -n 9
./qiskit_export_gate.py -c routing -n 6
./qiskit_export_gate.py -c routing -n 12
