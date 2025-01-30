#!/bin/bash
cd build/apps
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n17.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n19.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n21.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n12.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n14.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n18.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n20.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/tsp_n9.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/tsp_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/routing_n6.qasm --num_batch 200 --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/routing_n12.qasm --num_batch 200 --conversion_type 2