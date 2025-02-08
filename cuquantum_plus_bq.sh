#!/bin/bash
cd build/cuquantum_test
# cuquantum+b
./cuquantum vqe 12 256 200 2 0
./cuquantum tsp 9 256 200 2 0
./cuquantum routing 6 256 200 2 0

# cuquantum+q
./cuquantum vqe 12 256 200 1 0
./cuquantum vqe 14 256 200 1 0
./cuquantum vqe 16 256 200 1 0
./cuquantum portfolio_vqe 16 256 200 1 0
./cuquantum graph_state 16 256 200 1 0
./cuquantum tsp 9 256 200 1 0
./cuquantum routing 6 256 200 1 0
./cuquantum routing 12 256 200 1 0