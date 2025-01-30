#!/bin/bash
NumBatches=$1
cd build/apps

echo "---------------------  BQSim  ------------------------"
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n17.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n19.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/dnn_n21.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n12.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n14.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/vqe_n16.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n16.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n18.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/graph_state_n20.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/tsp_n9.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/tsp_n16.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/routing_n6.qasm --num_batch $NumBatches --conversion_type 2
./BQSim --ps  --batch_size 256 --file ../../circuits/routing_n12.qasm --num_batch $NumBatches --conversion_type 2

echo "---------------------  FlatDD  ------------------------"

echo "============Simulating DNN n=17"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
# one batch with 8 running processes 
for i in {1..8} ; 
do
    ./FlatDD  --batch_size 32 --file ../../circuits/dnn_n17.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"


echo "============Simulating DNN n=19"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/dnn_n19.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
cur_time=$(date +%s%3N) 
cur_duration=$((cur_time - start_time)) 
if [[ $cur_duration -gt 86400000 ]]
then
    echo "Duration $cur_duration longer than 24h!"
    break
fi

done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=21"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/dnn_n21.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
cur_time=$(date +%s%3N) 
cur_duration=$((cur_time - start_time)) 
if [[ $cur_duration -gt 86400000 ]]
then
        echo "Duration $cur_duration longer than 24h!"
        break
fi
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=12"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/vqe_n12.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=14"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/vqe_n14.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/vqe_n16.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Portfolio opt. w/ VQE n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Portfolio opt. w/ VQE n=17"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"


echo "============Simulating Portfolio opt. w/ VQE n=18"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating graph state n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/graph_state_n16.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"


echo "============Simulating graph state n=18"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/graph_state_n18.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating graph state n=20"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/graph_state_n20.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"


echo "============Simulating TSP n=9"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/tsp_n9.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/tsp_n16.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating routing n=6"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/routing_n6.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating routing n=12"
start_time=$(date +%s%3N) 
for ((i = 1; i <= NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./FlatDD  --batch_size 32 --file ../../circuits/routing_n12.qasm --num_batch 1 --custom_inputs  --fuse 1 --disable_switch & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "---------------------  cuQuantum  ------------------------"
cd ../cuquantum_test
./cuquantum dnn 17 256 $NumBatches 0 0
./cuquantum dnn 19 256 $NumBatches 0 0
./cuquantum dnn 21 256 $NumBatches 0 0 
./cuquantum vqe 12 256 $NumBatches 0 0
./cuquantum vqe 14 256 $NumBatches 0 0
./cuquantum vqe 16 256 $NumBatches 0 0
./cuquantum portfolio_vqe 16 256 $NumBatches 0 0
./cuquantum portfolio_vqe 17 256 $NumBatches 0 0
./cuquantum portfolio_vqe 18 256 $NumBatches 0 0
./cuquantum graph_state 16 256 $NumBatches 0 0
./cuquantum graph_state 18 256 $NumBatches 0 0
./cuquantum graph_state 20 256 $NumBatches 0 0
./cuquantum tsp 9 256 $NumBatches 0 0
./cuquantum tsp 16 256 $NumBatches 0 0
./cuquantum routing 6 256 $NumBatches 0 0
./cuquantum routing 12 256 $NumBatches 0 0

echo "---------------------  Qiskit Aer  ------------------------"
cd ../../qiskit_test

echo "============Simulating DNN n=17"
start_time=$(date +%s%3N) 
# running 200 batches
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
# running each batch on 8 processes
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=19"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 19 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=21"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 21 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=12"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 12 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=14"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 14 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Portfolio opt. w/ VQE n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Portfolio opt. w/ VQE n=17"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 17 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Portfolio opt. w/ VQE n=18"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 18 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Graph State n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c graph_state -n 16 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Graph State n=18"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c graph_state -n 18 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Graph State n=20"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c graph_state -n 20 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=9"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 9 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Routing n=6"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c routing -n 6 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating Routing n=12"
start_time=$(date +%s%3N) 
for ((i = 1; i <= $NumBatches; i++)) ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c routing -n 12 -r 32 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"