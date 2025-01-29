# !/usr/bin/bash

cd build/apps
echo "============Simulating DNN n=17"
start_time=$(date +%s%3N) 
# 200 batches
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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


echo "============Simulating TSP n=9"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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

echo "============Simulating graph state n=16"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
