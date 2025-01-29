# !/usr/bin/bash
echo "============Simulating DNN n=17"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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
for i in {1..200} ; do
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