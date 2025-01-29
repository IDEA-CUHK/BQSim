# !/usr/bin/bash
echo "============Simulating DNN n=17, batch size = 32"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 4 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=17, batch size = 64"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 8 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=17, batch size = 128"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 16 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=17, batch size = 256"
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

echo "============Simulating DNN n=17, batch size = 512"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 64 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating DNN n=17, batch size = 1024"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c dnn -n 17 -r 128 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16, batch size = 32"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 4 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16, batch size = 64"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 8 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16, batch size = 128"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 16 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16, batch size = 256"
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

echo "============Simulating VQE n=16, batch size = 512"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 64 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating VQE n=16, batch size = 1024"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c vqe -n 16 -r 128 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating PORT. VQE n=16, batch size = 32"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 4 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating PORT. VQE n=16, batch size = 64"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 8 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating PORT. VQE n=16, batch size = 128"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 16 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating PORT. VQE n=16, batch size = 256"
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

echo "============Simulating PORT. VQE n=16, batch size = 512"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 64 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating PORT. VQE n=16, batch size = 1024"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c portfolio_vqe -n 16 -r 128 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16, batch size = 32"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 4 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16, batch size = 64"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 8 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16, batch size = 128"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 16 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16, batch size = 256"
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

echo "============Simulating TSP n=16, batch size = 512"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 64 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"

echo "============Simulating TSP n=16, batch size = 1024"
start_time=$(date +%s%3N) 
for i in {1..200} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 128 & 
done
wait
done
end_time=$(date +%s%3N) 
duration_ms=$((end_time - start_time)) 
echo "============Execution time in ms: $duration_ms"