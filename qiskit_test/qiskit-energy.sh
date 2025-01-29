for i in {1..10} ; do
echo "Running batch $i"
echo "Spawning 8 processes"
for i in {1..8} ;
do
    ./qiskit_test.py -c tsp -n 16 -r 32 & 
done
wait
done