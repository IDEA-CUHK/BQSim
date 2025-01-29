cd build/apps

/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 2
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 4
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 8
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 16

cd ../../qpp/build
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 1 ../../log/results/time/knn_n25_QPP.txt
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 2 ../../log/results/time/knn_n25_QPP.txt
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 4 ../../log/results/time/knn_n25_QPP.txt
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 8 ../../log/results/time/knn_n25_QPP.txt
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 16 ../../log/results/time/knn_n25_QPP.txt