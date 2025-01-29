cd build/apps
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/ghz_state_n23.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/vqe_n16.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 16

/bin/time timeout 24h ./DDSIM --file ../../circuits/dnn_n16.qasm
/bin/time timeout 24h ./DDSIM --file ../../circuits/ghz_state_n23.qasm
/bin/time timeout 24h ./DDSIM --file ../../circuits/vqe_n16.qasm
/bin/time timeout 24h ./DDSIM --file ../../circuits/knn_n25.qasm

cd ../../qpp/build
/bin/time ./qpp_qasm ../../circuits/dnn_n16.qasm 16 ../../log/results/time/dnn_n16_QPP.txt
/bin/time ./qpp_qasm ../../circuits/ghz_state_n23.qasm 16 ../../log/results/time/ghz_state_n23_QPP.txt
/bin/time ./qpp_qasm ../../circuits/vqe_n16.qasm 16 ../../log/results/time/vqe_n16_QPP.txt
/bin/time ./qpp_qasm ../../circuits/knn_n25.qasm 16 ../../log/results/time/knn_n25_QPP.txt