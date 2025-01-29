#!/bin/bash
cd build/apps
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/vqe_n16.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/swap_test_n25.qasm -t 16
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 16


/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 16  --DDSIM_convert
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16  --DDSIM_convert
/bin/time timeout 24h ./FlatDD --file ../../circuits/vqe_n16.qasm -t 16  --DDSIM_convert
/bin/time timeout 24h ./FlatDD --file ../../circuits/knn_n25.qasm -t 16  --DDSIM_convert
/bin/time timeout 24h ./FlatDD --file ../../circuits/swap_test_n25.qasm -t 16  --DDSIM_convert
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 16  --DDSIM_convert