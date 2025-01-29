#!/bin/bash
cd build/apps
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16 --fuse 0
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16 --fuse 2
