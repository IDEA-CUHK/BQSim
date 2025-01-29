cd build/apps
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 1 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 2 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 4 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 8 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 16 --fuse 1

/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 1 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 2 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 4 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 8 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16 --fuse 1

/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 1 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 2 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 4 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 8 --fuse 1
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 16 --fuse 1


/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 1 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 2 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 4 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 8 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n16.qasm -t 16 --fuse 1  --no_cache

/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 1 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 2 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 4 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 8 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/dnn_n20.qasm -t 16 --fuse 1  --no_cache

/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 1 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 2 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 4 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 8 --fuse 1  --no_cache
/bin/time timeout 24h ./FlatDD --file ../../circuits/supremacy_n20.qasm -t 16 --fuse 1  --no_cache