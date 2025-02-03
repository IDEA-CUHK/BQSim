#!/bin/bash

cd cuquantum_test/qpp/
cmake -B build
cmake --build build --target install


cd ../../
mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j9
