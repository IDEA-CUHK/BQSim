mkdir build/
cd qpp/
cmake -B build
sudo cmake --build build --target install
cd ../build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j9