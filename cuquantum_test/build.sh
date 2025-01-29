#!/bin/bash
git submodule update --init --recursive
#git submodule update --remote --recursive
build_dir="./build"

if [ ! -d "$build_dir" ]; then
    mkdir $build_dir
fi

cd $build_dir

if [ $# -ge 1 ]
then
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$1 ..
else
    cmake -G "Unix Makefiles" ..
fi

make -j7