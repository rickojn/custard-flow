
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release



cmake --build build-debug

cmake --build build
