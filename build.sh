gcc -c -o cf.o src/CustardFlow.c -march=native -mfma debug -O1 -Wall -Wextra -Wpedantic -g
gcc -c -o main.o src/Main.c
gcc -o build/mm cf.o main.o