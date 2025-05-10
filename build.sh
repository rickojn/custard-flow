gcc -c -o cf.o src/CustardFlow.c -march=native -mfma
gcc -c -o main.o src/Main.c
gcc -o build/mm cf.o main.o