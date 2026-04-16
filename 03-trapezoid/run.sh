#!/bin/bash

if [ ! -f main.exe ]; then
    make -j
fi

mkdir -p results

run() {
    blocksize_exp=$1
    num_div_exp=$2
    num_div_sub=$3

    max_runs=50
    run_conf=$blocksize_exp'_'$num_div_exp'_'$num_div_sub
    filename='results/results.json'

    echo "\"$run_conf\": [" >> $filename

    for i in {1..50}; do
        ./main.exe $blocksize_exp $num_div_exp $num_div_sub >> $filename
        if [ $i -lt 50 ]; then
            echo "," >> $filename
        fi
    done

    echo "]" >> $filename
}

echo "{" > results/results.json
# 2^(i+5) block size, 2^20 subdivisions
run 1 20 0
echo "," >> results/results.json
run 2 20 0
echo "," >> results/results.json
run 3 20 0
echo "," >> results/results.json
run 4 20 0
echo "," >> results/results.json
run 5 20 0
echo "," >> results/results.json

# 2^(i+5) block size, 2^20 - 17 subdivisions (not a power of 2)
run 1 20 17
echo "," >> results/results.json
run 2 20 17
echo "," >> results/results.json
run 3 20 17
echo "," >> results/results.json
run 4 20 17
echo "," >> results/results.json
run 5 20 17
echo "}" >> results/results.json