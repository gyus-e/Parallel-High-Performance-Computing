#!/bin/bash

if [ ! -f main.exe ]; then
    make -j
fi

echo "{" > results.json
for i in {1..10}; do
    echo "\"Test $i\": " >> results.json
    ./main.exe >> results.json
    if [ $i -lt 10 ]; then
        echo "," >> results.json
    fi
done
echo "}" >> results.json