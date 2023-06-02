#!/bin/bash

# Declare the client job names
clients=("client1" "client2" "client3")

# Submit the server job to the "cuda" partition
python3 server.py

# Loop through the client job names and submit each client job to the "cuda" partition
for client in "${clients[@]}"; do
    sbatch client.py &
done

# Wait for all background client jobs to finish
wait

