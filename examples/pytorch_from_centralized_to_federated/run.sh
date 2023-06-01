#!/bin/bash

# Declare the client job names
clients=("client1" "client2" "client3")

# Submit the server job to the "cuda" partition
srun server.py

# Loop through the client job names and submit each client job to the "cuda" partition
for client in "${clients[@]}"; do
    srun --partition=cuda --nodes=1 --ntasks=1 --exclusive client.py "$client" &
done

# Wait for all background client jobs to finish
wait

