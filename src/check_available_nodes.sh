#!/bin/bash

# List of nodes in the SLURM cluster
NODES=$(sinfo -N -h -o "%n")

# Users to ignore (system and SLURM users)
IGNORE_USERS="root|slurm|sshd|nobody"

for NODE in $NODES; do
    echo -n "$NODE: "
    
    ssh "$NODE" "
        ps -eo user= | grep -vE '$IGNORE_USERS' | sort | uniq
    " | tr '\n' ' '
    
    echo
done