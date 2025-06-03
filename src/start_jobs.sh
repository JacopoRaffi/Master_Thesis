#!/bin/bash

sbatch --nodes=16 --exclude=node10 --job-name=dp16_b_64_eth1 dp.sh --minibatch 64 --model "google/vit-base-patch16-224-in21k" --interface=eth1

sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_64_2_eth1 dp.sh --mode="asynch" --tau=2 --minibatch 64 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_64_4_eth1 dp.sh --mode="asynch" --tau=4 --minibatch 64 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_64_8_eth1 dp.sh --mode="asynch" --tau=8 --minibatch 64 --model "google/vit-base-patch16-224-in21k" --interface=eth1

sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_128_2_eth1 dp.sh --mode="asynch" --tau=2 --minibatch 128 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_128_4_eth1 dp.sh --mode="asynch" --tau=4 --minibatch 128 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_128_8_eth1 dp.sh --mode="asynch" --tau=8 --minibatch 128 --model "google/vit-base-patch16-224-in21k" --interface=eth1

sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_256_2_eth1 dp.sh --mode="asynch" --tau=2 --minibatch 256 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_256_4_eth1 dp.sh --mode="asynch" --tau=4 --minibatch 256 --model "google/vit-base-patch16-224-in21k" --interface=eth1
sbatch --nodes=1 --exclude=node10 --job-name=dpa1_b_256_8_eth1 dp.sh --mode="asynch" --tau=8 --minibatch 256 --model "google/vit-base-patch16-224-in21k" --interface=eth1
# count jobs squeue -u j.raffi | tail -n +2 | wc -l
