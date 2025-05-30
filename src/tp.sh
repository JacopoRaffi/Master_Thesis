#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=log_%j.out

IFACE="eth1"

read_bytes() {
    RX=$(cat /sys/class/net/$IFACE/statistics/rx_bytes)
    TX=$(cat /sys/class/net/$IFACE/statistics/tx_bytes)
    echo "$HOSTNAME $RX $TX"
}

export IFACE
export LD_PRELOAD=/opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so:$LD_PRELOAD
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export LD_PRELOAD=<jemalloc.so/tcmalloc.so>:$LD_PRELOAD
ln -s /opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so ./libomp.so
echo "$@"
echo "=== Prima del training ==="
srun bash -c 'read_bytes() { RX=$(cat /sys/class/net/$IFACE/statistics/rx_bytes); TX=$(cat /sys/class/net/$IFACE/statistics/tx_bytes); echo "$(hostname) $RX $TX"; }; read_bytes'

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

# srun torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=1 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     tensor_parallelism.py "$@" #--minibatch 256 --model "google/vit-base-patch16-224-in21k" \

mpirun python tensor_parallelism.py "$@" #--minibatch 256 --model "google/vit-base-patch16-224-in21k" \

export IFACE
echo "=== Dopo il training ==="
srun bash -c 'read_bytes() { RX=$(cat /sys/class/net/$IFACE/statistics/rx_bytes); TX=$(cat /sys/class/net/$IFACE/statistics/tx_bytes); echo "$(hostname) $RX $TX"; }; read_bytes'


































# srun --nodelist=node09 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=0 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node13 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=1 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node12 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=2 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node15 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=3 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node16 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=4 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node17 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=5 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node18 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=6 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
# srun --nodelist=node19 torchrun --nproc_per_node=1 --nnodes=8 --node_rank=7 --rdzv_endpoint=10.0.1.9:29500 data_parallelism.py --model "google/vit-huge-patch14-224-in21k" --minibatch 256 &
