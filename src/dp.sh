#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --output=../jobs_log/log_%x.out
#SBATCH --error=../jobs_log/log_%x.err

IFACE="eth1"
PROVIDER="tcp"

for ARG in "$@"; do
  case "$ARG" in
    -interface=*|--interface=*)
      IFACE="${ARG#*=}"
      ;;
  esac
done

# Determine provider based on interface
if [[ "$IFACE" == "ib0" ]]; then
  PROVIDER="psm2"
elif [[ "$IFACE" == "eth1" ]]; then
  PROVIDER="tcp"
else
  echo "Unsupported interface: $IFACE"
  exit 1
fi

read_bytes() {
    RX=$(cat /sys/class/net/$IFACE/statistics/rx_bytes)
    TX=$(cat /sys/class/net/$IFACE/statistics/tx_bytes)
    echo "$HOSTNAME $RX $TX"
}

export IFACE
export I_MPI_OFI_PROVIDER=$PROVIDER

export LD_PRELOAD=/opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so:$LD_PRELOAD
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
[ -e ./libomp.so ] || ln -s /opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so ./libomp.so
echo "$@"
echo "=== Prima del training ==="
srun bash -c 'read_bytes() { RX=$(cat /sys/class/net/$IFACE/statistics/rx_bytes); TX=$(cat /sys/class/net/$IFACE/statistics/tx_bytes); echo "$(hostname) $RX $TX"; }; read_bytes'



# MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# MASTER_PORT=29500

# srun torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=1 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     data_parallelism.py "$@" #--minibatch 256 --model "google/vit-base-patch16-224-in21k" \

export I_MPI_DEBUG=5

mpirun -iface $IFACE python data_parallelism.py "$@" #--minibatch 256 --model "google/vit-base-patch16-224-in21k" \

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
