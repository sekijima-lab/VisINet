. /etc/profile.d/modules.sh
module load intel cuda/8.0.61 nccl/2.2.13 cudnn/6.0 openmpi/2.1.2

source ${HOME}/.pyenv/versions/anaconda3-4.4.0/etc/profile.d/conda.sh
conda activate py35

mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python mnist3.py ${prot}
