# VISINET, JOBDIR, TFRECORD_HEAD, SGE_TASK_ID

# set protein name (must match a function in ratio.py) and training tfrecord(s)
# (space-separated if more than one); JOBDIR is the model output directory.
protein=
train_tf=

shopt -s expand_aliases
source /opt/conda/etc/profile.d/conda.sh
conda activate visinet

mkdir -p ${JOBDIR}

# GPUs to use are selected by `docker run --gpus`, not here; this uses exactly
# 4 GPUs via -np 4, matching the original P100x4 training run.
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    --allow-run-as-root \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 mnist3.py ${protein} ${JOBDIR} ${train_tf}
