# VISINET, JOBNAME, JOBDIR, TFRECORD_HEAD, SGE_TASK_ID

# set tfrecord file and job name
test_tf=
jobname=

shopt -s expand_aliases
source ~/.bashrc
conda activate visinet

# . /etc/profile.d/modules.sh
# module load gcc/8.3.0 cuda/9.0.176 cudnn/7.4 nccl/2.4.2 intel/19.0.0.117 openmpi/3.1.4-opa10.10

mkdir -p result
python3 test4.py ${JOBDIR} ${jobname} ${test_tf}
