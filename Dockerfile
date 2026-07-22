# VisINet reproduction environment.
#
# Goal: reproduce the paper's reported accuracy numbers, not to improve them.
# Do not bump package versions beyond what environment.yaml/pip.txt pin,
# and do not change GPU/worker counts here -- that is decided at `docker run`
# / `mpirun -np` time (see README-docker.md), not baked into this image.
#
# Base image note: nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 was removed from
# Docker Hub; the equivalent image lives on NGC and matches environment.yaml's
# cudatoolkit=10.0.130 / cudnn=7.6.5 exactly (verified on the target host).
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# --- OS packages -------------------------------------------------------
# - openmpi: required to run mnist3.py via `mpirun` (Horovod), matching the
#   original train.sh which used OpenMPI on the HPC cluster. Not pinned by
#   environment.yaml/pip.txt (gap in the published env) so we take the
#   Ubuntu 18.04 apt version (2.1.1), which is close to the original
#   openmpi/2.1.2 module used on TSUBAME.
# - libgl1/libxrender1/libxext6/libsm6: runtime libs PyMol and opencv need
#   for offscreen rendering (`pymol -qc`) and image decoding.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
        git \
        ca-certificates \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
        libgl1 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda (python 3.7, per environment.yaml) -----------------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

WORKDIR /opt/visinet

# --- Recreate the published conda env -----------------------------------
COPY environment.yaml pip.txt ./
RUN conda env create -f environment.yaml

# pip.txt duplicates most of environment.yaml (both were exported from the
# same working env); install it into the "visinet" env for the packages
# environment.yaml doesn't carry (e.g. opencv-*). Two kinds of lines are
# dropped before installing, neither of which VisINet's code imports:
#  - horovod: excluded here, built separately below with MPI/NCCL build flags.
#  - flake8/pylint/astroid/etc.: linter/dev-tool packages captured by a
#    `pip freeze` of the author's dev env; flake8==4.0.1 and the pinned
#    importlib-metadata==4.11.4 have conflicting requirements, and none of
#    these tools are used by mnist3.py/test4.py/resnet_v2.py/getimage.py.
RUN grep -viE '^(horovod|flake8|pylint|astroid|autopep8|mccabe|pycodestyle|pyflakes|isort)==' pip.txt > /tmp/pip-filtered.txt \
    && conda run -n visinet pip install --no-cache-dir -r /tmp/pip-filtered.txt

# --- NCCL upgrade (host/kernel PCIe-reporting incompatibility) -----------
# environment.yaml pins nccl=2.8.3.1. On this host, NCCL's PCI topology
# scan fails hard while parsing the kernel's link-speed string:
#   NCCL WARN KV Convert to int : could not find value of '8.0 GT/s PCIe'
#   in dictionary
# -> ncclCommInitRank failed: internal error
# NCCL 2.8.3's lookup table doesn't have an entry for this exact string
# (a decimal-point PCIe Gen3 speed format from a newer kernel than the
# original P100 cluster used); this is fixed in later NCCL releases that
# extended the table. Allreduce is a plain sum regardless of NCCL version/
# ring-vs-tree algorithm choice, so this bump does not change what the
# model computes -- it is a hardware/kernel compatibility fix, not a
# modeling change.
#
# conda-forge's newer nccl builds all require a CUDA-11+ `__cuda` virtual
# package, incompatible with our pinned cudatoolkit=10.0.130 -- letting
# conda's classic solver reconcile "nccl>=2.15" with cudatoolkit=10.0.130
# sent it into an unbounded search (multi-hour, 100% CPU, never
# converging) and had to be killed. Instead we take NVIDIA's standalone
# `nvidia-nccl-cu12` PyPI wheel (ships both libnccl.so.2 and nccl.h, no
# conda solving involved).
#
# This must happen *before* building horovod below: horovod statically
# links libnccl_static.a by default when it finds one (confirmed via
# `ldd` -- the first build's mpi_lib.so had no dynamic libnccl dependency
# at all), so swapping the .so out afterwards had no effect. Installing
# the new NCCL first and pointing HOROVOD_NCCL_HOME/HOROVOD_NCCL_LINK at
# it makes horovod's build use the new one instead.
RUN conda run -n visinet pip install --no-cache-dir nvidia-nccl-cu12==2.23.4
ENV NCCL_ROOT=/opt/conda/envs/visinet/lib/python3.7/site-packages/nvidia/nccl

# --- Horovod (MPI + NCCL backend, matching the original mpirun-based setup) ---
# mpi4py is not listed anywhere in environment.yaml/pip.txt but is required
# for Horovod's MPI controller; added here as a build-environment gap-fill,
# it does not touch model/training code.
RUN conda run -n visinet pip install --no-cache-dir mpi4py==3.1.4
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_NCCL_HOME=${NCCL_ROOT} \
    HOROVOD_NCCL_LINK=SHARED \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    conda run -n visinet pip install --no-cache-dir --no-cache-dir horovod==0.26.1

# --- PyMol (required by getimage.py, not pinned anywhere upstream) -------
# NOTE: the original authors never recorded which PyMol version/build they
# rendered the training images with. pymol-open-source from conda-forge is
# the closest available substitute; rendering differences vs. the original
# (anti-aliasing, font, ray-tracing defaults) are a known, unavoidable risk
# for exact reproduction -- see REPRODUCTION_NOTES.md section 4.
RUN conda install -n visinet -y -c conda-forge pymol-open-source=2.5.0 \
    && conda clean -afy

# openssh-client: OpenMPI's plm_rsh launcher looks for an `ssh` binary even
# for single-node `localhost` runs; without it mpirun refuses to start.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# --- RDKit (workaround for no Incentive-PyMOL license) -------------------
# getimage.py's inputs are Schrodinger Maestro .mae/.maegz files. Reading
# that format is an Incentive-PyMOL-only feature; pymol-open-source refuses
# it with `IncentiveOnlyException`. We have no PyMOL license available, so
# as an explicit, user-approved workaround we use RDKit's MaeMolSupplier
# (built on Schrodinger's own BSD-licensed `maeparser`) to parse the
# receptor/ligand-pose structures, then hand them to pymol-open-source for
# the actual rendering. This changes *how* structures are read, not the
# structures themselves or the rendering step -- but it is still a real
# deviation from whatever PyMOL build the original authors used, and is a
# known, accepted source of possible pixel-level differences in the
# rendered training images. See REPRODUCTION_NOTES.md section 6a.
RUN conda install -n visinet -y -c conda-forge rdkit=2022.09.1 \
    && conda clean -afy

# Remove the old conda nccl=2.8.3.1 runtime files so nothing can shadow
# the newer library horovod was just built against (see the NCCL section
# above, near the horovod build, for why this swap is needed at all).
RUN rm -f /opt/conda/envs/visinet/lib/libnccl.so /opt/conda/envs/visinet/lib/libnccl.so.2 \
          /opt/conda/envs/visinet/lib/libnccl.so.2.8.3 /opt/conda/envs/visinet/lib/libnccl_static.a
ENV LD_LIBRARY_PATH=${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}

# --- ResNet source patched against the original TF1-contrib API ---------
COPY patch.sh ./
COPY patches ./patches
RUN bash patch.sh

# --- Application code -----------------------------------------------------
COPY *.py ./

# Horovod's MPI controller refuses to run as root by default; the container
# runs as root, so allow it explicitly instead of adding a non-root user
# (kept minimal on purpose -- this is a reproduction image, not a service).
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

SHELL ["/bin/bash", "-c"]
RUN echo "conda activate visinet" >> /root/.bashrc
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]
