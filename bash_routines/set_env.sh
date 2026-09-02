#!/bin/bash
# Environment setup shared by all PINNoDiffPhys SLURM routines.
# Usage: . bash_routines/set_env.sh <CLUSTER>
#   CLUSTER in {ICA, SD2_h100, SD2_gh200}

if [ -z "$1" ]; then
    echo "ERROR: set_env.sh requires a cluster id (ICA|SD2_h100|SD2_gh200)"
    exit 1
fi

export CLUSTER_ID=$1

# Root of the repo (this file lives at <repo>/bash_routines/set_env.sh)
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SRC_ROOT

if [ "$1" == "ICA" ]; then
    echo "----- selected ICA (cpu) -----"
    export PATH_ENV='/share_zeta/Proxy-Sim/guillermo.carrillo'
    export PATH_CODE="$SRC_ROOT"
    export CONTAINER_PATH="$PATH_ENV/envs/ICA_v4.sif"
    export SLURM_PARTITION='cpu'
    export SLURM_NODELIST='cpunode-2-1'
elif [ "$1" == "SD2_h100" ]; then
    echo "----- selected SDumont h100 -----"
    export PATH_ENV='/petrobr/parceirosbr/proxy-sim'
    export PATH_CODE="$SRC_ROOT"
    export CONTAINER_PATH="$PATH_ENV/users/guillermo.carrillo/Ambientes/h100_v3.sif"
    export SLURM_PARTITION='gpu_dev'
elif [ "$1" == "SD2_gh200" ]; then
    echo "----- selected SDumont gh200 -----"
    export PATH_ENV='/petrobr/parceirosbr/proxy-sim'
    export PATH_CODE="$SRC_ROOT"
    export CONTAINER_PATH="$PATH_ENV/users/guillermo.carrillo/Ambientes/gh200_v2.sif"
    export SLURM_PARTITION='gpu_normal'
else
    echo "ERROR: unknown cluster '$1'"
    exit 1
fi

chmod +x "$PATH_CODE"/srm_routines/*.srm "$PATH_CODE"/bash_routines/*.sh 2>/dev/null
echo "SRC_ROOT=$SRC_ROOT"
echo "CONTAINER_PATH=$CONTAINER_PATH"
