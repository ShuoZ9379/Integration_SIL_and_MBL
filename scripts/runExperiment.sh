#!/bin/bash
#SBATCH -A project00720
#SBATCH -J baselines_experiment
#SBATCH --mail-user=pajarinen@ias.tu-darmstadt.de
#SBATCH --mail-type=NONE
#SBATCH -e /work/scratch/jp89jera/tmp/baselines_experiment.err.%A_%a.txt
#SBATCH -o /work/scratch/jp89jera/tmp/baselines_experiment.out.%A_%a.txt
#SBATCH --mem-per-cpu=2500
#SBATCH -t 23:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -C avx2

dryrun=false

CMD=${1}

echo "This is Job $SLURM_JOB_ID: importing modules and initializing..."
module load gcc openmpi/gcc intel cuda/9.0
export PATH="$HOME/anaconda3/bin:$PATH"
export ROBOSCHOOL_PATH=/home/jp89jera/src/roboschool
export LD_LIBRARY_PATH=$HOME/usr/local/lib:$HOME/cuda/lib64:$LD_LIBRARY_PATH

# Sleep 1 second
sleep 1

# Add file path
CMD+=" --filepath $TMP/$SLURM_JOB_ID/"

# Log experiment parameters
echo "runExperiment.sh: '${CMD}'"
if [ "$dryrun" = false ] ; then
    echo "${CMD}" >> $HOME/experiment_list
fi

# 1. Put data into local directory with job ID
# 2. Run script
# 3. Copy data into global directory

if [ "$dryrun" = false ] ; then
    source activate openai
    cd $HOME/src/openai_baselines_extended
    
    # Disable core dumps
    ulimit -c 0

    mkdir -p $TMP/$SLURM_JOB_ID/
    
    eval $CMD
    
    # Copy results from local machine to scratch
    mkdir -p /work/scratch/$USER/$SLURM_JOB_ID/
    cp -a $TMP/$SLURM_JOB_ID/* /work/scratch/$USER/$SLURM_JOB_ID/

    source deactivate
fi

