'''
This script uses SLURM to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).
Command
    python3 run_slurm.py <NUM_TIMESTEPS> <ALG_NAME> <N_TRIALS> <ST_TRIAL> <ENV_LIST>
Example
    python3 run_slurm.py 5e4 mbl_copos 1 0 Pendulum-v0 Swimmer-v2
One job per run will be submitted.
Data is still saved as usual in `path/to/logs/the_env_name/the_alg_name_the_seed/`. For example, for the above run data will be saved in
~/Desktop/logs/EXP_V0/Pendulum-v0/mbl+copos_0/
~/Desktop/logs/EXP_V0/Pendulum-v0/mbl+copos_1/
...
Additionally, stdout and stderr are flushed to log files. For example
/home/sz52cacy/logs-trial/Pendumlum-v0_TEST/stdout_mbl_copos_0
/home/sz52cacy/logs-trial/Pendumlum-v0_TEST/stderr_mbl_copos_0
...
NOTE: Change the slurm script according to your needs (activate virtual env,
request more memory, more computation time, ...).
'''

import os, errno, sys

logdir = '/home/sz52cacy/logs-trial/' # directory to save log files (where stdout is flushed)
num_timesteps = sys.argv[1]
alg_name = sys.argv[2]
n_trials = int(sys.argv[3])
st_trial = int(sys.argv[4])
env_list = sys.argv[5:]
for env_name in env_list:
    envdir = env_name + '_TEST/'
    try:
        os.makedirs(logdir+envdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    test_trials=1
    for trial in range(st_trial, st_trial+test_trials):
        run_name = alg_name + '_' + str(trial)
      
        text = """\
#!/bin/bash
# job name
#SBATCH -J job_name
# logfiles
#SBATCH -o """ + logdir + """""" + envdir + """stdout_""" + run_name + """\
#SBATCH -e """ + logdir + """""" + envdir + """stderr_""" + run_name + """\
# request computation time hh:mm:ss
#SBATCH -t 24:00:00
# request virtual memory in MB per core
#SBATCH --mem-per-cpu=2000
# nodes for a single job
#SBATCH -n 1
#SBATCH -c 64
# activate virtual env
module load gcc intel openmpi
conda activate cse
python ~/Desktop/carla_sample_efficient/plt/command_offline_alg_lgd_test.py --num_timesteps=""" + num_timesteps + """ --seeds=1 --st_seed=""" + str(trial) + """ --alg=""" + alg_name + """\
    """

        text_file = open('r.sh', "w")
        text_file.write(text)
        text_file.close()
        
        os.system('sbatch r.sh')
        os.remove('r.sh')
