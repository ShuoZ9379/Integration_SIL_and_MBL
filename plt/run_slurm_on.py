'''
This script uses SLURM to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).
Command
    python3 run_slurm_on.py <NUM_TIMESTEPS> <N_TRIALS> <ST_TIAL> <ENV_LIST>
Example
    python3 run_slurm_on.py 1e7 10 0 HalfCheetah-v2 Ant-v2 Reacher-v2 Swimmer-v2
One job per run will be submitted.
Data is still saved as usual in `path/to/logs/the_env_name/the_alg_name_the_seed/`. For example, for the above run data will be saved in
~/Desktop/logs/EXP_V0/Pendulum-v0/mbl+copos_0/
~/Desktop/logs/EXP_V0/Pendulum-v0/mbl+copos_1/
...
Additionally, stdout and stderr are flushed to log files. For example
/home/sz52cacy/logs-trial/Pendumlum-v0/stdout_mbl_copos_0
/home/sz52cacy/logs-trial/Pendumlum-v0/stderr_mbl_copos_0
...
NOTE: Change the slurm script according to your needs (activate virtual env,
request more memory, more computation time, ...).
'''

import os, errno, sys
algo_names=["ppo2_sil_online","ppo2_sil_online","ppo2_online",
            "copos1_sil_online","copos1_sil_online","copos1_online",
            "copos2_sil_online","copos2_sil_online","copos2_online",
            "trpo_sil_online","trpo_sil_online","trpo_online"]

argus=['+sil_n10_l0.1','+sil_n2_l0.001','',
       '+sil_n10_l0.1','+sil_n2_l0.001','',
       '+sil_n10_l0.1','+sil_n2_l0.001','',
       '+sil_n10_l0.1','+sil_n2_l0.001','']

logdir = '/home/sz52cacy/logs-trial-on/' # directory to save log files (where stdout is flushed)
num_timesteps = sys.argv[1]
n_trials = int(sys.argv[2])
st_trial = int(sys.argv[3])
env_list = sys.argv[4:]

for env_name in env_list:
    envdir = env_name + '/'
    try:
        os.makedirs(logdir+envdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for k in range(len(algo_names)):
        
        for trial in range(st_trial, st_trial+n_trials):
            run_name = algo_names[k] + argus[k] +'_' + str(trial)
      
            text = """\
#!/bin/bash
# job name
#SBATCH -J job_name
# logfiles
#SBATCH -o """ + logdir + """""" + envdir + """stdout_""" + run_name + """
#SBATCH -e """ + logdir + """""" + envdir + """stderr_""" + run_name + """
# request computation time hh:mm:ss
#SBATCH -t 24:00:00
# request virtual memory in MB per core
#SBATCH --mem-per-cpu=1750
# nodes for a single job
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -C avx
# activate virtual env
module load intel openmpi/3
python ~/Desktop/carla_sample_efficient/plt/command_online.py --num_timesteps=""" + num_timesteps + """ --seeds=1 --st_seed=""" + str(trial) + """ --alg=""" + algo_names[k] + """ --env=""" + env_name + """ --argu=""" + argus[k] + """\
    """

            text_file = open('r.sh', "w")
            text_file.write(text)
            text_file.close()
            
            os.system('sbatch r.sh')
            os.remove('r.sh')
