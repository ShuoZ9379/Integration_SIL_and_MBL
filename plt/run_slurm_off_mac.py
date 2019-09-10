'''
This script uses SLURM to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).
Command
    python3 run_slurm_off.py <TIME_LIMIT> <NUM_TIMESTEPS> <N_TRIALS> <ST_TIAL> <ENV_LIST>
Example
    python3 run_slurm_off.py 24:00:00 5e6 10 0 HalfCheetah-v2 Ant-v2 Reacher-v2 Swimmer-v2
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
alg_ls=["copos1_offline","copos1_sil_offline","mbl_copos1","mbl_copos1_sil",
        "trpo_offline","trpo_sil_offline","mbl_trpo","mbl_trpo_sil",
        "ppo2_offline","ppo2_sil_offline","mbl_ppo2","mbl_ppo2_sil",
        "copos2_offline","copos2_sil_offline","mbl_copos2","mbl_copos2_sil"]
#alg_ls=["copos(const)_offline","copos(const)_sil_offline","mbl_copos(const)","mbl_copos(const)_sil"]
logdir = '/home/sz52cacy/logs-trial-off' # directory to save log files (where stdout is flushed)
time_limit = sys.argv[1]
num_timesteps = sys.argv[2]
n_trials = int(sys.argv[3])
st_trial = int(sys.argv[4])
env_list = sys.argv[5:]
if time_limit == "24:00:00":
    logdir = logdir+'-24'
    if num_timesteps=='5e6':
        logdir = logdir+'-5M/'
    elif num_timesteps=='1e6':
        logdir = logdir+'-1M/'
    else: logdir = logdir+'-do-not-use/'
else:
    logdir = logdir+'-120/'

for env_name in env_list:
    envdir = env_name + '/'
    try:
        os.makedirs(logdir+envdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for alg_name in alg_ls:
        
        for trial in range(st_trial, st_trial+n_trials):
            run_name = alg_name + '_' + str(trial)
      
            text = """\
#!/bin/bash
# job name
#SBATCH -J job_name
# logfiles
#SBATCH -o """ + logdir + """""" + envdir + """stdout_""" + run_name + """
#SBATCH -e """ + logdir + """""" + envdir + """stderr_""" + run_name + """
# request computation time hh:mm:ss
#SBATCH -t """ + time_limit + """
# request virtual memory in MB per core
#SBATCH --mem-per-cpu=6000
# nodes for a single job
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -C avx
# activate virtual env
module load intel openmpi/3
python ~/Desktop/carla_sample_efficient/plt/command_offline_alg_lgd_mac.py --num_timesteps=""" + num_timesteps + """ --seeds=1 --st_seed=""" + str(trial) + """ --alg=""" + alg_name + """ --env=""" + env_name + """ --time=""" + time_limit + """\
    """

            text_file = open('r.sh', "w")
            text_file.write(text)
            text_file.close()
            
            os.system('sbatch r.sh')
            os.remove('r.sh')
