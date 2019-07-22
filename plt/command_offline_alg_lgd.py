import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--alg', help='Algorithm', type=str, default='copos_offline')
    parser.add_argument('--seeds', help='number of seeds', type=int, default=1)
    parser.add_argument('--st_seed', help='start number of seeds', type=int, default=0)
    parser.add_argument('--num_timesteps', type=str, default="1e6")
    parser.add_argument('--filename', type=str, default='_Offline_Evaluation.png')
    args = parser.parse_args()
    if args.env=='Swimmer-v2' or args.env=='HalfCheetah-v2': 
         mbl_args='--num_samples=1500 --num_elites=10 --horizon=10 --eval_freq=10 --mbl_train_freq=5 --num_eval_episodes=5 --num_warm_start=20000 --use_mean_elites=1 --mbl_sh=1'
    elif arg.env=='Reacher-v2' or args.env=='Ant-v2':
         mbl_args='--num_samples=1500 --num_elites=10 --horizon=5 --eval_freq=10 --mbl_train_freq=5 --num_eval_episodes=5 --num_warm_start=20000 --use_mean_elites=1 --mbl_sh=1'
    
#    algo_names=["ppo2_sil_online","copos_sil_online","ppo2_online","copos_online"]
#    legend_names=["ppo2+sil","copos+sil","ppo2","copos"]
#    argus=["","","",""]

#    algo_names=["mbl_ppo2_sil","mbl_ppo2","ppo2_sil_offline","ppo2_offline",
#                "mbl_copos_sil","mbl_copos","copos_sil_offline","copos_offline",
#                "mbl_trpo_sil","mbl_trpo","trpo_sil_offline","trpo_offline"]
#    algo_names=["mbl_ppo2","ppo2_offline",
#                "mbl_copos","copos_offline"]
#    legend_names=["mbl+ppo+sil","mbl+ppo","ppo+sil","ppo",
#                  "mbl+copos+sil","mbl+copos","copos+sil","copos",
#                  "mbl+trpo+sil", "mbl+trpo", "trpo+sil","trpo"]
#    legend_names=["mbl+ppo","ppo",
#                  "mbl+copos","copos"]
    #argus=['--num_samples=1 --num_elites=1 --horizon=2' for _ in range(len(algo_names))]
    algo_names=[args.alg]
    dct = {'copos_offline': 'copos', 'mbl_copos': 'mbl+copos', 'mbl_copos_sil': 'mbl+copos+sil',
            'trpo_offline': 'trpo', 'mbl_trpo': 'mbl+trpo', 'mbl_trpo_sil': 'mbl+trpo+sil',
            'ppo2_offline': 'ppo', 'mbl_ppo2': 'mbl+ppo', 'mbl_ppo2_sil': 'mbl+ppo+sil'}
    legend_names=[dct[args.alg]]
    argus=[mbl_args for _ in range(len(algo_names))]

    for i in range(args.st_seed, args.st_seed+args.seeds):
        for j in range(len(algo_names)):
            os.system("python ~/Desktop/carla_sample_efficient/algos/"+algo_names[j]+"/run.py --alg="+algo_names[j]+" --num_timesteps="
                      +args.num_timesteps+" --seed="+str(i)+" --env="+args.env+" --log_path=~/Desktop/logs/EXP_OFF_V0/"
                      +args.env+"/"+legend_names[j]+"-"+str(i)+' '+argus[j])
            results = pu.load_results('~/Desktop/logs/EXP_OFF_V0/'+args.env+"/"+dct[algo_names[j]]+"-"+str(i))
            #   results = pu.load_results('~/Desktop/logs/EXP2/'+args.env)
            pu.plot_results(results,xy_fn=pu.progress_mbl_vbest_xy_fn,average_group=True,split_fn=lambda _: '')
            #plt.title(args.env+" Online Evaluation")
            plt.xlabel('Number of Timesteps [M]')
            plt.ylabel('Average Return [-]')
            fig = plt.gcf()
            fig.set_size_inches(9.5, 7.5)
            fig.savefig(args.env+"_"+dct[algo_names[j]]+'_'+str(i)+args.filename)
    
if __name__ == '__main__':
    main()

