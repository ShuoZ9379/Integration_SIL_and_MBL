import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seeds', help='number of seeds', type=int, default=1)
    parser.add_argument('--num_timesteps', type=str, default="3e4")
    parser.add_argument('--filename', type=str, default='_Offline Evaluation.png')
    args = parser.parse_args()
    
#    algo_names=["ppo2_sil_online","copos_sil_online","ppo2_online","copos_online"]
#    legend_names=["ppo2+sil","copos+sil","ppo2","copos"]
#    argus=["","","",""]

    algo_names=["mbl_ppo2_sil","mbl_ppo2","ppo2_sil_offline","ppo2_offline",
                "mbl_copos_sil","mbl_copos","copos_sil_offline","copos_offline",
                "mbl_trpo_sil","mbl_trpo","trpo_sil_offline","trpo_offline"]
#    algo_names=["mbl_ppo2","ppo2_offline",
#                "mbl_copos","copos_offline"]
    legend_names=["mbl+ppo2+sil","mbl+ppo2","ppo2+sil","ppo2",
                  "mbl+copos+sil","mbl+copos","copos+sil","copos",
                  "mbl+trpo+sil", "mbl+trpo", "trpo+sil","trpo"]
#    legend_names=["mbl+ppo2","ppo2",
#                  "mbl+copos","copos"]
    #argus=['--num_samples=1500 --num_elites=10 --horizon=10' for _ in range(len(algo_names))]
    argus=['--num_samples=1 --num_elites=1 --horizon=2' for _ in range(len(algo_names))]

    for i in range(args.seeds):
        for j in range(len(algo_names)):
            os.system("python ../algos/"+algo_names[j]+"/run.py --alg="+algo_names[j]+" --num_timestep="
                      +args.num_timesteps+" --seed="+str(i)+" --env="+args.env+" --log_path=~/Desktop/logs/EXP2/"
                      +args.env+"/"+legend_names[j]+"-"+str(i)+' '+argus[j])

    results = pu.load_results('~/Desktop/logs/EXP2/'+args.env)

    pu.plot_results(results,xy_fn=pu.progress_itermbl_xy_fn,average_group=True,split_fn=lambda _: '')
    #plt.title(args.env+" Online Evaluation")
    plt.xlabel('Evaluation Epochs [-]')
    plt.ylabel('Average Return [-]')
    fig = plt.gcf()
    fig.set_size_inches(10.5, 7.5)
    fig.savefig(args.env+"_"+args.filename)
    
if __name__ == '__main__':
    main()

