import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seeds', help='number of seeds', type=int, default=7)
    parser.add_argument('--num_timesteps', type=str, default="1e6")
    parser.add_argument('--filename', type=str, default='_Online Evaluation2.png')
    args = parser.parse_args()
    
#    algo_names=["ppo2_sil_online","copos_sil_online","ppo2_online","copos_online"]
#    legend_names=["ppo2+sil","copos+sil","ppo2","copos"]
#    argus=["","","",""]

    algo_names=["ppo2_sil_online","ppo2_sil_online","ppo2_online",
                "copos_sil_online","copos_sil_online","copos_online"]
    legend_names=["ppo2+sil_n10_l0.1","ppo2+sil_n1_l0.0001","ppo2",
                  "copos+sil_n10_l0.1","copos+sil_n1_l0.0001","copos"]
    argus=['','--sil_update=1 --sil_loss=0.0001','',
           '','--sil_update=1 --sil_loss=0.0001','']
    
    for i in range(3,args.seeds+3):
        for j in range(len(algo_names)):
            os.system("python ../algos/"+algo_names[j]+"/run.py --alg="+algo_names[j]+" --num_timestep="
                      +args.num_timesteps+" --seed="+str(i)+" --env="+args.env+" --log_path=~/Desktop/logs/EXP1/"
                      +args.env+"/"+legend_names[j]+"-"+str(i)+' '+argus[j])

    results = pu.load_results('~/Desktop/logs/EXP1/'+args.env)

    pu.plot_results(results,xy_fn=pu.progress_iter_xy_fn,average_group=True,split_fn=lambda _: '')
    #plt.title(args.env+" Online Evaluation")
    plt.xlabel('Iterations [-]')
    plt.ylabel('Average Return [-]')
    fig = plt.gcf()
    fig.set_size_inches(10.5, 7.5)
    fig.savefig(args.env+"_"+args.filename)
    
if __name__ == '__main__':
    main()

