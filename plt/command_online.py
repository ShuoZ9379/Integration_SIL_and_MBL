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
    parser.add_argument('--st_seed', help='start number of seeds', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='copos1_online')
    parser.add_argument('--num_timesteps', type=str, default="1e7")
    parser.add_argument('--argu', type=str, default='+sil_n10_l0.1')
    parser.add_argument('--filename', type=str, default='_Online_Evaluation.png')
    args = parser.parse_args()
    
#    algo_names=["ppo2_sil_online","copos_sil_online","ppo2_online","copos_online"]
#    legend_names=["ppo2+sil","copos+sil","ppo2","copos"]
#    argus=["","","",""]

#     algo_names=["ppo2_sil_online","ppo2_sil_online","ppo2_online",
 #                "copos1_sil_online","copos1_sil_online","copos1_online",
#                 "copos2_sil_online","copos2_sil_online","copos2_online",
#                 "trpo_sil_online","trpo_sil_online","trpo_online"]
#    legend_names=["ppo+sil_n10_l0.1","ppo+sil_n2_l0.001","ppo",
#                  "copos1+sil_n10_l0.1","copos1+sil_n2_l0.001","copos1",
#                  "copos2+sil_n10_l0.1","copos2+sil_n2_l0.001","copos2",
#		  "trpo+sil_n10_l0.1","trpo+sil_n2_l0.001","trpo"]
#    argus=['','--sil_update=2 --sil_loss=0.001','',
#           '','--sil_update=2 --sil_loss=0.001','',
#           '','--sil_update=2 --sil_loss=0.001','',
#           '','--sil_update=2 --sil_loss=0.001','']

    dct={"ppo2_sil_online":'ppo',"ppo2_sil_online":'ppo',"ppo2_online":'ppo',
         "copos1_sil_online":'copos1',"copos1_sil_online":'copos1',"copos1_online":'copos1',
         "copos2_sil_online":'copos2',"copos2_sil_online":'copos2',"copos2_online":'copos2',
         "trpo_sil_online":'trpo',"trpo_sil_online":'trpo',"trpo_online":'trpo'}
    argu_dct={'':'',
              '+sil_n10_l0.1':'',
              '+sil_n2_l0.001':'--sil_update=2 --sil_loss=0.001'}
    algo_names=[args.alg]
    legend_names=[dct[args.alg]+args.argu]
    argus=[argu_dct[args.argu]]
    
    for i in range(args.st_seed, args.st_seed+args.seeds):
        for j in range(1):
            os.system("python ~/Desktop/carla_sample_efficient/algos/"+algo_names[j]+"/run.py --alg="+algo_names[j]+" --num_timestep="
                      +args.num_timesteps+" --seed="+str(i)+" --env="+args.env+" --log_path=~/Desktop/logs/EXP_ON_V0/"
                      +args.env+"/"+legend_names[j]+"-"+str(i)+' '+argus[j])

#    results = pu.load_results('~/Desktop/logs/EXP_ON_V0/'+args.env)

#    pu.plot_results(results,xy_fn=pu.progress_default_xy_fn,average_group=True,split_fn=lambda _: '')
#    plt.title(args.env+" Online Evaluation")
#    plt.xlabel('Number of Timesteps [M]')
#    plt.ylabel('Average Return [-]')
#    fig = plt.gcf()
#    fig.set_size_inches(9.5, 7.5)
#    fig.savefig(args.env+"_SUM"+args.filename)
    
if __name__ == '__main__':
    main()

