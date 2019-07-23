import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def filt(results,name,name_2=''):
    ls=[r for r in results if name in r.dirname and name_2 in r.dirname]
    return ls
def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--extra_dir', type=str, default='')
    args = parser.parse_args()
    dirname = '~/Desktop/carla_sample_efficient/data/bk/bkup_EXP1_FINAL/'+args.extra_dir+args.env
    
    results = pu.load_results(dirname)
    r_copos,r_trpo,r_ppo=filt(results,'copos'),filt(results,'trpo'),filt(results,'ppo')
    r_sil_n2=filt(results,'sil_n2_l0.001')
    dt={'copos':r_copos, 'trpo':r_trpo, 'ppo':r_ppo, 'sil_n2_l0.001':r_sil_n2}

    for name in dt:
        pu.plot_results(dt[name],xy_fn=pu.progress_default_xy_fn,average_group=True,split_fn=lambda _: '',shaded_err=True,shaded_std=False)
        plt.xlabel('Number of Timesteps [M]')
        plt.ylabel('Average Return [-]')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(9, 7.5)
        fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/ONLINE/"+args.env+'/'+name+'.png')
        if name=='sil_n2_l0.001':
            pu.plot_results(dt[name],xy_fn=pu.progress_default_entropy_xy_fn,average_group=True,split_fn=lambda _: '',shaded_err=True,shaded_std=False,legend_entropy=1)
            plt.xlabel('Number of Timesteps [M]')
            plt.ylabel('Entropy [-]')
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(9, 7.5)
            fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/ONLINE/"+args.env+'/'+name+'_entropy.png')
    
if __name__ == '__main__':
    main()

