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
    parser.add_argument('--env', help='environment ID', type=str, default='Swimmer-v2')
    parser.add_argument('--dir', type=str, default='EXP_ON_fix_norm')
    parser.add_argument('--thesis', type=str, default='Online_V0')
    args = parser.parse_args()
#    dirname = '~/Desktop/carla_sample_efficient/data/bk/bkup_EXP1_FINAL/'+args.extra_dir+args.env
    dirname = '~/Desktop/ppo_test/'+args.dir+'/'+args.env
    results = pu.load_results(dirname)
#    r_copos1,r_copos2,r_trpo,r_ppo=filt(results,'copos1'),filt(results,'copos2'),filt(results,'trpo'),filt(results,'ppo')
#    r_sil_n2=filt(results,'sil_n2_l0.001')
#    dt={'copos1':r_copos1, 'copos2':r_copos2,'trpo':r_trpo, 'ppo':r_ppo, 'sil_slight':r_sil_n2}

    r_ppo=filt(results,'ppo')
    dt={'ppo':r_ppo}
    for name in dt:
        pu.plot_results(dt[name],xy_fn=pu.progress_default_xy_fn,average_group=True,split_fn=lambda _: '',shaded_err=True,shaded_std=False)
        plt.xlabel('Number of Timesteps [M]')
        plt.ylabel('Average Return [-]')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(9, 7.5)
#        fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/ONLINE/"+args.extra_dir+args.env+'/'+name+'.pdf', format='pdf')
#        fig.savefig("/Users/zsbjltwjj/Desktop/thesis/img/"+args.thesis+"/"+args.env+'/'+name+'.pdf', format="pdf")
        fig.savefig("/Users/zsbjltwjj/Desktop/ppo_test/"+args.dir+'-'+name+'.pdf', format="pdf")   
#        if name=='sil_slight':
#            pu.plot_results(dt[name],xy_fn=pu.progress_default_entropy_xy_fn,average_group=True,split_fn=lambda _: '',shaded_err=True,shaded_std=False,legend_entropy=1)
#            plt.xlabel('Number of Timesteps [M]')
#            plt.ylabel('Entropy [-]')
#            plt.tight_layout()
#            fig = plt.gcf()
#            fig.set_size_inches(9, 7.5)
#            fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/ONLINE/"+args.extra_dir+args.env+'/'+name+'_entropy.pdf', format="pdf")
#            fig.savefig("/Users/zsbjltwjj/Desktop/thesis/img/"+args.thesis+"/"+args.env+'/'+name+'_entropy.pdf', format="pdf")
    
if __name__ == '__main__':
    main()

