import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def filt(results,name,name_2=''):
    ls=[r for r in results if name in r.dirname and name_2 in r.dirname]
    return ls
def filt_or(results,name,name_2):
    ls=[r for r in results if name in r.dirname or name_2 in r.dirname]
    return ls
def filt_or_or_or(results,name,name_2,name_3,name_4):
    ls=[r for r in results if name in r.dirname or name_2 in r.dirname or name_3 in r.dirname or name_4 in r.dirname]
    return ls
def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--dir', type=str, default='logs')
    parser.add_argument('--thesis', type=str, default='Offline_VF1')
    args = parser.parse_args()
#    dirname = '~/Desktop/carla_sample_efficient/data/bk/bkup_EXP2_FINAL/'+args.extra_dir+args.env
    dirname = '~/Desktop/logs/'+args.dir+'/EXP_OFF_24_420K_VF1/'+args.env

    results = pu.load_results(dirname)
    r_copos1_nosil,r_copos2_nosil,r_trpo_nosil,r_ppo_nosil=filt(results,'copos1-'),filt(results,'copos2-'),filt(results,'trpo-'),filt(results,'ppo-')
    r_copos1_sil,r_copos2_sil,r_trpo_sil,r_ppo_sil=filt(results,'copos1+sil-'),filt(results,'copos2+sil-'),filt(results,'trpo+sil-'),filt(results,'ppo+sil-')
    r_mbl_sil=filt(results,'mbl+','sil-')
 #   r_mbl_nosil_tmp=[r for r in results if r not in r_mbl_sil]
    r_mbl_nosil=filt_or_or_or(results,'mbl+copos1-','mbl+copos2-','mbl+trpo-','mbl+ppo-')

    r_copos1_comp, r_copos2_comp, r_trpo_comp, r_ppo_comp=filt_or(results,'mbl+copos1','copos1+sil'),filt_or(results,'mbl+copos2','copos2+sil'),filt_or(results,'mbl+trpo','trpo+sil'),filt_or(results,'mbl+ppo','ppo+sil')
    
    dt={'copos1_nosil':r_copos1_nosil,'copos2_nosil':r_copos2_nosil, 'trpo_nosil':r_trpo_nosil, 'ppo_nosil':r_ppo_nosil,
        'copos1_sil':r_copos1_sil,'copos2_sil':r_copos2_sil, 'trpo_sil':r_trpo_sil, 'ppo_sil':r_ppo_sil,
        'mbl_nosil':r_mbl_nosil, 'mbl_sil':r_mbl_sil,
        'copos1_comp':r_copos1_comp,'copos2_comp':r_copos2_comp, 'trpo_comp':r_trpo_comp, 'ppo_comp':r_ppo_comp}

    for name in dt:
        pu.plot_results(dt[name],xy_fn=pu.progress_mbl_vbest_xy_fn,average_group=True,name=name,split_fn=lambda _: '',shaded_err=True,shaded_std=False)
        plt.xlabel('Number of Timesteps [M]')
        plt.ylabel('Best Average Return [-]')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(9, 7.5)
 #       fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/OFFLINE/"+args.extra_dir+args.env+'/'+name+'.pdf',format="pdf")
        fig.savefig("/Users/zsbjltwjj/Desktop/thesis/img/"+args.thesis+"/"+args.env+'/'+name+'.pdf', format="pdf")
        if name=='mbl_nosil' or name=='mbl_sil':
            pu.plot_results(dt[name],xy_fn=pu.progress_default_entropy_xy_fn,average_group=True,name=name,split_fn=lambda _: '',shaded_err=True,shaded_std=False,legend_entropy=1)
            plt.xlabel('Number of Timesteps [M]')
            plt.ylabel('Entropy [-]')
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(9, 7.5)
 #           fig.savefig("/Users/zsbjltwjj/Desktop/carla_sample_efficient/plot_f/OFFLINE/"+args.extra_dir+args.env+'/'+name+'_entropy.pdf',format="pdf")
            fig.savefig("/Users/zsbjltwjj/Desktop/thesis/img/"+args.thesis+"/"+args.env+'/'+name+'_entropy.pdf', format="pdf")
    
if __name__ == '__main__':
    main()

