import os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
from scipy.stats import ttest_ind
def arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def filt(results,name):
    ls=[r for r in results if name in r.dirname]
    return ls
def reconstruct(max_idx,legend_name_list,mn_ls,sd_ls,last_ls_ls):
    new_legends,new_mn_ls,new_sd_ls,new_last_ls_ls=[],[],[],[]
    for i in range(len(mn_ls)):
        if i != max_idx:
            new_mn_ls.append(mn_ls[i])
            new_sd_ls.append(sd_ls[i])
            new_last_ls_ls.append(last_ls_ls[i])
            new_legends.append(legend_name_list[i])
    return new_legends,new_mn_ls,new_sd_ls,new_last_ls_ls
def t_test(a,b):
    values1=np.array(a)
    values2=np.array(b)
    value, p = ttest_ind(values1, values2, equal_var=False)
    if p>0.05: bl=True
    else: bl=False
    return bl
def main():  
    parser = arg_parser()
    parser.add_argument('--dir', type=str, default='logs')
    parser.add_argument('--thesis', type=str, default='Online_VF1')
    args = parser.parse_args()
   
    location=args.dir
    thesis_dir=args.thesis
    env_name_list=["Ant-v2", "HalfCheetah-v2", "Reacher-v2", "Swimmer-v2"]
    #env_name_list=["HalfCheetah-v2"]
    legend_name_list=["copos1", "copos1+sil_n2_l0.001", "copos1+sil_n10_l0.1", 
                      "copos2", "copos2+sil_n2_l0.001", "copos2+sil_n10_l0.1",
                      "ppo", "ppo+sil_n2_l0.001", "ppo+sil_n10_l0.1",
                      "trpo", "trpo+sil_n2_l0.001", "trpo+sil_n10_l0.1"]
    #legend_name_list=["copos1", "copos1+sil_n2_l0.001", "copos1+sil_n10_l0.1"]
    for env_name in env_name_list:
        dirname = '~/Desktop/logs/'+location+'/EXP_ON_VF1/'+env_name
        results = pu.load_results(dirname)
        mn_ls, sd_ls,last_ls_ls=[],[],[]
        final_txt_name="/Users/zsbjltwjj/Desktop/thesis/img/"+thesis_dir+"/"+env_name+"-final-output.txt"
        for legend in legend_name_list:
            result=filt(results,legend+"-")
            mn, sd, last_ls = pu.table_results(result,xy_fn=pu.progress_default_xy_fn,average_group=True,split_fn=lambda _: '', 
                                               name=result[0].dirname,tp='online',freq=50)
            txt_name="/Users/zsbjltwjj/Desktop/logs/"+location+"/EXP_ON_VF1/"+env_name+"/"+legend+"-output.txt"
            with open(txt_name, "w") as text_file:
                text_file.write(str(mn)+'\n')
                text_file.write(str(sd)+'\n')
                for i in last_ls:
                    text_file.write(str(i)+' ')
    #        s=open(txt_name, "r")
    #        tmp=s.readlines()
    #        s.close()
            mn_ls.append(mn)
            sd_ls.append(sd)
            last_ls_ls.append(last_ls)
        #print(mn_ls)
        max_idx=np.argmax(mn_ls)
        with open(final_txt_name, "w") as txt_file:
            bolds=[]
            new_legends,new_mn_ls,new_sd_ls,new_last_ls_ls=reconstruct(max_idx,legend_name_list,mn_ls,sd_ls,last_ls_ls)
            for i in range(len(new_legends)):
                bold=t_test(last_ls_ls[max_idx],new_last_ls_ls[i])
                bolds.append(bold)
                txt_file.write(new_legends[i]+": "+str(new_mn_ls[i])+' '+str(new_sd_ls[i])+' '+str(bold)+'\n')
            if any(bolds): max_bold=True
            else: max_bold=False
            txt_file.write("max alg: "+legend_name_list[max_idx]+": "+str(mn_ls[max_idx])+' '+str(sd_ls[max_idx])+' '+str(max_bold)+'\n')

if __name__ == '__main__':
    main()

