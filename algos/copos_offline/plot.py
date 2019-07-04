import numpy as np
import os
import pandas as pd

import matplotlib
# For draw in bg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# For draw in bg
plt.switch_backend('agg')

from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

import seaborn as sns
sns.set(style="darkgrid")

from visdom import Visdom

INF = 1e9

def load_data_with_transform_best_cmd(dirname, smooth_max=4):
    df = pd.read_csv(os.path.join(dirname, 'progress.csv'))    
    targs_names = {k:k.split('MeanRew')[-1] for k in df.keys() if k.startswith('MeanRew')}
    return _load_data_with_transform_best(df, targs_names, 20,smooth_max)

def load_data_with_transform_best(dirname, targs_names, quant,smooth_max=4):
    df = pd.read_csv(os.path.join(dirname, 'progress.csv'))  
    return _load_data_with_transform_best(df, targs_names, quant,smooth_max)

def _load_data_with_transform_best(df, targs_names, quant,smooth_max):
    targs = (list(targs_names.keys()))
    df = df[targs]
    
    smooth_range = 1 if len(df) <= 20 else smooth_max
    plot_data = [] 

    for targ, name in sorted(targs_names.items()):
        d = df[[targ]].copy(deep=True)
        d['Epoch'] = range(0, len(d))
        d['Epoch'] = d['Epoch'].astype(int)
        d.insert(2, 'type', name)        
        d = d.rename(index=str, columns={targ: 'Reward'})
       
        max_reward_so_far = -INF
        for idx, row in d.iterrows():            
            max_reward_so_far = max(row['Reward'], max_reward_so_far)
            d.at[idx, 'Reward']=max_reward_so_far
        
        d = d[d['Epoch'] % quant == 0]
        d = d.rename(index=str, columns={'Reward': 'Best reward'})
        plot_data.append(d)    
    df = pd.concat(plot_data)
    
    return df, 'Best reward'

def load_data(dirname, targs_names, smooth_max=4):
    df = pd.read_csv(os.path.join(dirname, 'progress.csv'))

    targs = (list(targs_names.keys()))
    df = df[df[targs].notnull()]
    
    smooth_range = 1 if len(df) <= 20 else smooth_max
    
    plot_data = [] 
    for targ, name in sorted(targs_names.items()):
        d = df[[targ]].copy(deep=True)
        d['Epoch'] = range(0, len(d))
        d = d.rolling(smooth_range, center=True, min_periods=1).mean()
        d['Epoch'] = d['Epoch'].astype(int)
        d.insert(2, 'type', name)
        d = d.rename(index=str, columns={targ: 'Reward'})
        plot_data.append(d)    
    df = pd.concat(plot_data)
    return df, 'Reward'

def plot(viz, win, dirname, targs_names, quant=2, smooth=4, opt='each'):
    
    if opt == 'each': 
        data, y_name = load_data(dirname, targs_names, smooth_max=smooth)        
    elif opt == 'best': 
        data, y_name = load_data_with_transform_best(dirname, targs_names, quant=quant, smooth_max=smooth)

    sns_plot = sns.relplot(x="Epoch", y=y_name, hue='type', kind="line", data=data)
    fig = sns_plot.fig
    plt.title(dirname)
    figname = os.path.join(dirname, 'eval_result.png')
    fig.savefig(figname)
    image = plt.imread(figname)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))    
    return viz.image(image, win=win)

def plot_cmd(viz, win, dirname, smooth=4):
    
    data, y_name = load_data_with_transform_best_cmd(dirname, smooth_max=smooth)

    sns_plot = sns.catplot(x="Epoch", y=y_name, hue='type', kind="point", data=data)
    fig = sns_plot.fig
    plt.title(dirname)
    figname = os.path.join(dirname, 'eval_result.png')
    fig.savefig(figname)
    image = plt.imread(figname)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))    
    return viz.image(image, win=win)

if __name__ == "__main__":
    import argparse
    from visdom import Visdom
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--smooth', type=int, default=4)
    args = parser.parse_args()

    viz = Visdom(env='trpo')
    
    plot_cmd(viz, None, args.log_path, args.smooth)
