import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from data.celeba import DataModuleClipCelebA
from utils.figure_helper import set_matplotlib_sns_params, get_shifted_cmap

def read_tsv(filename: str) -> list[list]:
    assert filename.endswith('.tsv')
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = []
            tokens = line.split('\t')
            for i, token in enumerate(tokens):
                if i in [0, 1]:
                    values.append(int(token))
                    continue
                try:
                    token = float(token)
                except ValueError:
                    token = 0.0
                values.append(token)
            data.append(values)
    return data

def get_drop_ratio_attributes(
    data: list[list],
    tradeoff: float = 10.0,
    restricted_id: int = 0,
):
    max_inner_epoch = np.max([values[1] for values in data])
    data = [ values for values in data if values[1] == max_inner_epoch ]
    for i, values in enumerate(data):
        outer_epoch, _, _, _, _, _, *eval_list = values
        if outer_epoch == 0:
            original_eval_list = np.array(eval_list)
            break
    drop_o = np.array([ data[0][5] - values[5] for values in data ])
    closest_idx = np.argmin(tradeoff - drop_o)

    obstructed_eval_list = np.array(data[closest_idx][6:])

    drop_ratios = (original_eval_list - obstructed_eval_list) / (original_eval_list[restricted_id] - obstructed_eval_list[restricted_id] + 1e-12)
    return drop_ratios

def main():
    set_matplotlib_sns_params()

    datamodule = DataModuleClipCelebA()
    attributes = datamodule.attributes
    num_attributes = len(attributes)

    exp_root = 'experiments'
    defender = attacker = 'ce'
    dataset = 'celeba'
    tradeoff = 2.0
    max_value = 2.0
    min_value = 0.0
    obstructor = 'omaml'
    obs_title = 'Ours'
    start_attr_idx = 1 # Ignore the first attributes because it is used for validation

    fig, ax = plt.subplots(layout='constrained')
    confusion_matrix = []    
    for attr_idx, attr in enumerate(attributes):
        if attr_idx < start_attr_idx:
            continue
        filename = os.path.join(
            exp_root,
            dataset,
            f'{obstructor}_attr-{defender}-k5',
            f'results-{attacker}_attr-k5',
            f'{attr_idx}.tsv'
        )
        if not os.path.exists(filename):
            continue
        data = read_tsv(filename)
        drop_ratios = get_drop_ratio_attributes(
            data, tradeoff=tradeoff, restricted_id=attr_idx)
        confusion_matrix.append(drop_ratios)

    confusion_matrix = np.array(confusion_matrix)
    confusion_matrix = confusion_matrix[:, start_attr_idx:]
    confusion_matrix = np.minimum(confusion_matrix, max_value)
    confusion_matrix = np.maximum(confusion_matrix, min_value)
    
    midpoint = (1 - confusion_matrix.min()) / (confusion_matrix.max() - confusion_matrix.min())
    shifted_cmap = get_shifted_cmap(
        matplotlib.cm.coolwarm, midpoint=midpoint, 
        name=f'shifted')
    im = ax.imshow(
        confusion_matrix, cmap=shifted_cmap, interpolation="nearest")

    ax.set_xticks(range(num_attributes - start_attr_idx))
    ax.set_yticks([])
    ax.set_xticklabels(attributes[start_attr_idx:], rotation=90)
    ax.set_yticklabels([])
    ax.set_title(obs_title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['< 0.0', '1.0', '> 2.0'])
    cbar.ax.set_ylabel("$M_{a{a^'}}$", loc='center')
        
    ax.set_yticks(range(num_attributes - start_attr_idx))
    ax.set_yticklabels(attributes[start_attr_idx:])
 
    fig.set_size_inches(14, 5)
    
    plt.savefig(f'confusion-{dataset}.png', bbox_inches='tight')
    plt.savefig(f'confusion-{dataset}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()