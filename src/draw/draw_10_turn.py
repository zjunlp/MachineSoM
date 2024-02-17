import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw_10_turn(database:dict, file_name:str):
    titles = database['title']
    data, error = database['data'], database['error']
    # x_labels = ['0']
    # for i in range(1,11):
    #     x_labels.append
    # x_labels = ['$p_{'+str(i) +'}$' for i in range(11)]
    # print(x_labels)
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18,5))

    # Iterate over each subplot
    for i, ax in enumerate(axes.flatten()):
        x = np.arange(1, 9)  # Fixed x values from 1 to 9
        y = data[i]# [1:]
        yerr = error[i]# [1:]
        x_labels = [] #['$0$']
        strategy = titles[i][1:-1].replace('p','').replace('_','')
        print(strategy)
        for j in range(3, 11):
            x_labels.append(f'{j}\n($p_{strategy[j-1]}$)')
        print(x_labels)
        # Plotting
        print(x.shape, y.shape)
        ax.errorbar(x, y, yerr=yerr, fmt='-o', color='#3257A6', ecolor='#5C75AE', elinewidth=2, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(titles[i])
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle='--',)
        # ax.hlines(y=data[i][0], xmin=min(x), xmax=max(x), colors='#C30E23', linestyles='--', alpha=0.5)
        # ax.fill_between(x, data[i][0]-error[i][0], data[i][0]+error[i][0], color='#EA9490', alpha=0.5)
        # print(data[i][0],error[i][0])
    plt.tight_layout()
    # plt.show()
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  

