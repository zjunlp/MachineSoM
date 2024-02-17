import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw_consistent(data, file_name):

    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    colors = ['#ff595e','#ff924c','#ffca3a','#8ac926', '#52a675','#1982c4','#4267ac','#6a4c93']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))

    for i, ax in enumerate(axes, start=1):
        dataset_name = ['MMLU', 'MATH', 'Chess Move Validity'][i - 1]
        lines_data = data[dataset_name]

        for j, line_data in enumerate(lines_data):
            marker = ['o', 's', '^', 'v', '<', '>', 'p', '*'][j]
            marker = ['o', '^', 's', 'D', 'p', '*', 'P', 'H', 'X', 'h'][j]
            linestyles = ["solid","dashed","solid","dashed","solid","dashed","solid","dashed"]

            x_values = [1, 2, 3, 4]
            x_values = [0, 1, 2, 3]
            ax.plot(x_values, line_data, marker=marker, linewidth=2, alpha=0.7, markersize=13, 
            linestyle=linestyles[j], markerfacecolor='none',markeredgewidth=1.5, color=colors[j])

        ax.set_title(f'{dataset_name}')
        # ax.set_xlabel(dataset_name)

        ax.set_xlabel('Round')
        ax.set_ylabel('Average Quantity of\nConsensus Clusters')

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.grid(True, linestyle='--')
    fig.legend(['$p_0p_0p_0$', '$p_0p_0p_1$', '$p_0p_1p_0$', '$p_0p_1p_1$', '$p_1p_0p_0$', '$p_1p_0p_1$', '$p_1p_1p_0$',
                '$p_1p_1p_1$'], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=8, 
                 fontsize='small',  columnspacing=1.0)

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # if not save:
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)
    # else:
    #     plt.savefig("consistent_t.pdf", format='pdf', bbox_inches='tight', pad_inches=0.0)

if __name__ == '__main__':
    data = {
        'MMLU': [
            [2.246, 2.477, 2.641, 2.562],
            [2.238, 2.497, 2.607, 2.655],
            [2.241, 2.507, 2.525, 2.643],
            [2.235, 2.481, 2.526, 2.535],
            [2.261, 2.327, 2.487, 2.627],
            [2.253, 2.353, 2.523, 2.53 ],
            [2.218, 2.33, 2.328, 2.54 ],
            [2.224, 2.358, 2.345, 2.341]
        ],
        'MATH': [
            [0.493, 1.345, 1.951, 2.253],
            [0.506, 1.366, 2.027, 1.967],
            [0.494, 1.34, 1.291, 1.933],
            [0.522, 1.329, 1.309, 1.303],
            [0.485, 0.393, 1.407, 1.99 ],
            [0.484, 0.396, 1.405, 1.406],
            [0.519, 0.413, 0.408, 1.442],
            [0.516, 0.389, 0.379, 0.381]
        ],
        'Chess Move Validity': [
            [1.255, 1.697, 1.902, 1.964],
            [1.34, 1.713, 1.899, 1.894],
            [1.331, 1.677, 1.536, 1.806],
            [1.297, 1.637, 1.473, 1.358],
            [1.324, 0.832, 1.388, 1.508],
            [1.294, 0.788, 1.341, 1.033],
            [1.308, 0.751, 0.468, 1.158],
            [1.307, 0.805, 0.465, 0.336]
        ]
    }
    draw_consistent(data, "1.png")