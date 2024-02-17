
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.hatch import Shapes, _hatch_types
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.path import Path   # Path 对象
import matplotlib.patches as patches
import matplotlib
from .utils import ensure_directories_exist


distribute_data = {
    '知错能改': [[120, 113, 132, 109, 188, 165, 152, 117], [153, 140, 136, 85, 200, 127, 131, 48], [54, 72, 61, 41, 107, 86, 77, 41]], 
    '对了也改': [[42, 49, 80, 64, 94, 87, 93, 85], [54, 65, 82, 118, 140, 154, 146, 221], [66, 71, 177, 184, 377, 332, 175, 165]], 
    '漂浮不定': [[25, 44, 26, 44, 22, 36, 50, 53], [33, 82, 50, 102, 56, 163, 107, 131], [17, 33, 20, 30, 68, 119, 110, 223]], 
    '起始是对': [[406, 389, 407, 388, 398, 399, 394, 381], [388, 393, 381, 412, 376, 388, 407, 382], [660, 670, 667, 681, 679, 680, 661, 688]],  
    '结果是对': [[489, 470, 467, 445, 491, 478, 466, 413], [484, 477, 457, 385, 429, 318, 410, 169], [646, 666, 554, 543, 391, 406, 533, 428]],

    'cost': [[4364, 3510, 3295, 2665, 3476, 2651, 2691, 1976],[4439, 3965, 3857, 3414, 3840, 3234, 3482, 2681],[3046, 2611, 2604, 2179, 2705, 2251, 2252, 1830]],
    # 
    'self-consistent cost': [1976/3, 2681/3, 1830/3],
}



def draw_left(data, axes:list, mean_values:list, bar_color:str, line_color:str, x_labels:list, titles:list, y_label:str, legend_label:str, line_label:str, hatch=None) -> list:

    bar_wdith = 0.3
    assert data.shape[1] == len(x_labels)
    for dataset_id, ax in enumerate(axes):
        line1 = ax.bar(
            range(len(data[dataset_id])), data[dataset_id], color=bar_color, 
            # edgecolor='white', 
            hatch=hatch, label=legend_label, # width=bar_wdith
        )
        ax.set_xticks(range(len(data[dataset_id])), x_labels, rotation=25)
        ax.set_ylabel(y_label)
        ax.set_xlabel(titles[dataset_id])
        line2 = ax.axhline(y=mean_values[dataset_id], color=line_color, linewidth=2, label=line_label, linestyle='--', zorder=3)
    return [line1, line2]


def draw_right(data, axes:list, legend_labels:list, hatches:list, colors: list, x_labels: list, y_label:str, titles:list) -> list:
    # data: np.array([3,3,8])   n_dataset n_group n_bar
    lines = []
    base_x = np.array(list(range(data.shape[-1])))
    bar_wdith = 0.3
    assert len(data) == len(axes)
    assert len(titles) == data.shape[0]
    assert len(hatches) == data.shape[1]
    assert len(legend_labels) == data.shape[1]
    assert len(colors) == len(legend_labels), f"{len(colors)}, {len(legend_labels)}"
    for dataset_id, ax in enumerate(axes):
        for group_id in range(data.shape[1]):
            line = ax.bar(
                base_x + (group_id-1)*bar_wdith, 
                data[dataset_id][group_id], 
                width=bar_wdith, 
                label=legend_labels[group_id],
                hatch=hatches[group_id],
                color=colors[group_id],
                # edgecolor='black',
            )
            if dataset_id == 0:
                lines.append(line)
        ax.set_ylabel(y_label)
        ax.set_xlabel(titles[dataset_id])
        ax.set_xticks(range(len(data[dataset_id][group_id])), x_labels)
    return lines

def draw_distribution(distribute_data, file_name):
    plt.rcParams.update({'font.size': 15})
    # plt.rcParams['hatch.linewidth'] = 5
    # plt.rcParams["hatch.color"] = '#3257A6'
    plotGroup = [1,1,3]                             # 
    num_lines = len(distribute_data['cost'])        # 
    fig, axs = plt.subplots(num_lines, len(plotGroup), figsize=(24, 15), gridspec_kw={'width_ratios':plotGroup})
    
    
    x_labels = [
        '$p_0p_0p_0$', '$p_0p_0p_1$', '$p_0p_1p_0$', '$p_0p_1p_1$',
        '$p_1p_0p_0$', '$p_1p_0p_1$', '$p_1p_1p_0$', '$p_1p_1p_1$'
    ]
    titles = [
        ['(a) Cost tokens of MMLU.', '(b) Cost tokens of MATH.', '(c) Cost tokens of Chess Move Validity.'],
        ['(d) Accuracy of MMLU.', '(e) Accuracy of MATH.', '(f) Accuracy of Chess Move Validity.'],
        ['(g) Behaviour of MMLU.', '(h) Behaviour of MATH.', '(i) Behaviour of Chess Move Validity.']
    ]
    y_labels = ['Cost Tokens', 'Accruacy (%)', 'Proportion (%)']
    line_labels = ['Cost Tokens of Self-consistent', 'Accuracy of Self-consistent']
    legend_labels = [
        'Cost Tokens after 3-rounds collaboration', line_labels[0], 'Accuracy after 3-rounds collaboration', line_labels[1],
        "Percentage of correcting mistakes", "Percentage of changing correct answers", "Percentage of wavering answers"
    ]

    bar_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000', '#7030A0']
    line_colors = ['#ffd500', '#0b132b']
    hatches = ['s','*', 'D', '..', 'x']
    hatches = [None for i in range(5)]
    scaler = 1/1000*100

    # bar_colors = ['#79addc', '#81B29A', '#EAB69F', '#babf95','#f2cc8f']
    # bar_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000', '#7030A0']
    line_colors = ['#0353a4', '#0b132b']
    hatches = ['//','\\\\', '+', 'O', 'xx']
    hatches = [None for i in range(5)]


    legend_lines = list()
    for i, (key, scale, mean_value) in enumerate(
        zip(
            ['cost', '结果是对'], 
            [1, scaler], 
            [np.array(distribute_data['self-consistent cost']), np.array([distribute_data['起始是对'][_][0] for _ in range(3)])]
        )
    ):
        legend_lines.extend( 
            draw_left(
                data=np.array(distribute_data[key])*scale,
                axes=axs[:,i],
                mean_values=mean_value*scale,
                bar_color=bar_colors[i],
                line_color=line_colors[i],
                x_labels=x_labels,
                titles=titles[i],
                y_label=y_labels[i],
                legend_label=legend_labels[i*2],
                line_label=line_labels[i],
                hatch=hatches[i]
            )
        )
    
    data = np.array(
        [distribute_data['知错能改'], distribute_data['对了也改'], distribute_data['漂浮不定']]
    )   # [3,3,8] [n_group, n_dataset, n_strategy]
    legend_lines.extend(
        draw_right(
            data=data.transpose(1,0,2)*scaler,
            axes=axs[:,2],
            hatches=hatches[2:],
            colors=bar_colors[2:],
            x_labels=x_labels,
            y_label=y_labels[2],
            titles=titles[2],
            legend_labels=legend_labels[4:]
        )
    )

    for i in range(num_lines):
        for j in range(len(plotGroup)):
            axs[i, j].grid(axis='y', linestyle='--',  color='gray', alpha=0.3, zorder=-5)

    fig.legend(legend_lines, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.03, 1, 1])


    # plt.tight_layout(h_pad=1.5)
    
    ensure_directories_exist(file_name)

    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)

   
if __name__ == '__main__':
    draw_distribution(distribute_data, "1.pdf")