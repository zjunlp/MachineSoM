
import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw_turn(data_source:dict, file_name:str):
    plt.rcParams.update({'font.size': 14})
    def latex(value:int):
        binary:str = bin(value)[2:].zfill(4)[1:]
        ans = "$"
        for i in binary:
            ans = f"{ans}p_{i}"
        ans = ans + "$"
        return " "*12 + ans

    # data_source = {
    #     'data_mmlu': np.random.uniform(30,60, (16,3)).reshape(-1).tolist(), # 
    #     'data_chess': np.random.uniform(30,60, (16,3)).reshape(-1).tolist(),
    #     'errors_mmlu': np.random.uniform(5,10, (16,3)).reshape(-1).tolist(),
    #     'errors_chess': np.random.uniform(5,10, (16,3)).reshape(-1).tolist(),
    # }

    y1, y2 = data_source['data_mmlu'], data_source['data_chess']
    variance1, variance2 = data_source['errors_mmlu'], data_source['errors_chess']
    x_labels = [latex(i//3) if i%3==0 else '' for i in range(48)]

    # y1 = merge_turn(0, turn_data)
    # y1 = [_ / 50 * 100 for _ in y1]
    # print(len(y1))
    # variance1 = merge_turn(0, turn_var_data)
    # variance1 = [_ / 50 * 100 for _ in variance1]
    # variance3 = merge(chess["main"]["1 Harmony"]["var"], chess["strategy"]["1 Harmony"]["var"])

    # y2 = merge_turn(1, turn_data)
    # y2 = [_ / 50 * 100 for _ in y2]
    # variance2 = merge_turn(1, turn_var_data)
    # variance2 = [_ / 50 * 100 for _ in variance2]
    # variance1 = merge(mmlu["main"]["1 Harmony"]["var"], mmlu["strategy"]["1 Harmony"]["var"])

    show_labels = ["MMLU", "Chess Move Validity"]
    show_dot_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000']
    show_var_colors = ['#5C75AE', '#EA9490', '#87DAAD', '#FFE596']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 4))
    # plt.xlim(0-0.5, len(x_labels)-1+0.5)
    ax1.set_xlim(0 - 0.5, len(x_labels) - 1 + 0.5)
    ax2.set_xlim(0 - 0.5, len(x_labels) - 1 + 0.5)

    ax1.scatter(range(len(x_labels)), y1, color=show_dot_colors[0], label=show_labels[0])
    for i in range(0, len(x_labels), 3):
        ax1.plot([i, i + 1], [y1[i], y1[i + 1]], c=show_dot_colors[0])
        ax1.plot([i + 1, i + 2], [y1[i + 1], y1[i + 2]], c=show_dot_colors[0])
        ax1.axvline(x=(i+2+i+3)/2, linestyle='-', color='black', linewidth=0.8)

    ax1.errorbar(range(len(x_labels)), y1, yerr=variance1, fmt='o', color=show_dot_colors[0], ecolor=show_var_colors[0],
                 capsize=5)
    ax1.set_ylabel('Accuracy(%)')

    ax2.scatter(range(len(x_labels)), y2, color=show_dot_colors[2], label=show_labels[1])
    for i in range(0, len(x_labels), 3):
        ax2.plot([i, i + 1], [y2[i], y2[i + 1]], c=show_dot_colors[2])
        ax2.plot([i + 1, i + 2], [y2[i + 1], y2[i + 2]], c=show_dot_colors[2])
        ax2.axvline(x=(i+2+i+3)/2, color='black', linestyle='-', linewidth=0.8)
    ax2.errorbar(range(len(x_labels)), y2, yerr=variance2, fmt='o', color=show_dot_colors[2], ecolor=show_var_colors[2],
                 capsize=5)
    ax2.set_ylabel('Accuracy(%)')

    # plt.xticks(range(len(x_labels)), x_labels)
    ax1.set_xticks(range(len(x_labels)))
    ax2.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax2.set_xticklabels(x_labels)

    # lines, labels = ax1.get_legend_handles_labels()
    # lines3, labels3 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines3, labels + labels3, loc='lower left', ncol=2, handletextpad=0.1)

    # plt.suptitle('Scatter Plot with Variance (Two Data Sets)', fontsize=16)
    ax1.set_title(show_labels[0])
    ax2.set_title(show_labels[1])

    ax1.grid(True, linestyle='dashed')
    ax2.grid(True, linestyle='dashed')

    plt.tight_layout(h_pad=1.5)

    # if not save:
    #     plt.show()
    # else:
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)    
    # plt.savefig('10.png')

if __name__ == '__main__':
    draw_turn(None, None)