import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw_strategy(datasource:dict, file_name:str):
    # plt.rcParams.update({'font.size': 14})
    def latex(value:int):
        binary:str = bin(value)[2:].zfill(3)
        ans = "$"
        for i in binary:
            ans = f"{ans}p_{i}"
        ans = ans + "$"
        space = 8
        return f'{" "*space}{ans}\n{" "*space}All→Part'
        # return " "*8 + ans + "\nAll→Part"
    # datasource = {
    #     'data_mmlu': np.random.uniform(30,60,(8,2)).reshape(-1).tolist(),
    #     'data_chess': np.random.uniform(30,60,(8,2)).reshape(-1).tolist(),
    #     'errors_mmlu': np.random.uniform(5,10, (8,2)).reshape(-1).tolist(),
    #     'errors_chess': np.random.uniform(5,10, (8,2)).reshape(-1).tolist(),
    # }
    y1, y3 = datasource['data_mmlu'], datasource['data_chess']
    variance1, variance3 = datasource['errors_mmlu'], datasource['errors_chess']
    x_labels = [latex(i//2) if i%2==0 else '' for i in range(16)]

    # space = "        "
    # x_labels = [space+'000', '', space+'001', '', space+'010', '', space+'011', '', space+'100', '', space+'101', '', space+'110', '', space+'111', '']

    # x_labels = []
    # for i in range(16):
    #     if i % 2 == 0:
    #         x_labels.append(f"{space}{latex(label['3'][i//2])}\n{space}All→Part")
    #     else:
    #         x_labels.append('')

    # y3 = merge(chess["main"]["1 Harmony"]["mean"], chess["strategy"]["1 Harmony"]["mean"])
    # variance3 = merge(chess["main"]["1 Harmony"]["var"], chess["strategy"]["1 Harmony"]["var"])

    # y1 = merge(mmlu["main"]["1 Harmony"]["mean"], mmlu["strategy"]["1 Harmony"]["mean"])
    # variance1 = merge(mmlu["main"]["1 Harmony"]["var"], mmlu["strategy"]["1 Harmony"]["var"])

    # y1 = [_/50*100 for _ in y1]
    # y3 = [_/50*100 for _ in y3]
    # variance1 = [_/50*100 for _ in variance1]
    # variance3 = [_/50*100 for _ in variance3]

    show_labels = ["S1/MMLU", "S3/MMLU", "S1/Chess", "S3/Chess"]
    show_dot_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000']
    show_var_colors = ['#5C75AE', '#EA9490', '#87DAAD', '#FFE596']
    show_dot_colors = ['#C30E23', '#C30E23', '#C30E23', '#C30E23']
    show_var_colors = ['#EA9490', '#EA9490', '#EA9490', '#EA9490']

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 2.5))

    ax1.scatter(range(len(x_labels)), y1, color=show_dot_colors[0], label=show_labels[0])
    for i in range(0, len(x_labels), 2):
        ax1.plot([i, i + 1], [y1[i], y1[i + 1]], c=show_dot_colors[0])
    ax1.errorbar(range(len(x_labels)), y1, yerr=variance1, fmt='o', color=show_dot_colors[0], ecolor=show_var_colors[0], capsize=5)
    ax1.set_ylabel('Accuracy(%)')


    ax3.scatter(range(len(x_labels)), y3, color=show_dot_colors[2], label=show_labels[2])
    for i in range(0, len(x_labels), 2):
        ax3.plot([i, i + 1], [y3[i], y3[i + 1]], c=show_dot_colors[2])
    ax3.errorbar(range(len(x_labels)), y3, yerr=variance3, fmt='o', color=show_dot_colors[2], ecolor=show_var_colors[2], capsize=5)
    ax3.set_ylabel('Accuracy(%)')


    # plt.xticks(range(len(x_labels)), x_labels)
    ax1.set_xticks(range(len(x_labels)))
    ax3.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax3.set_xticklabels(x_labels)

    ax1.set_title("MMLU")
    ax1.set_title("MATH")
    ax3.set_title("Chess Move Validity")

    # lines, labels = ax1.get_legend_handles_labels()
    # lines3, labels3 = ax3.get_legend_handles_labels()
    # ax3.legend(lines +lines3, labels+ labels3, loc='lower left', ncol=2, handletextpad=0.1)

    # plt.suptitle('Scatter Plot with Variance (Two Data Sets)', fontsize=16)

    plt.tight_layout(h_pad=1.5)

    # if not save:
    #     plt.show()
    # else:
    #     plt.savefig("strategy.pdf", format='pdf', bbox_inches='tight', pad_inches=0.0)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  


def draw_three_strategy(datasource:dict, file_name:str):
    # plt.rcParams.update({'font.size': 14})
    def latex(value:int):
        binary:str = bin(value)[2:].zfill(3)
        ans = "$"
        for i in binary:
            ans = f"{ans}p_{i}"
        ans = ans + "$"
        space = 8
        return f'{" "*space}{ans}\n{" "*space}All→Part'
    
    y1, y2, y3 = datasource['data_mmlu'], datasource['data_math'], datasource['data_chess']
    variance1, variance2, variance3 = datasource['errors_mmlu'], datasource['errors_math'], datasource['errors_chess']
    x_labels = [latex(i//2) if i%2==0 else '' for i in range(16)]

    show_labels = ["S1/MMLU", "S3/MMLU", "S1/Chess", "S3/Chess"]
    show_dot_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000']
    show_var_colors = ['#5C75AE', '#EA9490', '#87DAAD', '#FFE596']
    # show_dot_colors = ['#C30E23', '#C30E23', '#C30E23', '#C30E23']
    # show_var_colors = ['#EA9490', '#EA9490', '#EA9490', '#EA9490']

    # fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 2.5))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,7.5))

    ax1.scatter(range(len(x_labels)), y1, color=show_dot_colors[0], label=show_labels[0])
    for i in range(0, len(x_labels), 2):
        ax1.plot([i, i + 1], [y1[i], y1[i + 1]], c=show_dot_colors[0])
    ax1.errorbar(range(len(x_labels)), y1, yerr=variance1, fmt='o', color=show_dot_colors[0], ecolor=show_var_colors[0], capsize=5)
    ax1.set_ylabel('Accuracy(%)')

    ax2.scatter(range(len(x_labels)), y2, color=show_dot_colors[1], label=show_labels[1])
    for i in range(0, len(x_labels), 2):
        ax2.plot([i, i + 1], [y2[i], y2[i + 1]], c=show_dot_colors[1])
    ax2.errorbar(range(len(x_labels)), y2, yerr=variance2, fmt='o', color=show_dot_colors[1], ecolor=show_var_colors[1], capsize=5)
    ax2.set_ylabel('Accuracy(%)')


    ax3.scatter(range(len(x_labels)), y3, color=show_dot_colors[2], label=show_labels[2])
    for i in range(0, len(x_labels), 2):
        ax3.plot([i, i + 1], [y3[i], y3[i + 1]], c=show_dot_colors[2])
    ax3.errorbar(range(len(x_labels)), y3, yerr=variance3, fmt='o', color=show_dot_colors[2], ecolor=show_var_colors[2], capsize=5)
    ax3.set_ylabel('Accuracy(%)')


    # plt.xticks(range(len(x_labels)), x_labels)
    ax1.set_xticks(range(len(x_labels)))
    ax2.set_xticks(range(len(x_labels)))
    ax3.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax2.set_xticklabels(x_labels)
    ax3.set_xticklabels(x_labels)

    ax1.set_title("MMLU")
    ax2.set_title("MATH")
    ax3.set_title("Chess Move Validity")

    # lines, labels = ax1.get_legend_handles_labels()
    # lines3, labels3 = ax3.get_legend_handles_labels()
    # ax3.legend(lines +lines3, labels+ labels3, loc='lower left', ncol=2, handletextpad=0.1)

    # plt.suptitle('Scatter Plot with Variance (Two Data Sets)', fontsize=16)

    plt.tight_layout(h_pad=1.5)

    # if not save:
    #     plt.show()
    # else:
    #     plt.savefig("strategy.pdf", format='pdf', bbox_inches='tight', pad_inches=0.0)
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  


if __name__ == '__main__':
    draw_strategy()