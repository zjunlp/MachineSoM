import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw_conformity(data, file_name):
    font_size = 10  # Further reduced font size for annotations
    annotation_font_size = 10  # Even smaller font for the annotations
    plt.rcParams.update({'font.size': font_size})
    # data = np.array([
    #     [[257, 645, 6666, 2358], [392, 546, 4830, 3443], [382, 645, 4922, 3244]],
    #     [[186, 662, 3179, 1382], [221, 631, 3647, 2018], [342, 565, 3938, 2659]],
    #     [[292, 513, 2807, 2072], [352, 532, 2904, 2420], [429, 514, 2924, 2988]]
    # ])/12000
    title = ['MMLU', 'MATH', 'Chess Move Validity']
    labels = ['True -> False','False -> True','True -> True','False -> False']
    new_colors = ['#ea9690', '#7030a0', '#00b050', '#395ea1']
    patterns = ['-', '+', 'x', '//']
    patterns_color = ['white', 'white', 'white', 'white']
    
    # Creating the bar plots with updated colors and a shared legend
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    patches = []  # For legend

    for i, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', linewidth=0.5)
        ax.xaxis.grid(False)  # Disable vertical grid lines
        for j in range(3):
            bottom = 0
            for k in range(4):
                value = data[i, j, k]
                bar = ax.bar(j, value, bottom=bottom, color=new_colors[k], width=0.5, hatch=patterns[k],
                             edgecolor=patterns_color[k])
                # Adjust annotation style based on bar length
                if value < 0.05:  # For shorter bars, use line and annotate outside
                    # Position the text to the right of the bar
                    text_x = j + 0.5
                    text_y = bottom + value / 2
                    ax.annotate(
                        f'{value:.2%}',
                        xy=(j + 0.25, bottom + value),  # Pointing to end of the bar
                        xytext=(text_x, text_y),
                        textcoords="data",
                        ha='left',
                        va='center',
                        color=new_colors[k],
                        fontsize=annotation_font_size,
                        arrowprops=dict(arrowstyle="-", color=new_colors[k]),
                        # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", lw=0.72)
                    )
                else:  # For longer bars, annotate inside with smaller background box
                    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0.72)
                    ax.annotate(f'{value:.2%}',
                                xy=(j, bottom + value / 2),
                                ha='center',
                                va='center',
                                color=new_colors[k],
                                fontsize=annotation_font_size,
                                bbox=bbox_props)
                bottom += value
                if i == 0 and j == 0:  # Add to legend only once
                    patches.append(bar)
        ax.set_title(title[i])
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'Round {x + 1}' for x in range(3)])
        ax.set_ylabel('Proportion(%)')
    # Creating a shared legend for all subplots
    fig.legend([p[0] for p in patches], labels, loc='lower center', ncol=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the legend

    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)

if __name__ == '__main__':
    draw_conformity()


