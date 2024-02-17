import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def plot_stacked_bar_chart(ax, data, group_labels, title, bar_width=0.8, spacing=1):
    colors = ['skyblue', 'orange', 'limegreen']
    group_width = bar_width * data.shape[1]  # Total width for each group of bars
    group_positions = np.arange(data.shape[0]) * (group_width + spacing)  # Starting position for each group

    # Plotting the bars
    # for i in range(data.shape[0]):
    #     bar_positions = group_positions[i] + np.arange(data.shape[1]) * bar_width
    #     for j in range(data.shape[2]):
    #         ax.bar(
    #             bar_positions, 
    #             data[i, :, j], 
    #             bar_width,
    #             bottom=0, # np.sum(data[i, :, :j], axis=1) if j > 0 else 0,
    #             color=colors[j], 
    #             edgecolor='black'
    #         )
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sorted_indices = np.argsort(-data[i, j])
            sorted_data = data[i, j][sorted_indices]
            sorted_colors = [colors[index] for index in sorted_indices]

            bar_position = group_positions[i] + j * bar_width
            for k in range(data.shape[2]):
                ax.bar(
                    bar_position,
                    sorted_data[k],
                    bar_width,
                    color=sorted_colors[k],
                    edgecolor='black',
                    bottom=0
                )
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         bar_positions = group_positions[i]

    # Setting x-axis labels for groups
    ax.set_xticks(group_positions + bar_width * data.shape[1] / 2)
    ax.set_xticklabels(group_labels)
    ax.set_title(title)

def draw_10_agent_consistent(database, file_name):
    # database = {
    #     'data': np.random.rand(8, 9, 5, 3),
    #     'xlabels': [],      
    #     'titles': []        
    # }
    data = database['data']
    titiles = database['title']
    xlabels = database['x_label']
    assert len(xlabels) == data.shape[1], f'{len(xlabels)} != {data.shape[1]}. {data.shape}'
    assert len(titiles) == 8
    group_labels = xlabels

    # Creating a figure with 8 subplots
    fig, axs = plt.subplots(8, 1, figsize=(15, 48))

    # Plotting each subplot
    for i in range(8):
        plot_stacked_bar_chart(axs[i], data[i], group_labels, titiles[i])

    # Adjusting legend and layout
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels=['Round 1', 'Round 2', 'Round 3'], 
        loc='lower center', 
        # bbox_to_anchor=(0.5, 0), 
        ncol=3
    )
    # fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))
    # plt.subplots_adjust(bottom=0.1)

    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  

