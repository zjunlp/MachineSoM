import matplotlib.pyplot as plt
import numpy as np
from matplotlib.hatch import Shapes, _hatch_types
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.path import Path   # Path 对象
import matplotlib.patches as patches
import matplotlib
from .utils import ensure_directories_exist

def draw_agent(data_source:dict, file_name:str):
    plt.rcParams.update({'font.size': 14})
    # data_source = {
    #     'data_mmlu': np.random.uniform(30,60, (8,3)).reshape(-1),
    #     'data_chess': np.random.uniform(30,60, (8,3)).reshape(-1),
    #     'errors_mmlu': np.random.uniform(5,10, (8,3)).reshape(-1),
    #     'errors_chess': np.random.uniform(5,10, (8,3)).reshape(-1),
    # }
    data_mmlu, data_chess, errors_mmlu, errors_chess = \
        data_source['data_mmlu'], data_source['data_chess'], data_source['errors_mmlu'], data_source['errors_chess']
    plt.rcParams['hatch.linewidth'] = 2
    # show_dot_colors = ['#3257A6', '#C30E23', '#00B050', '#FFC000']
    show_dot_colors = ['#00B050', '#80b828', '#FFC000']
    # show_dot_colors = ['#3257A6', '#7A3365','#C30E23']

    # Define the number of categories and subcategories
    n_categories = 8
    n_subcategories = 3
    n_groups = n_categories * n_subcategories

    # Generate synthetic data for three subcategories
    # np.random.seed(0)  # For reproducibility
    # data_mmlu = np.random.uniform(30, 60, (n_groups,))
    # data_chess = np.random.uniform(30, 50, (n_groups,))

    # # Generate synthetic errors to simulate error bars
    # errors_mmlu = np.random.uniform(5, 10, (n_groups,))
    # errors_chess = np.random.uniform(5, 10, (n_groups,))

    # Define colors for three subcategories
    colors_mmlu = show_dot_colors * n_categories  # Yellow shades for MMLU
    colors_chess = show_dot_colors * n_categories  # Blue shades for Chess
    labels = ['2 agents', '3 agents', '4 agents']
    hatch = ['O','+','']
    # hatch = ['/', 's','\\']

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))

    # Define the position of the bars for three subcategories
    bar_width = 0.3
    indices = np.arange(n_categories)

    # Function to create the bar plots for three subcategories
    def create_bar_plot(ax, data, errors, color, title):
        # Plotting the bars for subcategory 1
        ax.bar(indices - bar_width, data[::3], bar_width, label=labels[0], 
            yerr=errors[::3], capsize=5, color=color[::3], hatch=hatch[0], edgecolor='white')
        ax.bar(indices, data[1::3], bar_width, label=labels[1], 
            yerr=errors[1::3], capsize=5, color=color[1::3], hatch=hatch[1], edgecolor='white')
        ax.bar(indices + bar_width, data[2::3], bar_width, label=labels[2], 
            yerr=errors[2::3], capsize=5, color=color[2::3], hatch=hatch[2], edgecolor='white')

        # Setting the title
        ax.set_title(title)
        # Setting the x-axis labels with LaTeX formatting
        ax.set_xticks(indices)
        ax.set_xticklabels(
            [r'$p_0p_0p_0$', r'$p_0p_0p_1$', r'$p_0p_1p_0$', r'$p_0p_1p_1$',
             r'$p_1p_0p_0$', r'$p_1p_0p_1$', r'$p_1p_1p_0$', r'$p_1p_1p_1$'], 
            # rotation=45, ha='right'
        )
        # Setting the y-axis label
        ax.set_ylabel('Accuracy (%)')
        # Adding a legend
        ax.legend(ncol=3, loc='lower left')

    # Create the bar plots
    create_bar_plot(ax1, data_mmlu, errors_mmlu, colors_mmlu, 'MMLU')
    create_bar_plot(ax2, data_chess, errors_chess, colors_chess, 'Chess Move Validity')

    # Adjust layout
    plt.tight_layout()

    # Save the updated plot to a file
    # if save:
    #     print('save')
    #     updated_output_file = '7.png'
    #     plt.savefig(updated_output_file)
    # else:
    #     plt.show()
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  

if __name__ == '__main__':
    # draw_agent(True)
    draw_agent(
        data_source = {
            'data_mmlu': np.random.uniform(30,60, (8,3)).reshape(-1),
            'data_chess': np.random.uniform(30,60, (8,3)).reshape(-1),
            'errors_mmlu': np.random.uniform(5,10, (8,3)).reshape(-1),
            'errors_chess': np.random.uniform(5,10, (8,3)).reshape(-1),
        },
        
    )