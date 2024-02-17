import numpy as np
import matplotlib.pyplot as plt
from .utils import ensure_directories_exist

def draw_radar(database:dict, file_name:str):
    data, titles, attributes, legend_labels = database['data'], database['titles'], database['dot_names'], database['legend_labels']

    plt.rcParams.update({'font.size': 14})
    num_vars = len(attributes)

    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, axes = plt.subplots(1, 4, subplot_kw=dict(polar=True), figsize=(35, 7))

    # Define colors for each group in each plot
    # colors = [['blue', 'green', 'red'], 
    #           ['purple', 'orange', 'cyan'], 
    #           ['brown', 'pink', 'grey'], 
    #           ['navy', 'yellow', 'lime']]
    color_set = ['#a5d6a7', '#ce93d8', '#90caf9', '#5C75AE', '#EA9490', '#87DAAD', '#FAE334', '#E5844C', '#7ec8e3', '#ffe680', '#fa7f6f', '#8ecfc9',]#'#a2d5f2', '#fff1c1', '#a8d5a7', '#f4c2c2']#'#5F7AE3', '#FFAA40', '#55eeF8','#21DEC7']

    # Legend labels for each group in each plot

    legend_handles = []

    # Plot each radar chart
    color_idx = 0
    for ax, title, case_data in zip(axes, titles, data):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable and add LaTeX formatted labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(attributes, fontsize=16, ha='center')

        # Draw ylabels
        ax.set_rlabel_position(0)
        # plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
        # plt.ylim(0, 1)

        # Plot data with different colors, add to legend, and add markers
        for d in case_data:
            label = legend_labels[color_idx]
            c = color_set[color_idx]
            color_idx += 1
            d = np.append(d, d[0])  # Complete the loop
            line, = ax.plot(angles, d, 'o-', linewidth=2, color=c, label=label)
            ax.fill(angles, d, alpha=0.5, color=c)  # Adjust alpha for opacity
            # ax.scatter(angles, d, s=40, color=c, alpha=0.8, edgecolors='k')  # Add markers
            legend_handles.append(line)

        # Add a title
        ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.3))
        # ax.set_title(title, y=1.1)
        ax.xaxis.label.set_size(16)
        ax.set_xlabel(title)
    
    # fig.subplots_adjust(bottom=0.25)

    # Add a shared legend at the bottom
    # fig.legend(handles=legend_handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0))

    # plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.tight_layout(pad=2)
    # plt.show()
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  


if __name__ == '__main__':
    # Generate random data
    np.random.seed(0)  # For reproducibility

    database = {
        'data': [
            np.random.rand(3, 8), 
            np.random.rand(3, 8), 
            np.random.rand(2, 8), 
            np.array(
                [
                [52.954548, 53.409092, 50.681816, 50.      , 55.000004, 52.954544, 54.772728, 48.863636],
                [49.772728, 46.818176, 48.63636 , 44.318184, 48.86364 , 47.727276,45.454544, 41.363632],
                [30.833336, 24.166668, 25.000002, 25.000004, 28.333336, 29.166668,20.833334, 13.333335],
                [40.      , 44.375   , 42.5     , 45.      , 49.375   , 43.75    ,48.125   , 41.25    ]
        ])
        
    ], # [3,8], [3,8], [2,8], [4,]
        'titles': ["Chart 1", "Chart 2", "Chart 3", "Chart 4"],
        'legend_labels':[
            'Group 1 - Chart 1', 'Group 2 - Chart 1', 'Group 3 - Chart 1',
            'Group 1 - Chart 2', 'Group 2 - Chart 2', 'Group 3 - Chart 2',
            'Group 1 - Chart 3', 'Group 2 - Chart 3', 'Group 3 - Chart 3',
            'Group 1 - Chart 4', 'Group 2 - Chart 4', 'Group 3 - Chart 4'
        ],
        'dot_names': [
            "$p_0p_0p_0$", "$p_0p_0p_1$", "$p_0p_1p_1$", "$p_0p_1p_0$", 
            "$p_1p_0p_0$", "$p_1p_0p_1$", "$p_1p_1p_0$", "$p_1p_1p_1$"
        ]
    }

    # Call the plotting function with markers
    draw_radar(database, '17.png')
