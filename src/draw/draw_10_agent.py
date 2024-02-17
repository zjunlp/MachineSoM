
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from .utils import ensure_directories_exist

def get_middle_x(start, border, width, size):
    assert border>=start and size*width <= border-start, f"{border}>={start} and {size*width} <= {border-start}, size={size}, width={width}"
    results = []

    center = (border+start) / 2
    if size % 2 == 0:
        for i in range(size//2):
            results.append(center - width * (i+1))
        results = results[::-1]
        for i in range(size//2):
            results.append(center + width * i)
    else:
        
        for i in range((size-1)//2):
            results.append(center - width/2 - width*(i+1))
        results = results[::-1]
        results.append(center-width/2)
        for i in range((size-1)//2):
            results.append(center + width/2 + width*i)
    assert len(results) == size
    return results


def single_draw(data, error_data, transpose, ax, x_label, legend_label, ylabel:str, mask):
    legend = {label:None for label in legend_label}
    if transpose:
        # print(data.shape)
        data = data.T
        mask = mask.T
        # print(data.shape)
        error_data = error_data.T

    num_bars = data.shape[1]     
    num_groups = data.shape[0]
    assert data.shape == mask.shape

    # Define group names based on the data shape
    group_names = legend_label

    # Define colors and hatches
    # colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
    colors = ['#FFF5DE', '#FAE6B9', '#CFD1B6', '#CDCC94', '#FFE8A5', '#B7D1AC', '#FFBABA', '#8FCBA5', '#B6D1CC']
    if num_bars == 5:
        colors = [colors[i] for i in [0,2,4,7,8]]
    
    plt.rcParams['hatch.linewidth'] = 0.3
    hatches = ['oo', '////','*','+o', '..', '--\\\\','++', 'xx','/oo']
    # hatches = ['']

    # Set figure size
    # plt.figure(figsize=(15, 3))

    # Calculate the width of each bar
    group_width = Decimal('0.8')
    bar_width = group_width / num_bars

    # Plot bar chart
    # print(num_bars, num_groups)
    # exit()
    for group_id in range(num_groups):
        y = data[group_id]
        index = np.where(mask[group_id,:]==1)[0].tolist() 
        x = get_middle_x(group_id, border=group_id+group_width, width=bar_width, size=len(index))

        for draw_id, bar_idx in enumerate(index):
            line, = ax.bar(
                x[draw_id],
                y[bar_idx],
                width=bar_width,
                color=colors[bar_idx],
                edgecolor='black', 
                yerr=error_data[group_id,bar_idx],
                error_kw={'capthick':1, 'elinewidth':1},
                capsize=3,
                hatch=hatches[bar_idx], 
                label=group_names[bar_idx]
            )
            legend[group_names[bar_idx]] = line

    # Add legend
    # ax.legend(ncol=5, loc='lower center')

    # Set axis labels and title
    # ax.set_xlabel('Society' if len(x_label)==5 else 'Agent')
    # ax.set_ylim([50, ax.get_ylim()[1]])
    # ax.set_ylim([(min(data.reshape(-1)-error_data.reshape(-1)))*1.2,ax.get_ylim()[1]])
    ax.set_ylabel(ylabel) #'Accuracy (%)'
    # plt.title('Grouped Bar Chart')

    # Adjust x-axis tick label position
    ax.set_xticks(np.arange(num_groups) + bar_width * (Decimal(str(num_bars)) / 2), x_label)

    # Show plot
    # plt.show()
    assert list(legend.keys()) == legend_label
    return list(legend.values())

def draw_10_agent(database: dict, transpose:bool, file_name:str, ylabel:str='Accuracy (%)'):
    plt.rcParams.update({'font.size': 14})
    np.random.seed(0)
    data_list = [np.random.rand(5, 10) * 100 for _ in range(8)]
    error_data_list = [np.random.rand(5, 10) * 10 for _ in range(8)]
    data_list, error_data_list, legend_label, x_label, title = database['data'], database['error'], database['legend_label'], database['x_label'], database['title']
    mask = database['mask']

    # Create figure and axes
    fig, axes = plt.subplots(8, 1, figsize=(15, 24))

    # Plot each subplot
    for i, ax in enumerate(axes):
        # plot_subgrouped_bars(data_list[i], error_data_list[i], ax, group_names)
        handles = single_draw(data_list[i], error_data_list[i], transpose=transpose, ax=ax, x_label=x_label, legend_label=legend_label, ylabel=ylabel, mask=mask)
        # ax.set_title(title[i])
        ax.set_xlabel(title[i])

    # Add a common legend
    fig.legend(handles, legend_label, loc='lower center', ncol=5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 1])

    # Show plot
    # plt.show()
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)  
