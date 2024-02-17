import matplotlib.pyplot as plt
import numpy as np
from .utils import ensure_directories_exist

def draw(ax, data, curve_labels:list[str], title:str, y_label:str, 
         group_labels: list[str], accumulate_id:int, size: int, draw_y:bool, draw_x: bool, mask):
    handles = {i:None for i in curve_labels}
    ax.grid(True, axis='both', linestyle='--')
    assert len(curve_labels) == data.shape[1]
    assert len(group_labels) == data.shape[0]
    # all_data = data
    # data = all_data[accumulate_id:accumulate_id+size]
    x = np.arange(size * data.shape[2])    # [27]
    # colors = ['#FFF5DE', '#FAE6B9', '#CFD1B6', '#CDCC94', '#FFE8A5', '#B7D1AC', '#FFBABA', '#8FCBA5', '#B6D1CC']
    # colors = ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4', '#009688', '#4CAF50']
    # colors = ['#db5f57', '#dbb757', '#a7db57', '#57db5f', '#57dbb7',
    #           '#57a7db', '#5f57db', '#b757db', '#db57a7']
    # f4f1de 8cd0ec
    markers = ['o', 's', '^', 'D', '*', '+', 'x', '1', '2', '3']
    if data.shape[1] == 5:
        # society
        markers = [f'$S{i}$' for i in range(5)]
        colors = ['#eab69f', '#8f5d5d', '#5f797b', '#babf95','#f2cc8f']
        colors = ['#eab69f', '#8f5d5d', '#5f797b','#e6938d','#5d76ad']
        colors = ['#F07C12','#FFC200','#21B534','#0095AC','#1F64AD']
        print('check')
    else:
        # agent
        # matplotlib.rcParams['text.usetex'] = True
        markers = [f'${i+2}$' for i in range(data.shape[1])]
        # ' '#48cae4'
        colors = ['#e6938d', '#eab69f', '#e07a5f','#8f5d5d','#3d405b','#5f797b','#81b29a','#babf95','#f2cc8f']
        colors = ['#E03524','#F07c12','#ffc200','#90bc1a', '#21b534', '#0095ac', '#1f64ad', '#4040a0','#903498']
    markers = ['o', 's', '^', 'D', '*', '+', 'x', '>', '<', 'p']
    markers = ['o', '^', 's', 'D', 'p', '*', 'P', 'H', 'X', 'h']
    linestyles = ["solid","dashed", "solid","dashed","solid","dashed","solid","dashed","solid","dashed"]
    
    # markers = ['$1$','$2$','$3$','$4$', '$5$', '$6$', '$7$', '$8$', '$9$', '$10$']
    assert len(colors) >= data.shape[1], f"{len(colors)}, {data.shape}"
    # 一个一个子图画
    for group_id in range(accumulate_id, accumulate_id+size):
        x_start, x_end = data.shape[2] * (group_id-accumulate_id), data.shape[2] * (group_id+1-accumulate_id)
        for curve_id in range(data.shape[1]):
            if mask[group_id, curve_id] == 0:
                continue
            y = data[group_id, curve_id]

            line, = ax.plot(
                x[x_start: x_end], y, color=colors[curve_id], marker=markers[curve_id], label=curve_labels[curve_id],  
                alpha=0.8, linewidth=3, markersize=13, linestyle=linestyles[curve_id],
                markeredgewidth=1.5, markerfacecolor='none',
            )
            # if sum(mask[group_id]) == 0:
            #     handles.append(line)
            # handles.append(line)
            handles[curve_labels[curve_id]] = line
            # ax.scatter(
            #     x[x_start: x_end], y, color=colors[curve_id], marker=markers[curve_id], label=curve_labels[curve_id]
            # )
        if x_end < x.shape[0]:
            ax.axvline(x=x[x_end-1] + 0.5, color='grey', linestyle='-', alpha=0.7)
    if draw_x:
        ax.set_xlabel(title)
    if draw_y:
        ax.set_ylabel(y_label)
    # 0 1 2 3    4 5 6 7
    # 0 1 2 3    4 5 6 7
    xticks_label = []
    xticks_x = []
    idx = 0
    group_name_x = (data.shape[2]-1)/2
    for i in range(len(x)):
        xticks_x.append(i)
        if i % data.shape[-1] == 0:
            idx = 0
        if idx <= group_name_x and idx+1 > group_name_x:
            if data.shape[2] % 2 == 0:
                xticks_x.append(group_name_x + i // data.shape[2])
                xticks_label.append(f'\n{group_labels[accumulate_id + i // data.shape[2]]}')
            else:
                xticks_label.append(f'{idx+1}\n{group_labels[accumulate_id +i // data.shape[2]]}')
        else:
            xticks_label.append(str(idx+1))
        idx += 1
    ax.set_xticks(xticks_x, xticks_label)
    print(handles.keys())
    return list(handles.values())
    # ax.set_xticks([i+(data.shape[2]-1)/2 for i in range(0, x.shape[0], data.shape[2])], group_labels)
 
def draw_10_agent_consistent_line(database:dict, ylabel: str, file_name:str, plotGroup:list):
    plt.rcParams.update({'font.size': 22})
    data, curve_labels, group_labels, titles = database['data'], database['legend_label'], database['x_label'], database['title']
    mask = database['mask']

    num_subplot = data.shape[0]
    # plotGroup = [9]
    # plotGroup = [2, 5, 2]
    assert sum(plotGroup) == data.shape[1], f"{plotGroup} == {data.shape}"
    fig, axs = plt.subplots(num_subplot, len(plotGroup), figsize=(17, 48), gridspec_kw={'width_ratios':plotGroup})  # 17,48     24,36

    for row_id in range(num_subplot):
        if len(plotGroup) == 1:
            col_id = 0
            handles = draw(
                axs[row_id], 
                data[row_id], 
                curve_labels, 
                titles[row_id], 
                ylabel, 
                group_labels, 
                accumulate_id=sum(plotGroup[0:col_id]), 
                size=plotGroup[col_id],
                draw_x= True,
                draw_y= True,
                mask=mask
            )
        else:
            for col_id in range(len(plotGroup)):
                handles = draw(
                    axs[row_id, col_id], 
                    data[row_id], 
                    curve_labels, 
                    titles[row_id], 
                    ylabel, 
                    group_labels, 
                    accumulate_id=sum(plotGroup[0:col_id]), 
                    size=plotGroup[col_id],
                    draw_x= col_id == 1,
                    draw_y= col_id == 0,
                    mask=mask
                )
    fig.legend(handles=handles[0:len(curve_labels)],labels=curve_labels, loc='lower center', ncol=5)
    # fig.legend(curve_labels, loc='lower center', ncol=5)
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    ensure_directories_exist(file_name)
    if 'pdf' in file_name:
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(file_name)   
