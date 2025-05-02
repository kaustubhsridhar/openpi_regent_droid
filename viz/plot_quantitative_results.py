import os 
import numpy as np
import matplotlib.pyplot as plt
import copy

# hypers
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

shortener = {
    'π₀-FAST-DROID': 'π₀',
    'Retrieve and play': 'R&P',
    'RICL-π₀-FAST-DROID': 'RICL',
    'π₀-FAST-DROID-finetuned': 'π₀-\nfinetune',
    'RICL-π₀-FAST-DROID-finetuned': 'RICL-\nfinetune',
    'Diffusion Policy': 'DP',
}

def plot_ablations(task, dict_of_dicts):
    xvals = list(dict_of_dicts.keys())
    methods = list(dict_of_dicts[xvals[0]].keys())
    yvals = {}
    for method in methods:
        yvals[method] = []
        for xval in xvals:
            yvals[method].append(dict_of_dicts[xval][method])

    # plot line plot
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(xvals, yvals[method], label=shortener[method].replace('\n', ''), linewidth=3, marker='o', markersize=10)
    plt.xlabel('Number of demonstrations')
    plt.ylabel('Success rate (%)')
    plt.ylim(-5, 115)
    plt.xlim(3, max(xvals) + 1)
    plt.title(task)
    plt.xticks(xvals)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plot legend in reverse order
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', ncol=1, fontsize=SMALL_SIZE)
    plt.tight_layout()
    plt.savefig(f'plots_dont_delete/{task.replace(" ", "_").replace("(", "").replace(")", "")}.png', bbox_inches='tight')
    print(f'Done with {task=}; saved to plots_dont_delete/{task.replace(" ", "_").replace("(", "").replace(")", "")}.png')

def plot_for_a_task(task, dict_of_dicts):
    subtasks = list(dict_of_dicts.keys())
    methods = list(dict_of_dicts[subtasks[0]].keys())

    # create a grouped bar chart where each subtask is a group and each method is a bar
    # configuration
    bar_width = 1.0
    SCALE = 1.5
    fig, ax = plt.subplots(figsize=((SCALE+0.25)*len(methods), 7))

    xvals = [i*SCALE for i in range(len(methods))]
    bottoms = np.array([0] * len(methods))
    colors = ['royalblue', 'lightcoral', 'mediumturquoise'] # lightskyblue
    to_plot = []
    subtasks_reversed = list(subtasks[::-1])
    subtasks_reversed_without_extras = [subtask.split(f' (extra:')[0].replace('\n', ' ') for subtask in subtasks_reversed]
    
    ax.bar(xvals, [100] * len(methods), width=bar_width, color='lightgray')
    for j, subtask in enumerate(subtasks_reversed):
        actual_vals = []
        display_vals = []
        parenthesis_vals = []
        for i, method in enumerate(methods):
            istup = isinstance(dict_of_dicts[subtask][method], list)
            actual_vals.append(dict_of_dicts[subtask][method][0] if istup else dict_of_dicts[subtask][method])
            display_vals.append(max(2, actual_vals[-1])) # just to see a tiny sliver of a line for 0%
            parenthesis_vals.append(dict_of_dicts[subtask][method][1] if istup else None)

        if j == 0:
            first_display_vals = copy.deepcopy(display_vals)
            display_vals = np.array(display_vals)
        if j == 1:
            second_display_vals = copy.deepcopy(display_vals)
            display_vals = np.array(display_vals) - np.array(first_display_vals)
        if j == 2:
            third_display_vals = copy.deepcopy(display_vals)
            display_vals = np.array(display_vals) - np.array(second_display_vals)
            
        if j == 0 and 'Ablations' not in task:
            bars = ax.bar(xvals, display_vals, width=bar_width, color=colors[j], bottom=bottoms, label='Task completed', edgecolor='black', linewidth=3)
        else:
            bars = ax.bar(xvals, display_vals, width=bar_width, color=colors[j], bottom=bottoms, label=subtasks_reversed_without_extras[j])

        # annotate the bars with the actual values
        for i, bar in enumerate(bars):
            ax.annotate(
                # f'{actual_vals[i]} ({parenthesis_vals[i]})' if istup and parenthesis_vals[i] is not None else f'{actual_vals[i]}', 
                f'{actual_vals[i]}',
                xy=(xvals[i], bottoms[i] + bar.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=SMALL_SIZE)

        bottoms += display_vals 

    # plot legend below and outside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.125 if len(methods) > 2 else -0.075), ncol=3 if 'Ablations' in task else 1, fontsize=SMALL_SIZE)
    # Styling
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'{task}', fontsize=BIGGER_SIZE)
    ax.set_ylim(-5, 115)
    ax.set_xticks(xvals)
    methods_shortened = [shortener[method] for method in methods]
    ax.set_xticklabels(methods_shortened)
    # ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # plt.tight_layout()
    os.makedirs('plots_dont_delete', exist_ok=True)
    task = task.replace('\n', '_')
    saveloc = f'plots_dont_delete/{task.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(saveloc, bbox_inches='tight')
    print(f'Done with {task=}; saved to {saveloc}')

if __name__ == '__main__':
    from quantitative_results_info import quantitative_results
    from ablation_results_info import ablation_results

    for task_count, task in enumerate(quantitative_results):
        # if os.path.exists(f'plots_dont_delete/{task}.png'):
        #     print(f'We already have this and skipping {task=}')
        #     continue

        if 'no loss-of-capabilities' in task:
            continue

        plot_for_a_task(task, quantitative_results[task])

    # ablations results
    plot_ablations('Ablations (idli plate task)', ablation_results['move the idli plate to the right'])

