import os 
import numpy as np
import matplotlib.pyplot as plt

def plot_for_a_task(task, dict_of_dicts):
    # hypers
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 28

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = {
        'π₀-FAST-DROID': 'tab:orange',
        'Retrieve and play': 'tab:blue',
        'Regentic-π₀-FAST-DROID': 'tab:green',
        'π₀-FAST-DROID finetuned on 20 demos': 'tab:red', 'π₀-FAST-DROID finetuned on num demos': 'tab:red',
        'Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos': 'tab:purple', 'Regentic-π₀-FAST-DROID Regentic-tuned on num demos': 'tab:purple',
    }

    subtasks = list(dict_of_dicts.keys())
    methods = list(dict_of_dicts[subtasks[0]].keys())

    # create a grouped bar chart where each subtask is a group and each method is a bar
    # configuration
    bar_width = 0.135
    x = np.arange(len(subtasks))  # Base x locations for each group
    
    fig, ax = plt.subplots(figsize=(5*len(subtasks), 5))

    for i, method in enumerate(methods):
        offsets = x + i * bar_width - bar_width * (len(methods) - 1) / 2
        
        values_are_tuple = False
        if not 'Ablations' in task:
            if any("(extra: " in subtask for subtask in subtasks):
                values_are_tuple = True
        
        if values_are_tuple:
            actual_vals = [dict_of_dicts[subtask][method][0] for subtask in subtasks]
            display_vals = [max(2, val) for val in actual_vals] # just to see a tiny sliver of a line for 0%
            actual_vals_in_parentheses = [dict_of_dicts[subtask][method][1] for subtask in subtasks]
        else:
            actual_vals = [dict_of_dicts[subtask][method] for subtask in subtasks]
            display_vals = [max(2, val) for val in actual_vals] # just to see a tiny sliver of a line for 0%
        
        bars = ax.bar(offsets, display_vals, width=bar_width, label=method, color=colors[method])

        # Annotate values just inside the bar
        for j, bar in enumerate(bars):
            height = bar.get_height()
            label = f'{actual_vals[j]}'
            ax.annotate(f'{label}\n({actual_vals_in_parentheses[j]})' if (values_are_tuple and actual_vals_in_parentheses[j] is not None) else label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', 
                        va='bottom',
                        # fontsize=SMALL_SIZE,
                    )

    # Styling
    ax.set_ylabel('Success Percentage')
    ax.set_title(f'{task}', fontsize=BIGGER_SIZE)
    ax.set_ylim(-5, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    # ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    if 'Ablations' in task:
        ax.set_xlabel('Number of demonstrations to retrieve from or finetune on')

    plt.tight_layout()
    os.makedirs('plots_dont_delete', exist_ok=True)
    plt.savefig(f'plots_dont_delete/{task.replace(" ", "_").replace("(", "").replace(")", "")}.png', bbox_inches='tight')
    print(f'Done with {task=}')

if __name__ == '__main__':
    from quantitative_results_info import quantitative_results
    from ablation_results_info import ablation_results

    for task_count, task in enumerate(quantitative_results):
        # if os.path.exists(f'plots_dont_delete/{task}.png'):
        #     print(f'We already have this and skipping {task=}')
        #     continue

        plot_for_a_task(task, quantitative_results[task])

    # ablations results
    plot_for_a_task('Ablations (idli plate task)', ablation_results['move the idli plate to the right'])

