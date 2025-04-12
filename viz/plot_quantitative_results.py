import os 
import numpy as np
import matplotlib.pyplot as plt

from quantitative_results_info import quantitative_results
from ablation_results_info import ablation_results

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

for task in quantitative_results:
    subtasks = list(quantitative_results[task].keys())
    methods = list(quantitative_results[task][subtasks[0]].keys())

    # create a grouped bar chart where each subtask is a group and each method is a bar
    # configuration
    bar_width = 0.1
    x = np.arange(len(subtasks))  # Base x locations for each group
    
    fig, ax = plt.subplots(figsize=(5*len(subtasks), 5))

    for i, method in enumerate(methods):
        offsets = x + i * bar_width - bar_width * (len(methods) - 1) / 2
        
        values_are_tuple = False
        if "(extra: " in subtasks:
            values_are_tuple = True
        
        if values_are_tuple:
            actual_vals = [quantitative_results[task][subtask][method][0] for subtask in subtasks]
            display_vals = [max(2, val) for val in actual_vals] # just to see a tiny sliver of a line for 0%
            actual_vals_in_parentheses = [quantitative_results[task][subtask][method][1] for subtask in subtasks]
        else:
            actual_vals = [quantitative_results[task][subtask][method] for subtask in subtasks]
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
                        ha='center', va='bottom')

    # Styling
    ax.set_ylabel('Success Percentage')
    ax.set_title(f'{task}', fontsize=BIGGER_SIZE)
    ax.set_ylim(-5, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    # ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs('plots_dont_delete', exist_ok=True)
    plt.savefig(f'plots_dont_delete/{task}.png', bbox_inches='tight')
