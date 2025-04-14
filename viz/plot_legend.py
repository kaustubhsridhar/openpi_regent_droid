import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def plot_legend(colors, name):
    # Create custom legend handles
    legend_elements = [Patch(facecolor=color, label=label) for label, color in colors.items()]

    # Dummy figure and axis
    fig, ax = plt.subplots(figsize=(15, 1.5))
    ax.axis('off')  # Hide axes

    # Add legend only
    ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=False,
        ncol=3,  # 3 columns
        handlelength=1.5,
        columnspacing=1.5,
        handletextpad=0.5,
        fontsize=28
    )
    os.makedirs('plots_dont_delete', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'plots_dont_delete/{name}.png', bbox_inches='tight')

if __name__ == '__main__':
    colors_1 = {
        'π₀-FAST-DROID': 'tab:orange',
        'Retrieve and play': 'tab:blue',
        'Regentic-π₀-FAST-DROID': 'tab:green',
        'π₀-FAST-DROID finetuned on 20 demos': 'tab:red',
        # 'π₀-FAST-DROID finetuned on num demos': 'tab:red',
        'Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos': 'tab:purple',
        # 'Regentic-π₀-FAST-DROID Regentic-tuned on num demos': 'tab:purple',
    }

    colors_2 = {
        'π₀-FAST-DROID': 'tab:orange',
        'Retrieve and play': 'tab:blue',
        'Regentic-π₀-FAST-DROID': 'tab:green',
        # 'π₀-FAST-DROID finetuned on 20 demos': 'tab:red',
        'π₀-FAST-DROID finetuned on num demos': 'tab:red',
        # 'Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos': 'tab:purple',
        'Regentic-π₀-FAST-DROID Regentic-tuned on num demos': 'tab:purple',
    }

    plot_legend(colors_1, name='colors_1')
    plot_legend(colors_2, name='colors_2')
