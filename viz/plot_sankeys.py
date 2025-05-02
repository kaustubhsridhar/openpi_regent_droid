from quantitative_results_info import quantitative_results as quantitative_results_og
import os
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def plot_sankey(task_name, quantitative_results):
    num_stages = len(quantitative_results[task_name])
    stages = list(quantitative_results[task_name].keys())
    num_methods = len(quantitative_results[task_name][stages[0]])
    methods = list(quantitative_results[task_name][stages[0]].keys())
    
    tn = num_stages * 2 + 1
    if num_stages == 2:
        source_nodes = []
        target_nodes = []
        first_stage_success = np.array(list(quantitative_results[task_name][stages[0]].values())) / 100.0
        first_stage_fails = 1.0 - first_stage_success
        second_stage_success = np.array(list(quantitative_results[task_name][stages[1]].values())) / 100.0
        second_stage_fails = first_stage_success - second_stage_success
        values = []
        links_to_methods = []
        for i in range(num_methods):
            source_nodes.extend([0, 0])
            target_nodes.extend([tn+i, tn+i])
            source_nodes.extend([tn+i, tn+i])
            target_nodes.extend([1, 2])
            values.extend([first_stage_success[i], first_stage_fails[i]])
            values.extend([first_stage_success[i], first_stage_fails[i]])
            links_to_methods.extend([methods[i], methods[i]])
            links_to_methods.extend([methods[i], methods[i]])
        for i in range(num_methods):
            source_nodes.extend([1, 1])
            target_nodes.extend([tn+num_methods+i, tn+num_methods+i])
            source_nodes.extend([tn+num_methods+i, tn+num_methods+i])
            target_nodes.extend([3, 4])
            values.extend([second_stage_success[i], second_stage_fails[i]])
            values.extend([second_stage_success[i], second_stage_fails[i]])
            links_to_methods.extend([methods[i], methods[i]])
            links_to_methods.extend([methods[i], methods[i]])
    elif num_stages == 3:
        source_nodes = []
        target_nodes = []
        first_stage_success = np.array(list(quantitative_results[task_name][stages[0]].values())) / 100.0
        first_stage_fails = 1.0 - first_stage_success
        second_stage_success = np.array(list(quantitative_results[task_name][stages[1]].values())) / 100.0
        second_stage_fails = first_stage_success - second_stage_success
        third_stage_success = np.array(list(quantitative_results[task_name][stages[2]].values())) / 100.0
        third_stage_fails = second_stage_success - third_stage_success
        values = []
        links_to_methods = []
        for i in range(num_methods):
            source_nodes.extend([0, 0])
            target_nodes.extend([tn+i, tn+i])
            source_nodes.extend([tn+i, tn+i])
            target_nodes.extend([1, 2])
            values.extend([first_stage_success[i], first_stage_fails[i]])
            values.extend([first_stage_success[i], first_stage_fails[i]])
            links_to_methods.extend([methods[i], methods[i]])
            links_to_methods.extend([methods[i], methods[i]])
        for i in range(num_methods):
            source_nodes.extend([1, 1])
            target_nodes.extend([tn+num_methods+i, tn+num_methods+i])
            source_nodes.extend([tn+num_methods+i, tn+num_methods+i])
            target_nodes.extend([3, 4])
            values.extend([second_stage_success[i], second_stage_fails[i]])
            values.extend([second_stage_success[i], second_stage_fails[i]])
            links_to_methods.extend([methods[i], methods[i]])
            links_to_methods.extend([methods[i], methods[i]])
        for i in range(num_methods):
            source_nodes.extend([3, 3])
            target_nodes.extend([tn+num_methods*2+i, tn+num_methods*2+i])
            source_nodes.extend([tn+num_methods*2+i, tn+num_methods*2+i])
            target_nodes.extend([5, 6])
            values.extend([third_stage_success[i], third_stage_fails[i]])
            values.extend([third_stage_success[i], third_stage_fails[i]])
            links_to_methods.extend([methods[i], methods[i]])
            links_to_methods.extend([methods[i], methods[i]])
    else:
        raise NotImplementedError(f"Number of stages {num_stages} not supported")
    assert len(values) == len(source_nodes) == len(target_nodes), f"{len(values)} {len(source_nodes)} {len(target_nodes)}"
    
    palette_name = 'tab20'
    # cmap = plt.get_cmap(palette_name) # Get the colormap object
    cmap = sns.color_palette(palette_name, as_cmap=True)
    rgb_colors = cmap.colors
    palette_colors = [mcolors.to_hex(c) for c in rgb_colors]
    method_to_colors = {
            "π₀-FAST-DROID": palette_colors[0],
            "Retrieve and play": palette_colors[6],
            "Regentic-π₀-FAST-DROID": palette_colors[2],
            "π₀-FAST-DROID finetuned on 20 demos": palette_colors[1],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": palette_colors[3],
            "Diffusion Policy (from scratch)": palette_colors[8],
        }
    methods_to_colors_list = list(method_to_colors.values())
    # node_x = [0.0,  # Node 0 (Start) at left edge
    #     0.3,  # Node 1 (S1 Success) column 2
    #     0.3,  # Node 2 (S1 Fails) column 2
    #     0.6,  # Node 3 (S2 Success) column 3
    #     0.6,  # Node 4 (S2 Fails) column 3
    #     0.9,  # Node 5 (S3 Success) column 4
    #     0.9]  # Node 6 (S3 Fails) column 4

    # node_y = [0.5,  # Node 0 (Start) vertically centered
    #     0.3,  # Node 1 (S1 Success) - Higher up
    #     0.7,  # Node 2 (S1 Fails) - Lower down
    #     0.2,  # Node 3 (S2 Success) - Even higher
    #     0.4,  # Node 4 (S2 Fails) - Below S2 Success
    #     0.1,  # Node 5 (S3 Success) - Highest
    #     0.3]  # Node 6 (S3 Fails) - Below S3 Success

    # if num_stages == 2:
    #     node_x = node_x[:-2]
    #     node_y = node_y[:-2]
        
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0),
            label = [""] + sum([[s.replace("prev + ", ""), "Fails"] for s in stages], []),
            color = ["black"] + sum([["green", "red"] for s in stages], []) + methods_to_colors_list * num_stages,
            align='left',
            # x = node_x,
            # y = node_y,
        ),
        link = dict(
            source = source_nodes,
            target = target_nodes,
            value = values,
            color = [method_to_colors[m] for m in links_to_methods],
    ))])

    fig.update_layout(title_text=f"Task: {task_name}", font_size=10)
    # plot legend
    for method_name, color_hex in method_to_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],        # We don't want to plot any actual data points
            y=[None],
            mode='markers',  # Use markers so we get a colored symbol in the legend
            marker=dict(
                size=10,       # Size of the colored symbol in the legend
                color=color_hex  # Set the color for this legend entry
            ),
            name=method_name, # This text label will appear next to the color in the legend
            showlegend=True  # IMPORTANT: This tells Plotly to include this trace in the legend
        ))
    os.makedirs("sankeys_dont_delete", exist_ok=True)
    fig.write_image(f"sankeys_dont_delete/{task_name}.png")
    print(f"Saved sankeys_dont_delete/{task_name}.png")

if __name__ == "__main__":

    quantitative_results = {}
    for task_name, task_results in quantitative_results_og.items():
        quantitative_results[task_name] = {}
        for stage_name, stage_results in task_results.items():
            quantitative_results[task_name][stage_name] = {}
            for method_name, method_result in stage_results.items():
                quantitative_results[task_name][stage_name][method_name] = method_result[0] if isinstance(method_result, list) else method_result

    tasks = list(quantitative_results.keys())
    for task in tasks:
        plot_sankey(task, quantitative_results)
