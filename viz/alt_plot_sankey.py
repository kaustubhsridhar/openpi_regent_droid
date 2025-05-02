import plotly.graph_objects as go

# Define the nodes (stages/categories)
# Assign labels that might represent the stages in your sketch
labels = [
    "Source A", "Source B", # Stage 1 (Indices 0, 1)
    "Process C", "Process D", # Stage 2 (Indices 2, 3)
    "Output E", "Output F", "Output G" # Stage 3 (Indices 4, 5, 6)
]

# Define the links (flows between nodes)
# 'source' and 'target' use the indices from the 'labels' list
# 'value' represents the thickness of the flow
links = {
    'source': [
        0, 0,  # Flows from Source A
        1, 1,  # Flows from Source B
        2,      # Flow from Process C
        3, 3, 3 # Flows from Process D
    ],
    'target': [
        2, 3,  # Flows to Process C, Process D
        2, 3,  # Flows to Process C, Process D
        4,      # Flow to Output E
        4, 5, 6 # Flows to Output E, Output F, Output G
    ],
    'value': [
        10, 5,  # Values from Source A
        5, 15, # Values from Source B
        15,     # Value from Process C (must match sum of inputs: 10 + 5)
        5, 10, 5 # Values from Process D (must match sum of inputs: 5 + 15)
    ]
    # Optional: You can add color to the links
    # 'color': ['rgba(0,0,255,0.5)', 'rgba(0,0,255,0.5)', 'rgba(0,255,0,0.5)', 'rgba(0,255,0,0.5)', 'rgba(255,0,0,0.5)', 'rgba(255,0,255,0.5)', 'rgba(255,0,255,0.5)', 'rgba(255,0,255,0.5)']
}

# Optional: Define node positions to resemble the sketch more closely
# x coordinates control horizontal position (0 to 1)
# y coordinates control vertical position (0 to 1)
node_positions = {
    'x': [0.05, 0.05,  # Stage 1 nodes
          0.45, 0.45,  # Stage 2 nodes
          0.9, 0.9, 0.9], # Stage 3 nodes
    'y': [0.2, 0.8,   # Stage 1 nodes (adjust vertical spacing)
          0.1, 0.7,   # Stage 2 nodes (adjust vertical spacing)
          0.1, 0.5, 0.9]  # Stage 3 nodes (adjust vertical spacing)
}


# Create the Sankey diagram figure
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,              # Padding between nodes
      thickness = 20,        # Thickness of the nodes
      line = dict(color = "black", width = 0.5), # Node borders
      label = labels,        # Node labels
      color = "blue",        # Default node color (can be a list of colors)
      x = node_positions['x'], # Assign x positions
      y = node_positions['y']  # Assign y positions
    ),
    link = dict(
      source = links['source'], # Indices correspond to labels, eg A1, A2, A1, B1, ...
      target = links['target'],
      value = links['value']
      # color = links['color'] # Uncomment if you defined link colors
  ))])

# Update layout for title and size
fig.update_layout(
    title_text="Sankey Diagram Example",
    font_size=12,
    height=500, # Adjust height as needed
    width=700   # Adjust width as needed
    )

# Show the figure
fig.write_image("sankey.png")