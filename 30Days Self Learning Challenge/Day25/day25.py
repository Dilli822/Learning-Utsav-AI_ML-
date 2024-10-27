import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# Define a flow graph with the specified number of nodes
G = nx.DiGraph()

# Add nodes (layers of the neural network)
input_nodes = [f'Input{i+1}' for i in range(2)]
hidden1_nodes = [f'Hidden1_{i+1}' for i in range(5)]
hidden2_nodes = [f'Hidden2_{i+1}' for i in range(4)]
hidden3_nodes = [f'Hidden3_{i+1}' for i in range(3)]
output_nodes = [f'Output{i+1}' for i in range(2)]




# Position nodes for visualization
positions = {
    'Input1': (0, 4),
    'Input2': (0, 2),
}

for i in range(5):
    positions[hidden1_nodes[i]] = (1, 5 - i)

for i in range(4):
    positions[hidden2_nodes[i]] = (2, 4 - i)

for i in range(3):
    positions[hidden3_nodes[i]] = (3, 3 - i)

for i in range(2):
    positions[output_nodes[i]] = (4, 2 - i)

# Add edges (connections between layers)
for input_node in input_nodes:
    for hidden1_node in hidden1_nodes:
        G.add_edge(input_node, hidden1_node)

for hidden1_node in hidden1_nodes:
    for hidden2_node in hidden2_nodes:
        G.add_edge(hidden1_node, hidden2_node)

for hidden2_node in hidden2_nodes:
    for hidden3_node in hidden3_nodes:
        G.add_edge(hidden2_node, hidden3_node)

for hidden3_node in hidden3_nodes:
    for output_node in output_nodes:
        G.add_edge(hidden3_node, output_node)

# Prepare the figure for animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # Hide axes

# Function to draw the graph with arrows
def draw_graph(forward_edges=None, backward_edges=None):
    ax.clear()
      # Draw the nodes with white interior, black border, and white labels
    nx.draw(G, pos=positions, with_labels=True, node_size=3000, 
            node_color='darkgreen', edge_color='blue', font_size=8, 
            font_weight='normal', arrows=False, font_color='white',
            linewidths=1)  # Black border and white text

    # Forward arrows
    if forward_edges:
        for (start, end) in forward_edges:
            plt.arrow(positions[start][0], positions[start][1],
                      positions[end][0] - positions[start][0],
                      positions[end][1] - positions[start][1],
                      color='blue', alpha=1, width=0.0015,
                      head_width=0.1, length_includes_head=True, zorder=2)  # Dark green arrowhead

    # Backward arrows (standard arrowheads)
    if backward_edges:
        for (start, end) in backward_edges:
            plt.arrow(positions[end][0], positions[end][1],
                      positions[start][0] - positions[end][0],
                      positions[start][1] - positions[end][1],
                      color='red', alpha=1, width=0.0015,
                      head_width=0.1, length_includes_head=True, zorder=2)  # Red arrowhead

# Initialize the animation
def init():
    draw_graph([], [])  # No arrows at the start

# Animation function
def animate(i):
    ax.clear()  # Clear the axes for redrawing
    if i % 40 < 20:  # Forward propagation (first 10 frames of each 20)
        forward_edges = G.edges()  # Get all edges for forward propagation
        draw_graph(forward_edges, [])  # Draw forward arrows
        ax.text(2.20, 0.15, "Forward Propagation", fontsize=14, ha='center', color='blue', zorder=3)
    else:  # Backward propagation (next 10 frames of each 20)
        backward_edges = G.edges()  # Get all edges for backward propagation
        draw_graph([], backward_edges)  # Draw backward arrows
        ax.text(2.20, 0.15, "Backward Propagation", fontsize=14, ha='center', color='red', zorder=3)

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=400, init_func=init, repeat=True, interval=100)

# Show the chain rule formula
ax.text(0.0, 1.0, r'Chain Rule: $\frac{dL}{dW} = \frac{dL}{da} \cdot \frac{da}{dW}$', fontsize=9, ha='center', color='darkgreen', zorder=3)

plt.title("Backpropagation with Forward and Backward Arrows in a Neural Network")
plt.show()
