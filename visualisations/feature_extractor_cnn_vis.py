import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_cube(ax, origin, size, color, label=None):
    """
    Plots a cube in 3D space.

    Parameters:
    - ax: The matplotlib 3D axes object.
    - origin: A tuple (x, y, z) representing the cube's origin.
    - size: A tuple (l, w, h) representing the cube's dimensions.
    - color: The face color of the cube.
    - label: Optional label to place on the cube.
    """
    o = np.array(origin)
    l, w, h = size
    # Define the cube's vertices
    x = [o[0], o[0] + l]
    y = [o[1], o[1] + w]
    z = [o[2], o[2] + h]
    # Vertices of the cube
    vertices = [[x[0], y[0], z[0]],
                [x[1], y[0], z[0]],
                [x[1], y[1], z[0]],
                [x[0], y[1], z[0]],
                [x[0], y[0], z[1]],
                [x[1], y[0], z[1]],
                [x[1], y[1], z[1]],
                [x[0], y[1], z[1]]]
    # Define the 6 faces of the cube
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
             [vertices[4], vertices[5], vertices[6], vertices[7]],
             [vertices[0], vertices[1], vertices[5], vertices[4]],
             [vertices[2], vertices[3], vertices[7], vertices[6]],
             [vertices[1], vertices[2], vertices[6], vertices[5]],
             [vertices[4], vertices[7], vertices[3], vertices[0]]]
    # Plot the cube
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=color, linewidths=1, edgecolors='black', alpha=0.6))
    # Add label if provided
    if label:
        ax.text(o[0] + l / 2, o[1] + w / 2, o[2] + h / 2, label,
                ha='center', va='center', fontsize=10, color='black')


# Set up the figure and 3D axes
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Assume input image size (Height x Width): 256 x 256
input_spatial_size = 256

# Calculate spatial dimensions after each layer
spatial_sizes = [input_spatial_size]

# Layers configurations
layers_config = [
    {'name': 'Conv Block 1', 'in_channels': 1, 'out_channels': 64},
    {'name': 'Conv Block 2', 'in_channels': 64, 'out_channels': 128},
    {'name': 'Conv Block 3', 'in_channels': 128, 'out_channels': 256},
    {'name': 'Conv Block 4', 'in_channels': 256, 'out_channels': 512},
]

# Colors for the cubes
colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5B7B1']

# Scaling factors for visualization
spatial_scaling = 0.05  # Adjust to fit the plot nicely
depth_scaling = 0.02

x_positions = []
current_x = 0

# Offset for z-axis labels to prevent overlap
z_label_offset = 5  # Adjust as needed

# Plot each layer as a cube
for idx, layer in enumerate(layers_config):
    # Update spatial size (after Conv + Pooling)
    if idx == 0:
        # First Conv layer, spatial size remains the same after Conv
        spatial_size = spatial_sizes[-1]
    else:
        # For subsequent layers, spatial size is halved after pooling
        spatial_size = spatial_sizes[-1] // 2
    spatial_sizes.append(spatial_size)

    # Dimensions for visualization
    size_x = 2  # Constant width along x-axis (network depth)
    size_y = spatial_size * spatial_scaling  # Spatial dimension scaled
    size_z = layer['out_channels'] * depth_scaling  # Depth scaled

    # Origin of the cube
    origin = (current_x, -size_y / 2, -size_z / 2)

    # Plot the cube
    plot_cube(ax, origin, (size_x, size_y, size_z), colors[idx])

    # Adjusted z position for the layer label to prevent overlap
    z_position = size_z / 2 + 2 + idx * z_label_offset

    # Add layer label
    ax.text(current_x + size_x / 2, size_y / 2 + 2, z_position,
            layer['name'], ha='center', va='bottom', fontsize=12)

    # Add dimension annotations (adjusted z position)
    annotation = f"{layer['out_channels']}@{spatial_size}x{spatial_size}"
    ax.text(current_x + size_x / 2, -size_y / 2 - 1, -size_z / 2 - 1 - idx * z_label_offset,
            annotation, ha='center', va='top', fontsize=10)

    # Update x position for the next layer
    x_positions.append(current_x)
    current_x += size_x + 2  # Adjust gap between layers

# Draw arrows between the layers
for i in range(len(x_positions) - 1):
    # Start point (end of current cube)
    start_x = x_positions[i] + 2
    start_y = 0
    start_z = 0
    # End point (start of next cube)
    end_x = x_positions[i + 1]
    end_y = 0
    end_z = 0
    # Draw arrow
    ax.quiver(start_x, start_y, start_z,
              end_x - start_x, end_y - start_y, end_z - start_z,
              arrow_length_ratio=0.1, color='black', linewidth=1)

# Input Layer Annotation
input_annotation = f"{layers_config[0]['in_channels']}@{input_spatial_size}x{input_spatial_size}"
ax.text(x_positions[0] - 1, 0, 0 - z_label_offset, input_annotation,
        ha='center', va='center', fontsize=10)

# Set labels and limits
ax.set_xlabel('Network Depth (Layers)', fontsize=12)
ax.set_ylabel('Spatial Dimensions (Height/Width)', fontsize=12)
ax.set_zlabel('Channels (Feature Maps)', fontsize=12)
ax.set_xlim(-5, current_x + 5)
ax.set_ylim(-10, 10)
ax.set_zlim(-15, size_z / 2 + 2 + (len(layers_config) - 1) * z_label_offset + 5)
ax.view_init(elev=20, azim=-60)  # Adjust the view angle

# Hide grid lines
ax.grid(False)
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Display the plot
plt.tight_layout()
plt.show()