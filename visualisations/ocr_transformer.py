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
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

# Assume input image size (Height x Width): 256 x 256
input_spatial_size = 256
fixed_height = 256  # As per your model's fixed height
num_pooling_layers = 4  # As per your CNN architecture

# Calculate spatial dimensions after each layer
spatial_sizes = [input_spatial_size]

# CNN Layers configurations
cnn_layers_config = [
    {'name': 'Conv Block 1', 'in_channels': 1, 'out_channels': 64},
    {'name': 'Conv Block 2', 'in_channels': 64, 'out_channels': 128},
    {'name': 'Conv Block 3', 'in_channels': 128, 'out_channels': 256},
    {'name': 'Conv Block 4', 'in_channels': 256, 'out_channels': 512},
]

# Transformer Encoder and Decoder configurations
transformer_config = {
    'encoder_layers': 6,
    'decoder_layers': 6,
    'd_model': 512,
    'nhead': 8,
    'dim_feedforward': 2048,
}

# Colors for the cubes
cnn_colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5B7B1']
transformer_colors = ['#C39BD3', '#BB8FCE']
decoder_color = '#F1948A'

# Scaling factors for visualization
spatial_scaling = 0.05  # Adjust to fit the plot nicely
depth_scaling = 0.02

x_positions = []
current_x = 0

# Offset for z-axis labels to prevent overlap
z_label_offset = 5  # Adjust as needed

# Plot CNN layers
for idx, layer in enumerate(cnn_layers_config):
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
    plot_cube(ax, origin, (size_x, size_y, size_z), cnn_colors[idx])

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

# Reshaping Layer
# Since the reshaping layer flattens the spatial dimensions to prepare data for the Transformer, we'll represent it differently.

# Dimensions for visualization
reshaping_width = 1  # Width along x-axis
reshaping_height = 1  # Very thin to represent flattening
reshaping_depth = 5  # Arbitrary depth for visualization

# Origin of the reshaping layer
origin = (current_x, -reshaping_height / 2, -reshaping_depth / 2)

# Plot the reshaping layer
ax.bar3d(origin[0], origin[1], origin[2], reshaping_width, reshaping_height, reshaping_depth, color='#85C1E9',
         alpha=0.6)
ax.text(current_x + reshaping_width / 2, reshaping_height / 2 + 1, reshaping_depth / 2 + 2,
        'Flatten & Reshape', ha='center', va='bottom', fontsize=12)

# Add reshaping layer annotation
annotation = f"Seq Len: {spatial_size * spatial_size}"
ax.text(current_x + reshaping_width / 2, -reshaping_height / 2 - 1, -reshaping_depth / 2 - 1,
        annotation, ha='center', va='top', fontsize=10)

# Update x position
x_positions.append(current_x)
current_x += reshaping_width + 2  # Adjust gap between layers

# Transformer Encoder
encoder_layers = transformer_config['encoder_layers']
d_model = transformer_config['d_model']
nhead = transformer_config['nhead']
dim_feedforward = transformer_config['dim_feedforward']

# Dimensions for visualization
encoder_width = 2
encoder_height = 3
encoder_depth = 10  # Arbitrary depth for visualization

# Origin of the encoder
origin = (current_x, -encoder_height / 2, -encoder_depth / 2)

# Plot the encoder as a taller cube
plot_cube(ax, origin, (encoder_width, encoder_height, encoder_depth), transformer_colors[0])

# Adjusted z position for the encoder label
z_position = encoder_depth / 2 + 2 + (len(cnn_layers_config)) * z_label_offset

# Add encoder label
ax.text(current_x + encoder_width / 2, encoder_height / 2 + 2, z_position,
        f'Transformer Encoder\n{encoder_layers} Layers', ha='center', va='bottom', fontsize=12)

# Add encoder annotation
annotation = f"d_model: {d_model}, Heads: {nhead}"
ax.text(current_x + encoder_width / 2, -encoder_height / 2 - 1,
        -encoder_depth / 2 - 1 - (len(cnn_layers_config)) * z_label_offset,
        annotation, ha='center', va='top', fontsize=10)

# Update x position
x_positions.append(current_x)
current_x += encoder_width + 2  # Adjust gap between layers

# Transformer Decoder
decoder_layers = transformer_config['decoder_layers']

# Dimensions for visualization
decoder_width = 2
decoder_height = 3
decoder_depth = 10  # Arbitrary depth for visualization

# Origin of the decoder
origin = (current_x, -decoder_height / 2, -decoder_depth / 2)

# Plot the decoder as a taller cube
plot_cube(ax, origin, (decoder_width, decoder_height, decoder_depth), transformer_colors[1])

# Adjusted z position for the decoder label
z_position = decoder_depth / 2 + 2 + (len(cnn_layers_config) + 1) * z_label_offset

# Add decoder label
ax.text(current_x + decoder_width / 2, decoder_height / 2 + 2, z_position,
        f'Transformer Decoder\n{decoder_layers} Layers', ha='center', va='bottom', fontsize=12)

# Add decoder annotation
annotation = f"d_model: {d_model}, Heads: {nhead}"
ax.text(current_x + decoder_width / 2, -decoder_height / 2 - 1,
        -decoder_depth / 2 - 1 - (len(cnn_layers_config) + 1) * z_label_offset,
        annotation, ha='center', va='top', fontsize=10)

# Update x position
x_positions.append(current_x)
current_x += decoder_width + 2  # Adjust gap between layers

# Output Layer
# Represented as a flat rectangle labeled 'Output'

# Dimensions for visualization
output_width = 1
output_height = 1
output_depth = 5  # Arbitrary depth

# Origin of the output layer
origin = (current_x, -output_height / 2, -output_depth / 2)

# Plot the output layer
ax.bar3d(origin[0], origin[1], origin[2], output_width, output_height, output_depth, color=decoder_color, alpha=0.6)
ax.text(current_x + output_width / 2, output_height / 2 + 1, output_depth / 2 + 2,
        'Output Layer', ha='center', va='bottom', fontsize=12)

# Add output layer annotation
annotation = f"Classes: {transformer_config['output_dim'] if 'output_dim' in transformer_config else 'N/A'}"
ax.text(current_x + output_width / 2, -output_height / 2 - 1, -output_depth / 2 - 1,
        annotation, ha='center', va='top', fontsize=10)

# Draw arrows between all layers
for i in range(len(x_positions) - 1):
    # Start point (end of current cube)
    start_x = x_positions[i] + 2 if i < len(cnn_layers_config) else x_positions[i] + 2
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
input_annotation = f"{cnn_layers_config[0]['in_channels']}@{input_spatial_size}x{input_spatial_size}"
ax.text(x_positions[0] - 1, 0, 0 - z_label_offset, input_annotation,
        ha='center', va='center', fontsize=10)

# Set labels and limits
ax.set_xlabel('Network Depth (Layers)', fontsize=12)
ax.set_ylabel('Spatial Dimensions (Height/Width)', fontsize=12)
ax.set_zlabel('Channels / Model Dimensions', fontsize=12)
ax.set_xlim(-5, current_x + 5)
ax.set_ylim(-10, 10)
ax.set_zlim(-15, decoder_depth / 2 + 2 + (len(cnn_layers_config) + len(transformer_config)) * z_label_offset + 5)
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