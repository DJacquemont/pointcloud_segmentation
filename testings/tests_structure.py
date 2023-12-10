from controller import Supervisor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os

def get_segments_from_webots(supervisor):
    """Retrieve segment information from Webots."""
    segments_webots = []
    i = 1
    while True:
        node = supervisor.getFromDef(f"SEG{i}")
        if node is None:
            break
        a = node.getPosition()
        # b is the solid's z-axis
        b_raw = [node.getOrientation()[2], node.getOrientation()[5], node.getOrientation()[8]]
        b = list(b_raw / np.linalg.norm(b_raw))
        child = node.getField("children").getMFNode(0)
        cylinder_node = child.getField('geometry').getSFNode()
        height = cylinder_node.getField('height').getSFFloat()
        segments_webots.append({
            'a': a,
            'b': b,
            'endpoints': [-height / 2, height / 2]
        })
        i += 1
    return segments_webots

def read_segments_from_file(data_file_path):
    """Read segment information from a file."""
    df = pd.read_csv(data_file_path, header=0)
    segments_proc = []
    for _, row in df.iterrows():
        segments_proc.append({
            'a': [row['a_x'], row['a_y'], row['a_z']],
            'b': [row['b_x'], row['b_y'], row['b_z']],
            'endpoints': [row['t_min'], row['t_max']]
        })
    return segments_proc

def find_seg_out():
    """Find the path to the processing time data file."""
    pointcloud_segmentation_dir = os.path.dirname(os.path.dirname(__file__))
    yaml_file_path = os.path.join(pointcloud_segmentation_dir, 'config_pc_seg', 'config.yaml')
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    path_to_output = config.get('path_to_output', None)
    data_file_path = os.path.join(path_to_output, 'segments.txt')
    return data_file_path

def are_directions_similar(b1, b2, angle_threshold=0.1):
    """Check if two direction vectors are nearly parallel within a given threshold."""
    b1_normalized = b1 / np.linalg.norm(b1)
    b2_normalized = b2 / np.linalg.norm(b2)
    dot_product = np.dot(b1_normalized, b2_normalized)
    angle = np.arccos(dot_product)
    if abs(angle) < angle_threshold or abs(angle - np.pi) < angle_threshold:
        return True
    else:
        return False

def shortest_distance_between_lines(seg1, seg2):
    """Calculate the distance between the lines middle points"""
    # Calculate the middle points of the lines
    middle_point_1 = np.array(seg1['a']) + np.array(seg1['b']) * (seg1['endpoints'][0] + seg1['endpoints'][1]) / 2
    middle_point_2 = np.array(seg2['a']) + np.array(seg2['b']) * (seg2['endpoints'][0] + seg2['endpoints'][1]) / 2
    distance = np.linalg.norm(middle_point_1 - middle_point_2)
    return distance
    
def get_similar_segments(segments_webots, segments_proc):
    """Compare segments from Webots and from the processing time data file."""
    distance_threshold = 0.5  # Threshold for considering lines close
    similar_lines = []
    for i in range(len(segments_webots)):
        for j in range(len(segments_proc)):
            if are_directions_similar(segments_webots[i]['b'], segments_proc[j]['b']):
                print(f"Lines {i} and {j} have similar directions")
                distance = shortest_distance_between_lines(segments_webots[i], segments_proc[j])
                if distance < distance_threshold:
                    similar_lines.append((i, j, distance))
    return similar_lines

def plot_segments(segments_webots, segments_proc, similar_segments):
    """Plot segments from Webots and from the processing time data file."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Function to draw a single segment
    def draw_segment(segment, color='b', linestyle='-'):
        a, b, endpoints = segment['a'], segment['b'], segment['endpoints']
        points = [np.array(a) + np.array(b) * t for t in endpoints]
        xs, ys, zs = zip(*points)
        ax.plot(xs, ys, zs, color=color, linestyle=linestyle)

    # Draw segments from Webots
    for segment in segments_webots:
        draw_segment(segment, color='b')

    # Draw segments from file
    for segment in segments_proc:
        draw_segment(segment, color='b', linestyle=':')

    # List of colors to cycle through for similar segments
    colors = ['g', 'r', 'c', 'm', 'y', 'k']
    color_index = 0

    # Highlight similar segments with different colors
    for i, j, _ in similar_segments:
        current_color = colors[color_index % len(colors)]
        draw_segment(segments_webots[i], color=current_color)
        draw_segment(segments_proc[j], color=current_color, linestyle=':')

        # Move to the next color for the next pair
        color_index += 1

    plt.show()

def main():
    # Create a Supervisor instance
    supervisor = Supervisor()

    # Get segments from Webots
    segments_webots = get_segments_from_webots(supervisor)

    # Read segments from the file
    segments_proc = read_segments_from_file(find_seg_out())

    # Find similar segments
    similar_segments = get_similar_segments(segments_webots, segments_proc)

    # Print pairs of similar lines
    for pair in similar_segments:
        print(f"Lines {pair[0]} and {pair[1]} are similar with a distance of {pair[2]:.2f}")

    # Plot segments
    plot_segments(segments_webots, segments_proc, similar_segments)

    plt.show()

if __name__ == "__main__":
    main()
