import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

def main():
    # Extracting the filepath to the pointcloud_segmentation directory
    pointcloud_segmentation_dir = os.path.dirname(os.path.dirname(__file__))

    # Path to the YAML configuration file
    yaml_file_path = pointcloud_segmentation_dir + '/config_pc_seg/config.yaml'

    # Read the YAML file to get the path to the processing time data file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract the path to the processing time data file
    path_to_output = config.get('path_to_output', None)
    data_file_path = os.path.join(path_to_output, 'processing_time.csv')

    # Load the data from the text file
    data = pd.read_csv(data_file_path, header=0)

    data['wall_time'] = data['wall_time'].astype(float)/10e6
    data['processing_time'] = data['processing_time'].astype(float)/10e6
    data['seg_vec_size'] = data['seg_vec_size'].astype(int)

    # Set the Seaborn style and context for larger fonts
    sns.set(style="whitegrid")
    sns.set_context("talk")  # This line sets a larger scale for fonts and elements

    # Creating a figure for subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plotting the boxplot for processing time using Seaborn
    sns.boxplot(data=data['processing_time'], ax=axes[0])
    axes[0].set_title('Processing Time Boxplot', fontsize=18)
    axes[0].set_ylabel('Time [s]', fontsize=16)

    # Plotting the line and scatter plot for seg_vec_size using Seaborn
    sns.lineplot(x='wall_time', y='seg_vec_size', data=data, ax=axes[1])
    sns.scatterplot(x='wall_time', y='seg_vec_size', data=data, color='red', ax=axes[1])
    axes[1].set_title('Segment Vector Size Over Time', fontsize=18)
    axes[1].set_xlabel('Wall Time [s]', fontsize=16)
    axes[1].set_ylabel('Segment Vector Size', fontsize=16)

    plt.show()

if __name__ == "__main__":
    main()
