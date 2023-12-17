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

    # Set the Seaborn style and context for larger fonts
    sns.set(style="whitegrid")
    sns.set_context("talk")

    # Plotting the boxplot for processing time using Seaborn
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data['processing_time'])
    plt.ylabel('Processing Time [s]', fontsize=20)

    plt.figure(figsize=(10, 8))
    sns.boxplot(x='nblines', y='processing_time', data=data)
    plt.xlabel('Number of Lines', fontsize=20)
    plt.ylabel('Processing Time [s]', fontsize=20)
    # plt.title('Processing Time Distribution by Number of Lines', fontsize=22)
    plt.show()

if __name__ == "__main__":
    main()
