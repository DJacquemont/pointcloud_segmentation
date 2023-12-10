import pandas as pd
import matplotlib.pyplot as plt
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

    # Plotting the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data['processing_time'])
    plt.title('Processing Time Boxplot')
    plt.xlabel('Processing Time')
    plt.ylabel('Time (microseconds)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()