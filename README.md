# Point Cloud Segmentation of Infrastructural Steel Elements

## Overview

This project focuses on developing a real-time ROS segmentation pipeline. The project utilizes a Time of Flight (ToF) camera mounted on a Micro Aerial Vehicle (MAV) to generate 3D point-cloud data, which is then processed to extract 3D structures.

The core of this project is to adapt and integrate the [`hough-3d-lines`](https://github.com/LucasWaelti/hough-3d-lines) C++ library into a ROS node for real-time operation. The library applies the 3D Hough Transform for effective line segmentation in 3D point clouds. 

## Installation and Setup

### Prerequisites
- Ubuntu 20.04
- Webots 2023a
- C++ Compiler with C++17 support
- [auto_pilot](https://gitlab.epfl.ch/waelti/auto_pilot) package (drone and sensors simulation)
- ROS Noetic
- Eigen3
- PCL v1.10

This repo includes the following submodules:
- [hough-3d-lines](https://github.com/LucasWaelti/hough-3d-lines) (cloned with ssh)

To update the submodules (typically after cloning this repo), run the command:
```bash
git submodule update --init --recursive
```

### Getting started

#### Configuration file

The YAML file `config.yaml` sets the nodes parameters:
- vebose_level: NONE - 0, INFO - 1, DEBUG - 2
- path_to_output: path to the processed output data (segments and intersections)
- floor_trim_height: cut off the pointcloud below this height
- min_pca_coeff: minimum pca coeff to be considered as a line
- min_weight: minimum weight coefficient for the line’s fusion
- rad_2_leaf_ratio: ratio of radius to leaf size to determine leaf size
- opt_minvotes: minimum number of votes to be considered a line
- granularity: granularity of the search (Hough transform) between 0 and 6
- opt_nlines: number of lines to be detected
- radius_sizes: beam radius sizes

#### Simulation

First open your favorite simulation world on webots 2023. 

To launch all the packages in the simulation world `flying_arena_ros_tower.wbt`, with the waypoints `wp_tower.csv`
```bash
roslaunch pointcloud_segmentation all.launch
```

To launch all the packages in the simulation world `flying_arena_ros_mockup.wbt`, with the waypoints `wp_mockup.csv`
```bash
roslaunch pointcloud_segmentation all.launch waypoints:=wp_mockup.csv
```

To launch all the packages in the simulation world `flying_arena_ros_obs.wbt`
```bash
roslaunch pointcloud_segmentation all.launch trajectory:=eight
```

To launch all the packages in the simulation world `flying_arena_ros_obs_tests.wbt`
```bash
roslaunch pointcloud_segmentation all.launch waypoints:=wp_tests.csv
```
#### Standalone

To only run the pointcloud segmentation node 
```bash
rosrun pointcloud_segmentation pointcloud_segmentation_node
```

To only run the pointcloud TF transform broadcaster
```bash
rosrun pointcloud_segmentation pointcloud_tfbr_node
```

#### Output
Three output csv files are created at the `path_to_output` path specified in the configuration file `config.yaml`.
- `intersection.csv`
- `segments.csv`
- `processing_time.csv`


## Implementation Details

### Key Components
- **Hough-3D-Lines Library**: An open-source C++ library used for 3D line segmentation using the Hough Transform. Adapted and enhanced for this project.
- **ROS Node Integration**: The library is wrapped within a ROS node, allowing for real-time data processing and interaction with other ROS components.
- **PointCloud Processing**: Includes pre-processing steps like filtering and segmentation on the 3D point-cloud data received from the ToF camera.

### Pipeline Overview
1. **Data Acquisition**: 3D point cloud data is captured by the ToF camera mounted on the MAV.
2. **Pre-processing**: Raw data is filtered to remove noise and irrelevant information.
3. **Segmentation**: The Hough-3D-Lines library processes the filtered data to identify and extract line segments.
4. **Post-processing**: Extracted lines are analyzed to build a structural representation of the environment.
5. **Visualization**: Real-time visualization of the segmented lines and structures is provided in RViz.

