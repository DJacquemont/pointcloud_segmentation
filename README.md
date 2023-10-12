# pointcloud_segmentation

## Dependencies
- [auto_pilot](https://gitlab.epfl.ch/waelti/auto_pilot) package (drone and sensors simulation)

## Submodules

This repo includes the following submodules:
- [hough-3d-lines](https://github.com/LucasWaelti/hough-3d-lines) (cloned with ssh)

To update the submodules (typically after cloning this repo), run the command:
```bash
git submodule update --init --recursive
```

## Getting started

To launch the package and the AutoPilot package use
```roslaunch pointcloud_segmentation all.launch```

To run only the segmentation node use 
```rosrun pointcloud_segmentation pointcloud_segmentation_node```

To run only the pointcloud TF transform broadcaster use
```rosrun pointcloud_segmentation pointcloud_tfbr_node```
