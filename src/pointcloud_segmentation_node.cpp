#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <eigen3/Eigen/Dense>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

// Callback function receiving ToF images from the Autopilot package
void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  ROS_INFO("Pointcloud Received");

  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

  pcl::fromROSMsg(*msg, pcl_cloud);
}

// Main function
int main(int argc, char* argv[]){

    ROS_INFO("Pointcloud semgmentation node starting");

    ros::init(argc, argv, "pointcloud_seg");

    ros::NodeHandle n;

    ros::Subscriber tof_pc_sub = n.subscribe("tof_pc", 1000, pointcloudCallback);

    ros::spin();

    return 0;
}
