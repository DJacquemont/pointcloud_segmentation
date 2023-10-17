#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <visualization_msgs/Marker.h>

#include <eigen3/Eigen/Dense>
#include "hough_3d_lines.h"



// Class caring for the node's communication (Subscription & Publication)
class PtCdProcessing
{
  public:

  PtCdProcessing() {
    tof_pc_sub = node.subscribe("/tof_pc", 1000, &PtCdProcessing::pointcloudCallback, this);
    marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 10);
    filtered_pc_pub = node.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 11);
  }

  // Callback function receiving ToF images from the Autopilot package
  // and publishing marker lines to vizualise the computed lines in rviz
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

  private:
  ros::NodeHandle node;
  ros::Subscriber tof_pc_sub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pc_pub;
};



//--------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------

int main(int argc, char* argv[]){

  ROS_INFO("Pointcloud semgmentation node starting");

  ros::init(argc, argv, "pointcloud_seg");

  PtCdProcessing SAPobject;

  ros::spin();

  return 0;
}


//--------------------------------------------------------------------
// Callback functions
//--------------------------------------------------------------------

// Callback function receiving ToF images from the Autopilot package
// and publishing marker lines to vizualise the computed lines in rviz
void PtCdProcessing::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    ROS_INFO("Pointcloud Received : w %d, h %d, is_dense %d", 
                    cloud->width, cloud->height, cloud->is_dense);

    // Filtering pointcloud
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud (cloud);
    voxel_grid.setLeafSize (0.005f, 0.005f, 0.005f);
    voxel_grid.filter(*cloud);

    ROS_INFO("Number of points after filter: %li", cloud->points.size());

    // Rviz markers
    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "world";
    line_list.header.stamp = ros::Time::now();
    line_list.ns = "points_and_lines";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.1;
    line_list.scale.y = 0.1;
    line_list.scale.z = 0.1;
    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    // line extraction with the Hough transform
    if (hough3dlines(*cloud, line_list))
      ROS_INFO("ERROR - Unable to perform the Hough transform");

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);

    
    marker_pub.publish(line_list);
    filtered_pc_pub.publish(output);
  }