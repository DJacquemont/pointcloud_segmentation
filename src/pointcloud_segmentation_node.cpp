#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <visualization_msgs/Marker.h>

#include <eigen3/Eigen/Dense>
#include "hough_3d_lines.h"

// Class caring for the node's communication (Subscription & Publication)
class PtCdProcessing
{
public:
  PtCdProcessing()
  {
    tof_pc_sub = node.subscribe("/tof_pc", 1, &PtCdProcessing::pointcloudCallback, this);
    marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    filtered_pc_pub = node.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1);
    hough_pc_pub = node.advertise<sensor_msgs::PointCloud2>("hough_pointcloud", 1);
  }

  // Callback function receiving ToF images from the Autopilot package
  // and publishing marker lines to vizualise the computed lines in rviz
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

private:
  ros::NodeHandle node;
  ros::Subscriber tof_pc_sub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pc_pub;
  ros::Publisher hough_pc_pub;
};

//--------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------

int main(int argc, char *argv[])
{

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
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2);
  pcl_conversions::toPCL(*msg, *cloud);

  // Filtering pointcloud
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(-2.0f, 2.0f);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-2.0f, 2.0f);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(-0.75f, 1.0f);
  pass.filter(*filtered_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_grid.filter(*filtered_cloud);

  // line extraction with the Hough transform
  std::vector<line> computed_lines;
  pcl::PointCloud<pcl::PointXYZ> pc_out;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_XYZ( new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*filtered_cloud, *filtered_cloud_XYZ ); 
  if (hough3dlines(*filtered_cloud_XYZ, computed_lines, pc_out))
    ROS_INFO("ERROR - Unable to perform the Hough transform");

  // Rviz markers
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "world";
  line_list.header.stamp = ros::Time::now();
  line_list.ns = "points_and_lines";
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.05;
  line_list.scale.y = 0.05;
  line_list.scale.z = 0.05;
  line_list.color.r = 1.0;
  line_list.color.a = 1.0;

  // Loop through the computed lines and add them to the marker list
  for (size_t i = 0; i < computed_lines.size(); i++)
  {
    geometry_msgs::Point p1, p2;
    p1.x = computed_lines[i].p1.x;
    p1.y = computed_lines[i].p1.y;
    p1.z = computed_lines[i].p1.z;
    p2.x = computed_lines[i].p2.x;
    p2.y = computed_lines[i].p2.y;
    p2.z = computed_lines[i].p2.z;
    line_list.points.push_back(p1);
    line_list.points.push_back(p2);
  }

  // Publishing line Hough
  marker_pub.publish(line_list);

  // Publishing points used Hough
  pcl::PCLPointCloud2::Ptr pts_hough (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 output_hough;
  pcl::toPCLPointCloud2(pc_out, *pts_hough);
  pcl_conversions::fromPCL(*pts_hough, output_hough);
  output_hough.header.frame_id = "world";
  hough_pc_pub.publish(output_hough);

  // Publishing PCL filtered cloud
  sensor_msgs::PointCloud2 output_filtered;
  pcl_conversions::fromPCL(*filtered_cloud, output_filtered);
  filtered_pc_pub.publish(output_filtered);
}