#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>

#include <eigen3/Eigen/Dense>
#include "hough_3d_lines.h"

#include <chrono>
using namespace std::chrono;

// Class caring for the node's communication (Subscription & Publication)
class PtCdProcessing
{
public:
  PtCdProcessing()
  {
    tof_pc_sub = node.subscribe("/tof_pc", 1, &PtCdProcessing::pointcloudCallback, this);
    pose_sub = node.subscribe("/mavros/local_position/pose", 1, &PtCdProcessing::poseCallback, this);
    marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    filtered_pc_pub = node.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1);
    hough_pc_pub = node.advertise<sensor_msgs::PointCloud2>("hough_pointcloud", 1);
  }

  // Callback function receiving & processing ToF images from the Autopilot package
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

  // Callback function receiving & storing the drone's pose from the Autopilot package
  void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

  // Filtering pointcloud
  void cloudFiltering(pcl::PCLPointCloud2::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud_XYZ);

  // Transform the lines from the drone's frame to the world frame
  void drone2WorldLines(std::vector<line>& drone_lines);

  // Publish the computed lines in rviz
  void linesVisualization();

private:
  ros::NodeHandle node;

  ros::Subscriber tof_pc_sub;
  ros::Subscriber pose_sub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pc_pub;
  ros::Publisher hough_pc_pub;
  
  geometry_msgs::Point drone_position;
  geometry_msgs::Quaternion drone_orientation;

  std::vector<line> world_lines;
};

//--------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------

int main(int argc, char *argv[])
{

  ROS_INFO("Pointcloud semgmentation node starting");

  ros::init(argc, argv, "pointcloud_seg");

  initHoughSpace();

  PtCdProcessing SAPobject;

  ros::spin();

  return 0;
}

//--------------------------------------------------------------------
// Callback functions
//--------------------------------------------------------------------

// Callback function receiving & storing the drone's pose from the Autopilot package
void PtCdProcessing::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  drone_position = msg->pose.position;
  drone_orientation = msg->pose.orientation;
}

// Callback function receiving & processing ToF images from the Autopilot packag
void PtCdProcessing::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  auto callStart = high_resolution_clock::now();

  // Filtering pointcloud
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_XYZ( new pcl::PointCloud<pcl::PointXYZ> );
  pcl_conversions::toPCL(*msg, *cloud);
  cloudFiltering(cloud, filtered_cloud_XYZ);

  // line extraction with the Hough transform
  std::vector<line> drone_lines;
  pcl::PointCloud<pcl::PointXYZ> pc_out;
  if (hough3dlines(*filtered_cloud_XYZ, drone_lines, pc_out))
    ROS_INFO("ERROR - Unable to perform the Hough transform");

  // Publishing points used Hough
  pcl::PCLPointCloud2::Ptr pts_hough (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 output_hough;
  pcl::toPCLPointCloud2(pc_out, *pts_hough);
  pcl_conversions::fromPCL(*pts_hough, output_hough);
  output_hough.header.frame_id = "world";
  hough_pc_pub.publish(output_hough);

  // Transform the lines from the drone's frame to the world frame
  drone2WorldLines(drone_lines);

  linesVisualization();

  auto callEnd = high_resolution_clock::now();
  ROS_INFO("Callback execution time: %ld us",
            duration_cast<microseconds>(callEnd - callStart).count());
}


//--------------------------------------------------------------------
// Utility functions
//--------------------------------------------------------------------

// Filtering pointcloud
void PtCdProcessing::cloudFiltering(pcl::PCLPointCloud2::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud_XYZ){
  pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2);

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
  pass.setFilterLimits(1-drone_position.z, 1.0f);
  pass.filter(*filtered_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_grid.filter(*filtered_cloud);

  std::vector<line> drone_lines;
  pcl::PointCloud<pcl::PointXYZ> pc_out;
  pcl::fromPCLPointCloud2(*filtered_cloud, *filtered_cloud_XYZ );

  // Publishing PCL filtered cloud
  sensor_msgs::PointCloud2 output_filtered;
  pcl_conversions::fromPCL(*filtered_cloud, output_filtered);
  filtered_pc_pub.publish(output_filtered);
}


// Transform the lines from the drone's frame to the world frame
void PtCdProcessing::drone2WorldLines(std::vector<line>& drone_lines){

  // Convert the quaternion to a rotation matrix
  Eigen::Quaterniond q(drone_orientation.w, drone_orientation.x, drone_orientation.y, drone_orientation.z);
  Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();

  for (const line& computed_line : drone_lines) {
    line world_line;

    // Transform the first endpoint
    Eigen::Vector3d drone_point1(computed_line.p1.x, computed_line.p1.y, computed_line.p1.z);
    Eigen::Vector3d world_point1 = rotation_matrix * drone_point1 + Eigen::Vector3d(drone_position.x, drone_position.y, drone_position.z);
    world_line.p1.x = world_point1.x();
    world_line.p1.y = world_point1.y();
    world_line.p1.z = world_point1.z();

    // Transform the second endpoint
    Eigen::Vector3d drone_point2(computed_line.p2.x, computed_line.p2.y, computed_line.p2.z);
    Eigen::Vector3d world_point2 = rotation_matrix * drone_point2 + Eigen::Vector3d(drone_position.x, drone_position.y, drone_position.z);
    world_line.p2.x = world_point2.x();
    world_line.p2.y = world_point2.y();
    world_line.p2.z = world_point2.z();

    // Add the transformed line to the world_lines vector
    world_lines.push_back(world_line);
  }
}


// Publish the computed lines in rviz
void PtCdProcessing::linesVisualization(){

  // Rviz markers
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "mocap";
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
  for (size_t i = 0; i < world_lines.size(); i++)
  {
    geometry_msgs::Point p1, p2;
    p1.x = world_lines[i].p1.x;
    p1.y = world_lines[i].p1.y;
    p1.z = world_lines[i].p1.z;
    p2.x = world_lines[i].p2.x;
    p2.y = world_lines[i].p2.y;
    p2.z = world_lines[i].p2.z;
    line_list.points.push_back(p1);
    line_list.points.push_back(p2);
  }

  // Publishing line Hough
  marker_pub.publish(line_list);
}