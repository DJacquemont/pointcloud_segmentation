#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
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
    computed_lines_pub = node.advertise<visualization_msgs::MarkerArray>("computed_lines", 1);
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
  ros::Publisher computed_lines_pub;
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
  pass.setFilterLimits(1.0f-drone_position.z, 1.0f);
  pass.filter(*filtered_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_grid.filter(*filtered_cloud);

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

    bool lineExists = false;

    for (const line& existing_line : world_lines) {
      // Define a tolerance for considering lines as equal
      double tolerance_point = 0.6;
      double tolerance_slope_a = 1;
      double tolerance_slope_b = 1;

      // Check if the endpoints and radius are sufficiently close
      if ((computed_line.p1.isApprox(existing_line.p1, tolerance_point) && computed_line.p2.isApprox(existing_line.p2, tolerance_point)) ||
          (computed_line.p1.isApprox(existing_line.p2, tolerance_point) && computed_line.p2.isApprox(existing_line.p1, tolerance_point)) ||
          (computed_line.a.isApprox(existing_line.a, tolerance_slope_a) && computed_line.b.isApprox(existing_line.b, tolerance_slope_b))) {
        lineExists = true;
        break;
      }
    }

    if (!lineExists) {
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

      world_line.radius = computed_line.radius;

      // Add the transformed line to the world_lines vector
      world_lines.push_back(world_line);
    }
  }
}

// Publish the computed lines as cylinders in RViz
void PtCdProcessing::linesVisualization() {
  // Create a marker array to hold the cylinders
  visualization_msgs::MarkerArray markers;

  // Loop through the computed lines and create a cylinder for each line
  for (size_t i = 0; i < world_lines.size(); i++) {
    visualization_msgs::Marker cylinder;

    // Set the marker properties for the cylinder
    cylinder.header.frame_id = "mocap";
    cylinder.header.stamp = ros::Time::now();
    cylinder.ns = "cylinders";
    cylinder.id = i;
    cylinder.action = visualization_msgs::Marker::ADD;
    cylinder.pose.orientation.w = 1.0;
    cylinder.type = visualization_msgs::Marker::CYLINDER;

    // Set the cylinder's position (midpoint between p1 and p2)
    cylinder.pose.position.x = (world_lines[i].p1.x + world_lines[i].p2.x) / 2.0;
    cylinder.pose.position.y = (world_lines[i].p1.y + world_lines[i].p2.y) / 2.0;
    cylinder.pose.position.z = (world_lines[i].p1.z + world_lines[i].p2.z) / 2.0;

    // Set the cylinder's orientation
    Eigen::Vector3d direction(world_lines[i].p2.x - world_lines[i].p1.x,
                              world_lines[i].p2.y - world_lines[i].p1.y,
                              world_lines[i].p2.z - world_lines[i].p1.z);
    direction.normalize();
    Eigen::Quaterniond q;
    q.setFromTwoVectors(Eigen::Vector3d(0, 0, 1), direction);
    cylinder.pose.orientation.x = q.x();
    cylinder.pose.orientation.y = q.y();
    cylinder.pose.orientation.z = q.z();
    cylinder.pose.orientation.w = q.w();

    // Set the cylinder's scale (height and radius)
    double cylinder_height = (world_lines[i].p2 - world_lines[i].p1).norm();
    double cylinder_radius = world_lines[i].radius;
    cylinder.scale.x = cylinder_radius * 2.0;
    cylinder.scale.y = cylinder_radius * 2.0;
    cylinder.scale.z = cylinder_height;

    // Set the cylinder's color and transparency
    cylinder.color.r = 1.0;
    cylinder.color.g = 0.0;
    cylinder.color.b = 0.0;
    cylinder.color.a = 1.0;

    // Add the cylinder marker to the marker array
    markers.markers.push_back(cylinder);
  }

  // Publish the marker array containing the cylinders
  computed_lines_pub.publish(markers);
}