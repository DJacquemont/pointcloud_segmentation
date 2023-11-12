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

  void lineIdentification(std::vector<line>& drone_lines);

  void structureOutput();

  // Publish the points used, and the computed lines as cylinders in RViz
  void visualization();

private:
  ros::NodeHandle node;

  ros::Subscriber tof_pc_sub;
  ros::Subscriber pose_sub;
  ros::Publisher computed_lines_pub;
  ros::Publisher filtered_pc_pub;
  ros::Publisher hough_pc_pub;
  
  Eigen::Vector3d drone_position;
  Eigen::Quaterniond drone_orientation;

  std::vector<line> world_lines;
  std::vector<line> struct_segm;
  std::vector<Eigen::Vector3d> struct_joints;

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
  Eigen::Vector3d pose(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  Eigen::Quaterniond orientation(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
  drone_position = pose;
  drone_orientation = orientation;
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
  if (hough3dlines(*filtered_cloud_XYZ, drone_lines))
    ROS_INFO("ERROR - Unable to perform the Hough transform");

  // Transform the lines from the drone's frame to the world frame
  drone2WorldLines(drone_lines);

  lineIdentification(drone_lines);

  struct_segm.clear();
  struct_joints.clear();

  structureOutput();

  printf("---------------------------------------------------\n");

  // Print the world lines and the struc output
  printf("World lines: %li\n", world_lines.size());
  for (const line& existing_line : world_lines) {
    ROS_INFO("World line: : a = (%f, %f, %f), b = (%f, %f, %f), t_min = %f, t_max = %f",
              existing_line.a.x(), existing_line.a.y(), existing_line.a.z(),
              existing_line.b.x(), existing_line.b.y(), existing_line.b.z(),
              existing_line.t_min, existing_line.t_max);
  }

  printf("Struct output: %li\n", struct_segm.size());
  for (const line& existing_line : struct_segm) {
    ROS_INFO("Struct line: a = (%f, %f, %f), b = (%f, %f, %f), t_min = %f, t_max = %f",
              existing_line.a.x(), existing_line.a.y(), existing_line.a.z(),
              existing_line.b.x(), existing_line.b.y(), existing_line.b.z(),
              existing_line.t_min, existing_line.t_max);
  }

  // printf("Struct joints: %li\n", struct_joints.size());
  // for (size_t i = 0; i < struct_joints.size(); ++i) {
  //   for (size_t j = 0; j < struct_joints[i].size(); ++j) {
  //     ROS_INFO("Struct joint: (%f, %f, %f)",
  //               struct_joints[i][j].x(), struct_joints[i][j].y(), struct_joints[i][j].z());
  //   }
  // }


  visualization();

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
  pass.setFilterLimits(1.0f-drone_position.z(), 1.0f);
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
  Eigen::Matrix3d rotation_matrix = drone_orientation.toRotationMatrix();

  for (line& computed_line : drone_lines) {
    bool lineExists = false;
    bool line_inside = false;

    // Transform the line in the world frame
    line test_line;
    test_line.p1 = rotation_matrix * computed_line.p1 + drone_position;
    test_line.p2 = rotation_matrix * computed_line.p2 + drone_position;
    test_line.a = rotation_matrix * computed_line.a + drone_position;
    test_line.b = rotation_matrix * computed_line.b;
    test_line.t_min = computed_line.t_min;
    test_line.t_max = computed_line.t_max;
    test_line.radius = computed_line.radius;

    for (const Eigen::Vector3d& point : computed_line.points){
      test_line.points.push_back(rotation_matrix * point + drone_position);
    }

    computed_line = test_line;
  }
}

void PtCdProcessing::lineIdentification(std::vector<line>& drone_lines){

  double tolerance = 0.5;

  for (line& test_line : drone_lines) {
    bool lineExists = false;
    bool line_inside = false;

    for (line& existing_line : world_lines) {
      Eigen::Vector3d test_p1 = find_proj(existing_line.a, existing_line.b, test_line.p1);
      Eigen::Vector3d test_p2 = find_proj(existing_line.a, existing_line.b, test_line.p2);

      if (((test_p1-test_line.p1).norm() < tolerance) && ((test_p2-test_line.p2).norm() < tolerance)) {
        
        double ttest_p1 = (test_p1.x() - existing_line.a.x()) / existing_line.b.x();
        double ttest_p2 = (test_p2.x() - existing_line.a.x()) / existing_line.b.x();

        if (ttest_p1 < ttest_p2){
          if (!(ttest_p2 <= existing_line.t_min || ttest_p1 >= existing_line.t_max)){
            lineExists = true;
            if (ttest_p1 < existing_line.t_min){
              existing_line.p1 = test_line.p1;
            }
            if (ttest_p2 > existing_line.t_max){
              existing_line.p2 = test_line.p2;
            }
            if (ttest_p1 > existing_line.t_min && ttest_p2 < existing_line.t_max){
              line_inside = true;
            }
          }
        } else {
          if (!(ttest_p1 <= existing_line.t_min || ttest_p2 >= existing_line.t_max)){
            lineExists = true;
            if (ttest_p2 < existing_line.t_min){
              existing_line.p1 = test_line.p2;
            }
            if (ttest_p1 > existing_line.t_max){
              existing_line.p2 = test_line.p1;
            }
            if (ttest_p2 > existing_line.t_min && ttest_p1 < existing_line.t_max){
              line_inside = true;
            }            
          }
        }

        if (lineExists){
          if (line_inside){
            double coef_learn = 0.2;
            existing_line.radius = (1-coef_learn)*existing_line.radius + coef_learn*test_line.radius;
          } else {
            existing_line.a = existing_line.p1;
            existing_line.b = (existing_line.p2 - existing_line.p1).normalized();
            existing_line.t_min = 0;
            existing_line.t_max = (existing_line.p2.x() - existing_line.a.x()) / existing_line.b.x();
            existing_line.points = test_line.points;
          }
          break;
        }
      }
    } 
    
    if (!lineExists) {
      world_lines.push_back(test_line);
    }
  }
}

void PtCdProcessing::structureOutput(){

  double tolerance = 0.3;

  for (size_t i = 0; i < world_lines.size(); ++i) {
    const line& existing_line_1 = world_lines[i];
    bool hasIntersection = false;

    for (size_t j = i + 1; j < world_lines.size(); ++j) {
      const line& existing_line_2 = world_lines[j];

      Eigen::Vector3d cross_prod = existing_line_2.b.cross(existing_line_1.b).normalized();
      Eigen::Vector3d RHS = existing_line_2.p1 - existing_line_1.p1;
      Eigen::Matrix3d LHS;
      LHS << existing_line_1.b, -existing_line_2.b, cross_prod;

      Eigen::Vector3d solution = LHS.colPivHouseholderQr().solve(RHS);
      
      double dist_lines = abs(solution[2] * cross_prod.norm());


      if (((solution[0] >= existing_line_1.t_min && solution[0] <= existing_line_1.t_max) &&
          (solution[1] > existing_line_2.t_min && solution[1] < existing_line_2.t_max)) &&
          dist_lines < tolerance) {

        Eigen::Vector3d existing_line_1_cross = existing_line_1.p1 + solution[0] * existing_line_1.b;
        Eigen::Vector3d existing_line_2_cross = existing_line_2.p1 + solution[1] * existing_line_2.b;

        line new_line_1 = existing_line_1;
        new_line_1.p1 = existing_line_1_cross;
        new_line_1.t_min = solution[0];
        struct_segm.push_back(new_line_1);

        line new_line_2 = existing_line_1;
        new_line_2.p2 = existing_line_1_cross;
        new_line_2.t_max = solution[0];
        struct_segm.push_back(new_line_2);

        struct_joints.push_back(existing_line_1_cross);

        hasIntersection = true;
        break;
      }
    }
    if (!hasIntersection) {
    // If no intersection is found, add the original line to the output
      struct_segm.push_back(existing_line_1);
    }
  }
}

// Publish the points used, and the computed lines as cylinders in RViz
void PtCdProcessing::visualization() {
  // Create a pointcloud to hold the points used for Hough
  sensor_msgs::PointCloud2 output_hough;
  pcl::PointCloud<pcl::PointXYZ> pc_out;
  // Create a marker array to hold the cylinders
  visualization_msgs::MarkerArray markers;
    

  // Loop through the computed lines and create a cylinder for each line
  for (size_t i = 0; i < struct_segm.size(); i++) {
    for (const Eigen::Vector3d& point : struct_segm[i].points){
      pcl::PointXYZ p_pcl;
      p_pcl.x = point.x();
      p_pcl.y = point.y();
      p_pcl.z = point.z();

      pc_out.points.push_back(p_pcl);
    }
    
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
    cylinder.pose.position.x = (struct_segm[i].p1.x() + struct_segm[i].p2.x()) / 2.0;
    cylinder.pose.position.y = (struct_segm[i].p1.y() + struct_segm[i].p2.y()) / 2.0;
    cylinder.pose.position.z = (struct_segm[i].p1.z() + struct_segm[i].p2.z()) / 2.0;

    // Set the cylinder's orientation
    Eigen::Vector3d direction(struct_segm[i].p2 - struct_segm[i].p1);
    direction.normalize();
    Eigen::Quaterniond q;
    q.setFromTwoVectors(Eigen::Vector3d(0, 0, 1), direction);
    cylinder.pose.orientation.x = q.x();
    cylinder.pose.orientation.y = q.y();
    cylinder.pose.orientation.z = q.z();
    cylinder.pose.orientation.w = q.w();

    // Set the cylinder's scale (height and radius)
    double cylinder_height = (struct_segm[i].p2 - struct_segm[i].p1).norm();
    double cylinder_radius = struct_segm[i].radius;
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

  // Loop through the computed lines and create a sphere for each joint
  for (size_t i = 0; i < struct_joints.size(); i++) {
    visualization_msgs::Marker sphere;

    // Set the marker properties for the cylinder
    sphere.header.frame_id = "mocap";
    sphere.header.stamp = ros::Time::now();
    sphere.ns = "spheres";
    sphere.id = i;
    sphere.action = visualization_msgs::Marker::ADD;
    sphere.pose.orientation.w = 1.0;
    sphere.type = visualization_msgs::Marker::SPHERE;

    // Set the cylinder's position (midpoint between p1 and p2)
    sphere.pose.position.x = struct_joints[i].x();
    sphere.pose.position.y = struct_joints[i].y();
    sphere.pose.position.z = struct_joints[i].z();

    // Set the cylinder's scale (height and radius)
    double sphere_radius = 0.15;
    sphere.scale.x = sphere_radius * 2.0;
    sphere.scale.y = sphere_radius * 2.0;
    sphere.scale.z = sphere_radius * 2.0;

    // Set the cylinder's color and transparency
    sphere.color.r = 0.0;
    sphere.color.g = 1.0;
    sphere.color.b = 0.0;
    sphere.color.a = 1.0;

    // Add the cylinder marker to the marker array
    markers.markers.push_back(sphere);
  }

  // Publishing points used Hough
  pcl::PCLPointCloud2::Ptr pts_hough (new pcl::PCLPointCloud2);
  pcl::toPCLPointCloud2(pc_out, *pts_hough);
  pcl_conversions::fromPCL(*pts_hough, output_hough);
  output_hough.header.frame_id = "mocap";
  hough_pc_pub.publish(output_hough);

  // Publish the marker array containing the cylinders
  computed_lines_pub.publish(markers);
}