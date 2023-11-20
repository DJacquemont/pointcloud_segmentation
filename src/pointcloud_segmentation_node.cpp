#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
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

  int cloudLinearShapeCheck(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  void segIdentification(std::vector<line>& drone_lines);

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

  std::vector<double> radius = {0.2, 0.07};
  std::vector<double> theta = {0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, 5*M_PI/4, 3*M_PI/2, 7*M_PI/4};

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

  segIdentification(drone_lines);

  struct_segm.clear();
  struct_joints.clear();

  structureOutput();

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
  pass.setFilterLimits(0.3f-drone_position.z(), 2.0f);
  pass.filter(*filtered_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(0.03f, 0.03f, 0.03f);
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


int PtCdProcessing::cloudLinearShapeCheck(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

  // Performing PCA using PCL
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cloud);

  // Eigenvalues are in descending order
  Eigen::Vector3d eigenvalues = pca.getEigenValues().cast<double>();

  // Assess the distribution of points based on eigenvalues
  double linearityThreshold = 25;  // Define an appropriate threshold
  if (eigenvalues[0] > linearityThreshold * (eigenvalues[1] + eigenvalues[2])) {
    ROS_INFO("Points form a linear structure");
    return 0;
  } else {
    ROS_INFO("Points form a dispersed structure");
    return 1;
  }
}


void PtCdProcessing::segIdentification(std::vector<line>& drone_lines){

  double tolerance = 0.5;
  double coef_learn = 0.25;
  double parallelism_tolerance = 0.2; // Tolerance for parallelism check

  for (const line& test_line : drone_lines) {

    bool addLine = true;

    // Extract points from test_line and create a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : test_line.points) {
      cloud->points.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }

    // Check the shape of the point cloud
    if (cloudLinearShapeCheck(cloud)) {
      addLine = false;
      continue;
    }

    for (size_t i = 0; i < world_lines.size(); ++i) {
      Eigen::Vector3d test_line_p1 = test_line.t_min * test_line.b + test_line.a;
      Eigen::Vector3d test_line_p2 = test_line.t_max * test_line.b + test_line.a;
      Eigen::Vector3d existing_line_p1 = world_lines[i].t_min * world_lines[i].b + world_lines[i].a;
      Eigen::Vector3d existing_line_p2 = world_lines[i].t_max * world_lines[i].b + world_lines[i].a;

      Eigen::Vector3d test_line_proj_p1 = find_proj(world_lines[i].a, world_lines[i].b, test_line_p1);
      Eigen::Vector3d test_line_proj_p2 = find_proj(world_lines[i].a, world_lines[i].b, test_line_p2);

      Eigen::Vector3d cross_product = test_line.b.cross(world_lines[i].b);

      if (((test_line_proj_p1-test_line_p1).norm() < tolerance) && ((test_line_proj_p2-test_line_p2).norm() < tolerance && cross_product.norm() < parallelism_tolerance)) {

        Eigen::Vector3d new_world_line_a = (test_line_proj_p1+test_line_p1)/2;
        Eigen::Vector3d new_world_line_b = ((test_line_proj_p2+test_line_p2)/2 - (test_line_proj_p1+test_line_p1)/2).normalized();

        Eigen::Vector3d test_line_proj_p1 = find_proj(new_world_line_a, new_world_line_b, test_line_p1);
        Eigen::Vector3d test_line_proj_p2 = find_proj(new_world_line_a, new_world_line_b, test_line_p2);
        Eigen::Vector3d existing_line_proj_p1 = find_proj(new_world_line_a, new_world_line_b, existing_line_p1);
        Eigen::Vector3d existing_line_proj_p2 = find_proj(new_world_line_a, new_world_line_b, existing_line_p2);

        double test_line_proj_t1 = (test_line_proj_p1.x() - new_world_line_a.x()) / new_world_line_b.x();
        double test_line_proj_t2 = (test_line_proj_p2.x() - new_world_line_a.x()) / new_world_line_b.x();
        double existing_line_proj_t1 = (existing_line_proj_p1.x() - new_world_line_a.x()) / new_world_line_b.x();
        double existing_line_proj_t2 = (existing_line_proj_p2.x() - new_world_line_a.x()) / new_world_line_b.x();
        
        if (!(std::min(test_line_proj_t1, test_line_proj_t2)>std::max(existing_line_proj_t1, existing_line_proj_t2)) ||
              (std::max(test_line_proj_t1, test_line_proj_t2)<std::min(existing_line_proj_t1, existing_line_proj_t2))){

          addLine = false;

          world_lines[i].a = new_world_line_a;
          world_lines[i].b = new_world_line_b;

          std::vector<double> list_t = {test_line_proj_t1, test_line_proj_t2, existing_line_proj_t1, existing_line_proj_t2};
          world_lines[i].t_max = *std::max_element(list_t.begin(), list_t.end());
          world_lines[i].t_min = *std::min_element(list_t.begin(), list_t.end());
          world_lines[i].radius = (1-coef_learn)*world_lines[i].radius + coef_learn*test_line.radius;
          world_lines[i].points.insert(world_lines[i].points.end(), test_line.points.begin(), test_line.points.end());

          break;
        }
      }
    }
    if (addLine) {
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

      Eigen::Vector3d existing_line_1_p1 = existing_line_1.t_min * existing_line_1.b + existing_line_1.a;
      Eigen::Vector3d existing_line_2_p1 = existing_line_2.t_min * existing_line_2.b + existing_line_2.a;

      Eigen::Vector3d cross_prod = existing_line_2.b.cross(existing_line_1.b).normalized();
      Eigen::Vector3d RHS = existing_line_2_p1 - existing_line_1_p1;
      Eigen::Matrix3d LHS;
      LHS << existing_line_1.b, -existing_line_2.b, cross_prod;

      Eigen::Vector3d solution = LHS.colPivHouseholderQr().solve(RHS);
      
      double dist_lines = abs(solution[2] * cross_prod.norm());


      if (((solution[0] >= existing_line_1.t_min && solution[0] <= existing_line_1.t_max) &&
          (solution[1] > existing_line_2.t_min && solution[1] < existing_line_2.t_max)) &&
          dist_lines < tolerance) {

        Eigen::Vector3d existing_line_1_cross = existing_line_1_p1 + solution[0] * existing_line_1.b;

        line new_line_1 = existing_line_1;
        new_line_1.t_min = solution[0];
        struct_segm.push_back(new_line_1);

        line new_line_2 = existing_line_1;
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
    Eigen::Vector3d segment_p1 = struct_segm[i].t_min * struct_segm[i].b + struct_segm[i].a;
    Eigen::Vector3d segment_p2 = struct_segm[i].t_max * struct_segm[i].b + struct_segm[i].a;


    cylinder.pose.position.x = (segment_p1.x() + segment_p2.x()) / 2.0;
    cylinder.pose.position.y = (segment_p1.y() + segment_p2.y()) / 2.0;
    cylinder.pose.position.z = (segment_p1.z() + segment_p2.z()) / 2.0;

    // Set the cylinder's orientation
    Eigen::Vector3d direction(segment_p2 - segment_p1);
    direction.normalize();
    Eigen::Quaterniond q;
    q.setFromTwoVectors(Eigen::Vector3d(0, 0, 1), direction);
    cylinder.pose.orientation.x = q.x();
    cylinder.pose.orientation.y = q.y();
    cylinder.pose.orientation.z = q.z();
    cylinder.pose.orientation.w = q.w();

    // Set the cylinder's scale (height and radius)
    double cylinder_height = (segment_p2 - segment_p1).norm();
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


// // Publish the points used, and the computed lines as cylinders in RViz
// void PtCdProcessing::visualization() {
//   // Create a pointcloud to hold the points used for Hough
//   sensor_msgs::PointCloud2 output_hough;
//   pcl::PointCloud<pcl::PointXYZ> pc_out;
//   // Create a marker array to hold the cylinders
//   visualization_msgs::MarkerArray markers;
    

//   // Loop through the computed lines and create a cylinder for each line
//   for (size_t i = 0; i < world_lines.size(); i++) {
//     for (const Eigen::Vector3d& point : world_lines[i].points){
//       pcl::PointXYZ p_pcl;
//       p_pcl.x = point.x();
//       p_pcl.y = point.y();
//       p_pcl.z = point.z();

//       pc_out.points.push_back(p_pcl);
//     }
    
//     visualization_msgs::Marker cylinder;

//     // Set the marker properties for the cylinder
//     cylinder.header.frame_id = "mocap";
//     cylinder.header.stamp = ros::Time::now();
//     cylinder.ns = "cylinders";
//     cylinder.id = i;
//     cylinder.action = visualization_msgs::Marker::ADD;
//     cylinder.pose.orientation.w = 1.0;
//     cylinder.type = visualization_msgs::Marker::CYLINDER;

//     // Set the cylinder's position (midpoint between p1 and p2)
//     Eigen::Vector3d segment_p1 = world_lines[i].t_min * world_lines[i].b + world_lines[i].a;
//     Eigen::Vector3d segment_p2 = world_lines[i].t_max * world_lines[i].b + world_lines[i].a;


//     cylinder.pose.position.x = (segment_p1.x() + segment_p2.x()) / 2.0;
//     cylinder.pose.position.y = (segment_p1.y() + segment_p2.y()) / 2.0;
//     cylinder.pose.position.z = (segment_p1.z() + segment_p2.z()) / 2.0;

//     // Set the cylinder's orientation
//     Eigen::Vector3d direction(segment_p2 - segment_p1);
//     direction.normalize();
//     Eigen::Quaterniond q;
//     q.setFromTwoVectors(Eigen::Vector3d(0, 0, 1), direction);
//     cylinder.pose.orientation.x = q.x();
//     cylinder.pose.orientation.y = q.y();
//     cylinder.pose.orientation.z = q.z();
//     cylinder.pose.orientation.w = q.w();

//     // Set the cylinder's scale (height and radius)
//     double cylinder_height = (segment_p2 - segment_p1).norm();
//     double cylinder_radius = world_lines[i].radius;
//     cylinder.scale.x = cylinder_radius * 2.0;
//     cylinder.scale.y = cylinder_radius * 2.0;
//     cylinder.scale.z = cylinder_height;

//     // Set the cylinder's color and transparency
//     cylinder.color.r = 1.0;
//     cylinder.color.g = 0.0;
//     cylinder.color.b = 0.0;
//     cylinder.color.a = 1.0;

//     // Add the cylinder marker to the marker array
//     markers.markers.push_back(cylinder);
//   }

//   // // Loop through the computed lines and create a sphere for each joint
//   // for (size_t i = 0; i < struct_joints.size(); i++) {
//   //   visualization_msgs::Marker sphere;

//   //   // Set the marker properties for the cylinder
//   //   sphere.header.frame_id = "mocap";
//   //   sphere.header.stamp = ros::Time::now();
//   //   sphere.ns = "spheres";
//   //   sphere.id = i;
//   //   sphere.action = visualization_msgs::Marker::ADD;
//   //   sphere.pose.orientation.w = 1.0;
//   //   sphere.type = visualization_msgs::Marker::SPHERE;

//   //   // Set the cylinder's position (midpoint between p1 and p2)
//   //   sphere.pose.position.x = struct_joints[i].x();
//   //   sphere.pose.position.y = struct_joints[i].y();
//   //   sphere.pose.position.z = struct_joints[i].z();

//   //   // Set the cylinder's scale (height and radius)
//   //   double sphere_radius = 0.15;
//   //   sphere.scale.x = sphere_radius * 2.0;
//   //   sphere.scale.y = sphere_radius * 2.0;
//   //   sphere.scale.z = sphere_radius * 2.0;

//   //   // Set the cylinder's color and transparency
//   //   sphere.color.r = 0.0;
//   //   sphere.color.g = 1.0;
//   //   sphere.color.b = 0.0;
//   //   sphere.color.a = 1.0;

//   //   // Add the cylinder marker to the marker array
//   //   markers.markers.push_back(sphere);
//   // }

//   // Publishing points used Hough
//   pcl::PCLPointCloud2::Ptr pts_hough (new pcl::PCLPointCloud2);
//   pcl::toPCLPointCloud2(pc_out, *pts_hough);
//   pcl_conversions::fromPCL(*pts_hough, output_hough);
//   output_hough.header.frame_id = "mocap";
//   hough_pc_pub.publish(output_hough);

//   // Publish the marker array containing the cylinders
//   computed_lines_pub.publish(markers);
// }