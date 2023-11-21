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


const size_t MAX_POSE_HISTORY_SIZE = 50;
const double FLOOR_TRIM_HEIGHT = 0.3;

enum verbose {NONE, INFO, WARN};
const int VERBOSE = WARN;

// structure storing the drone's pose with a timestamp
struct PoseWithTimestamp {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    ros::Time timestamp;
};

// Class in charge of the node's communication (Subscription & Publication)
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

  // Callback function receiving & storing the drone's pose from the Autopilot package
  void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

  // Callback function receiving & processing ToF images from the Autopilot package
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

  // Finds the closest correct pointcloud pose to the given timestamp
  void findClosestPose(const ros::Time& timestamp);

  // Filtering pointcloud
  void cloudFiltering(pcl::PCLPointCloud2::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud_XYZ);

  // Transform the lines from the drone's frame to the world frame
  void drone2WorldLines(std::vector<segment>& drone_lines);

  // Check if the point cloud is linear
  int cloudLinearShapeCheck(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  // Identify intersections between lines
  void segIdentification(std::vector<segment>& drone_lines);

  // Output the structure
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
  
  std::vector<PoseWithTimestamp> poseHistory;
  Eigen::Vector3d drone_position;
  Eigen::Quaterniond drone_orientation;

  std::vector<segment> world_lines;
  std::vector<segment> struct_segm;
  std::vector<Eigen::Vector3d> struct_joints;
  std::vector<std::vector<std::tuple<double, double>>> intersection_matrix;

  // Parameters
  const std::vector<double> radius_sizes = {0.03}; // 0.2, 0.07, 0.05, 0.1
  const double leaf_size = std::min(radius_sizes[0], radius_sizes[radius_sizes.size()-1]);
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
    PoseWithTimestamp newPose;
    newPose.position = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    newPose.orientation = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    newPose.timestamp = msg->header.stamp;

    poseHistory.push_back(newPose);

    if (poseHistory.size() > MAX_POSE_HISTORY_SIZE) {
        // Remove the oldest entry
        poseHistory.erase(poseHistory.begin());
    }
}



// Callback function receiving & processing ToF images from the Autopilot packag
void PtCdProcessing::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  auto callStart = high_resolution_clock::now();

  // Find the closest pose in time to the point cloud's timestamp
  findClosestPose(msg->header.stamp);

  // Filtering pointcloud
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_XYZ( new pcl::PointCloud<pcl::PointXYZ> );
  pcl_conversions::toPCL(*msg, *cloud);
  cloudFiltering(cloud, filtered_cloud_XYZ);

  // segment extraction with the Hough transform
  std::vector<segment> drone_lines;
  const double opt_dx = sqrt(3)*leaf_size; // optimal discretization step
  if (hough3dlines(*filtered_cloud_XYZ, drone_lines, opt_dx, radius_sizes, VERBOSE) && VERBOSE > INFO){
    ROS_WARN("Unable to perform the Hough transform");
  }

  if (VERBOSE > NONE){
    ROS_INFO("Number of lines detected: %ld", drone_lines.size());
  }

  // Transform the lines from the drone's frame to the world frame
  drone2WorldLines(drone_lines);

  // Filter the segments that already exist in the world_lines
  segIdentification(drone_lines);

  // Identify intersections between lines, and ready the data for output
  structureOutput();

  visualization();

  auto callEnd = high_resolution_clock::now();
  if (VERBOSE > NONE){
    ROS_INFO("Callback execution time: %ld us",
              duration_cast<microseconds>(callEnd - callStart).count());
  }
}


//--------------------------------------------------------------------
// Utility functions
//--------------------------------------------------------------------

// finds the closest correct pointcloud pose to the given timestamp
void PtCdProcessing::findClosestPose(const ros::Time& timestamp) {

  PoseWithTimestamp closestPose;
  double minTimeDiff = std::numeric_limits<double>::max();

  for (const auto& pose : poseHistory) {
    double timeDiff = std::abs((pose.timestamp - timestamp).toSec());

    if (timeDiff < minTimeDiff) {
      minTimeDiff = timeDiff;
      closestPose = pose;
    }
  }

  if (minTimeDiff > 0.1 && VERBOSE > INFO) {
    ROS_WARN("No pose found close enough to the pointcloud timestamp");
  }

  drone_position = closestPose.position;
  drone_orientation = closestPose.orientation;
}


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
  pass.setFilterLimits(FLOOR_TRIM_HEIGHT-drone_position.z(), 2.0f);
  pass.filter(*filtered_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid.filter(*filtered_cloud);

  pcl::fromPCLPointCloud2(*filtered_cloud, *filtered_cloud_XYZ );

  // Publishing PCL filtered cloud
  sensor_msgs::PointCloud2 output_filtered;
  pcl_conversions::fromPCL(*filtered_cloud, output_filtered);
  filtered_pc_pub.publish(output_filtered);
}


// Transform the lines from the drone's frame to the world frame
void PtCdProcessing::drone2WorldLines(std::vector<segment>& drone_lines){

  // Convert the quaternion to a rotation matrix
  Eigen::Matrix3d rotation_matrix = drone_orientation.toRotationMatrix();

  std::vector<segment> valid_segments;

  for (segment& computed_line : drone_lines) {
    bool lineExists = false;
    bool line_inside = false;

    // Transform the segment in the world frame
    segment test_line;
    test_line.a = rotation_matrix * computed_line.a + drone_position;
    test_line.b = rotation_matrix * computed_line.b;
    test_line.t_min = computed_line.t_min;
    test_line.t_max = computed_line.t_max;

    // Check if the segment is above the ground
    Eigen::Vector3d test_line_p1 = test_line.t_min * test_line.b + test_line.a;
    Eigen::Vector3d test_line_p2 = test_line.t_max * test_line.b + test_line.a;

    if (test_line_p1.z() > FLOOR_TRIM_HEIGHT || test_line_p2.z() > FLOOR_TRIM_HEIGHT){

      test_line.radius = computed_line.radius;
      for (const Eigen::Vector3d& point : computed_line.points){
        test_line.points.push_back(rotation_matrix * point + drone_position);
      }

      valid_segments.push_back(test_line);
    }
  }

  drone_lines = valid_segments;
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
    return 0;
  } else {
    if (VERBOSE > INFO)
      ROS_WARN("Points form a dispersed structure");
    return 1;
  }
}


void PtCdProcessing::segIdentification(std::vector<segment>& drone_lines){

  for (const segment& test_line : drone_lines) {
    double max_dx = 2*sqrt(3)*leaf_size;
    double tolerance = 2*max_dx + 2*test_line.radius;

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

      if (((test_line_proj_p1-test_line_p1).norm() < tolerance) && 
          ((test_line_proj_p2-test_line_p2).norm() < tolerance) && 
          (test_line.radius == world_lines[i].radius)) {

        if (VERBOSE > NONE){
          ROS_INFO("Lines are close, checking for similarity");
        }

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

          if (VERBOSE > NONE){
            ROS_INFO("The 2 segments are similar");
          }

          addLine = false;

          world_lines[i].a = new_world_line_a;
          world_lines[i].b = new_world_line_b;

          std::vector<double> list_t = {test_line_proj_t1, test_line_proj_t2, existing_line_proj_t1, existing_line_proj_t2};
          world_lines[i].t_max = *std::max_element(list_t.begin(), list_t.end());
          world_lines[i].t_min = *std::min_element(list_t.begin(), list_t.end());
          world_lines[i].radius = test_line.radius;
          world_lines[i].points.insert(world_lines[i].points.end(), test_line.points.begin(), test_line.points.end());

          break;
        }
      } else if (VERBOSE > NONE){
        ROS_INFO("Lines are not close");
      }
    }
    if (addLine) {
      world_lines.push_back(test_line);
    }
  }
}

void PtCdProcessing::structureOutput(){

  // initiate variables
  struct_segm.clear();
  size_t numLines = world_lines.size();
  intersection_matrix.resize(numLines);
  for (size_t i = 0; i < numLines; ++i) {
      intersection_matrix[i].resize(numLines, std::make_tuple(-1.0, -1.0));
  }

  for (size_t i = 0; i < world_lines.size(); ++i) {
    const segment& existing_line_1 = world_lines[i];
    bool hasIntersection = false;

    for (size_t j = i + 1; j < world_lines.size(); ++j) {
      const segment& existing_line_2 = world_lines[j];

      double tolerance = 2*sqrt(3)*leaf_size + 2*(existing_line_1.radius + existing_line_2.radius)/2;

      Eigen::Vector3d existing_line_1_p1 = existing_line_1.t_min * existing_line_1.b + existing_line_1.a;
      Eigen::Vector3d existing_line_2_p1 = existing_line_2.t_min * existing_line_2.b + existing_line_2.a;

      Eigen::Vector3d cross_prod = existing_line_2.b.cross(existing_line_1.b);
      if (cross_prod.norm() < 1e-6) {
        if (VERBOSE > INFO)
          ROS_WARN("Lines are parallel, skipping intersection check");
        continue;
      }
      cross_prod.normalize();

      Eigen::Vector3d RHS = existing_line_2_p1 - existing_line_1_p1;
      Eigen::Matrix3d LHS;
      LHS << existing_line_1.b, -existing_line_2.b, cross_prod;

      Eigen::Vector3d solution = LHS.colPivHouseholderQr().solve(RHS);
      
      double dist_lines = abs(solution[2] * cross_prod.norm());


      if (((solution[0] >= existing_line_1.t_min && solution[0] <= existing_line_1.t_max) &&
          (solution[1] > existing_line_2.t_min && solution[1] < existing_line_2.t_max)) &&
          dist_lines < tolerance) {

        Eigen::Vector3d intersection_point = existing_line_1_p1 + solution[0] * existing_line_1.b;

        // Create new segments for each segment at the intersection point
        segment new_line_1 = existing_line_1;
        new_line_1.t_max = solution[0];
        struct_segm.push_back(new_line_1);

        segment new_line_1_part2 = existing_line_1;
        new_line_1_part2.t_min = solution[0];
        struct_segm.push_back(new_line_1_part2);

        segment new_line_2 = existing_line_2;
        new_line_2.t_max = solution[1];
        struct_segm.push_back(new_line_2);

        segment new_line_2_part2 = existing_line_2;
        new_line_2_part2.t_min = solution[1];
        struct_segm.push_back(new_line_2_part2);

        struct_joints.push_back(intersection_point);

        intersection_matrix[i][j] = std::make_tuple(solution[0], solution[1]);
        intersection_matrix[j][i] = std::make_tuple(solution[1], solution[0]);

        hasIntersection = true;
      }
    }
    if (!hasIntersection) {
    // If no intersection is found, add the original segment to the output
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
    

  // Loop through the computed lines and create a cylinder for each segment
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

      // Loop through the intersection matrix to create a sphere for each intersection
  for (size_t i = 0; i < intersection_matrix.size(); i++) {
    for (size_t j = i + 1; j < intersection_matrix[i].size(); j++) {
      double t1, t2;
      std::tie(t1, t2) = intersection_matrix[i][j];

      // Check if an intersection exists between line i and line j
      if (t1 != -1.0 && t2 != -1.0) {
          Eigen::Vector3d intersection_point = world_lines[i].a + t1 * world_lines[i].b;

          visualization_msgs::Marker sphere;
          sphere.header.frame_id = "mocap";
          sphere.header.stamp = ros::Time::now();
          sphere.ns = "intersections";
          sphere.id = i * intersection_matrix.size() + j;  // Unique ID for each sphere
          sphere.action = visualization_msgs::Marker::ADD;
          sphere.pose.orientation.w = 1.0;
          sphere.type = visualization_msgs::Marker::SPHERE;
          sphere.pose.position.x = intersection_point.x();
          sphere.pose.position.y = intersection_point.y();
          sphere.pose.position.z = intersection_point.z();

          double sphere_radius = 3/2*std::max(radius_sizes[0], radius_sizes[radius_sizes.size()-1]);
          sphere.scale.x = sphere_radius * 2.0;
          sphere.scale.y = sphere_radius * 2.0;
          sphere.scale.z = sphere_radius * 2.0;
          sphere.color.r = 0.0;
          sphere.color.g = 1.0;
          sphere.color.b = 0.0;
          sphere.color.a = 1.0;

          markers.markers.push_back(sphere);
      }
    }
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