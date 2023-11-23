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

enum verbose {NONE, INFO, WARN};


const int VERBOSE = WARN;
const size_t MAX_POSE_HISTORY_SIZE = 50;
const double FLOOR_TRIM_HEIGHT = 0.3;
const double RADIUS_2_LENGTH_RATIO = 25;
const double LEARNING_RATE = 0.05;
const std::vector<double> RADIUS_SIZES = {0.1}; // 0.25, 0.07, 0.05, 0.1, 0.03
const double LEAF_SIZE = std::min(RADIUS_SIZES[0], RADIUS_SIZES[RADIUS_SIZES.size()-1]);
const double OPT_DX = sqrt(3)*LEAF_SIZE;
// const int OPT_MINVOTES = 1/2*(2*std::max(RADIUS_SIZES[0], RADIUS_SIZES[RADIUS_SIZES.size()-1]) * (1+RADIUS_2_LENGTH_RATIO))/(LEAF_SIZE*LEAF_SIZE);
const int OPT_MINVOTES = 20;


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
    marker_pub = node.advertise<visualization_msgs::MarkerArray>("markers", 1);
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

  // Transform the segments from the drone's frame to the world frame
  void drone2WorldSeg(std::vector<segment>& drone_segments);

  // Check if the point cloud is linear
  int cloudLinearShapeCheck(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  // Identify intersections between segments
  void segIdentification(std::vector<segment>& drone_segments);

  // Output the structure
  void structureOutput();

  // Publish the points used, and the computed segments as cylinders in RViz
  void visualization();

private:
  ros::NodeHandle node;

  ros::Subscriber tof_pc_sub;
  ros::Subscriber pose_sub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pc_pub;
  ros::Publisher hough_pc_pub;
  
  std::vector<PoseWithTimestamp> poseHistory;
  Eigen::Vector3d drone_position;
  Eigen::Quaterniond drone_orientation;

  std::vector<segment> world_segments;
  std::vector<segment> struct_segm;
  std::vector<Eigen::Vector3d> struct_joints;
  std::vector<std::vector<std::tuple<double, double>>> intersection_matrix;

  // Parameters
  // const double LEAF_SIZE = std::min(RADIUS_SIZES[0], RADIUS_SIZES[RADIUS_SIZES.size()-1]);
};

//--------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------

int main(int argc, char *argv[])
{
  ROS_INFO("Pointcloud semgmentation node starting");
  ROS_INFO("OPT_MINVOTES: %i", OPT_MINVOTES);

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
  std::vector<segment> drone_segments;
  // const double OPT_DX = sqrt(3)*LEAF_SIZE; // optimal discretization step
  if (hough3dlines(*filtered_cloud_XYZ, drone_segments, OPT_DX, RADIUS_SIZES, OPT_MINVOTES, VERBOSE) && VERBOSE > INFO){
    ROS_WARN("Unable to perform the Hough transform");
  }

  if (VERBOSE > NONE){
    ROS_INFO("Number of segments detected: %ld", drone_segments.size());
  }

  // Transform the segments from the drone's frame to the world frame
  drone2WorldSeg(drone_segments);

  // Filter the segments that already exist in the world_segments
  segIdentification(drone_segments);

  // Identify intersections between segments, and ready the data for output
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

  PoseWithTimestamp closest_pose;
  double min_time_diff = std::numeric_limits<double>::max();

  for (const auto& pose : poseHistory) {
    double time_diff = std::abs((pose.timestamp - timestamp).toSec());

    if (time_diff < min_time_diff) {
      min_time_diff = time_diff;
      closest_pose = pose;
    }
  }

  if (min_time_diff > 0.1 && VERBOSE > INFO) {
    ROS_WARN("No pose found close enough to the pointcloud timestamp");
  }

  drone_position = closest_pose.position;
  drone_orientation = closest_pose.orientation;
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
  voxel_grid.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
  voxel_grid.filter(*filtered_cloud);

  pcl::fromPCLPointCloud2(*filtered_cloud, *filtered_cloud_XYZ );

  // Publishing PCL filtered cloud
  sensor_msgs::PointCloud2 output_filtered;
  pcl_conversions::fromPCL(*filtered_cloud, output_filtered);
  filtered_pc_pub.publish(output_filtered);
}


// Transform the segments from the drone's frame to the world frame
void PtCdProcessing::drone2WorldSeg(std::vector<segment>& drone_segments){

  // Convert the quaternion to a rotation matrix
  Eigen::Matrix3d rotation_matrix = drone_orientation.toRotationMatrix();

  std::vector<segment> valid_segments;

  for (segment& seg : drone_segments) {

    // Transform the segment in the world frame
    segment test_seg;
    test_seg.a = rotation_matrix * seg.a + drone_position;
    test_seg.b = rotation_matrix * seg.b;
    test_seg.t_min = seg.t_min;
    test_seg.t_max = seg.t_max;

    // Check if the segment is above the ground
    Eigen::Vector3d test_seg_p1 = test_seg.t_min * test_seg.b + test_seg.a;
    Eigen::Vector3d test_seg_p2 = test_seg.t_max * test_seg.b + test_seg.a;

    if (test_seg_p1.z() > FLOOR_TRIM_HEIGHT || test_seg_p2.z() > FLOOR_TRIM_HEIGHT){

      test_seg.radius = seg.radius;
      for (const Eigen::Vector3d& point : seg.points){
        test_seg.points.push_back(rotation_matrix * point + drone_position);
      }

      valid_segments.push_back(test_seg);
    }
  }

  drone_segments = valid_segments;
}


int PtCdProcessing::cloudLinearShapeCheck(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

  // Performing PCA using PCL
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cloud);

  // Eigenvalues are in descending order
  Eigen::Vector3d eigenvalues = pca.getEigenValues().cast<double>();

  // Assess the distribution of points based on eigenvalues
  if (eigenvalues[0] > RADIUS_2_LENGTH_RATIO * (eigenvalues[1] + eigenvalues[2])/2) {
    return 0;
  } else {
    if (VERBOSE > INFO)
      ROS_WARN("Points form a dispersed structure");
    return 1;
  }
}


void PtCdProcessing::segIdentification(std::vector<segment>& drone_segments){

  for (const segment& seg : drone_segments) {

    Eigen::Vector3d test_seg_p1 = seg.t_min * seg.b + seg.a;
    Eigen::Vector3d test_seg_p2 = seg.t_max * seg.b + seg.a;

    double epsilon = 2*OPT_DX + 2*(test_seg_p2-test_seg_p1).norm()/RADIUS_2_LENGTH_RATIO;
    bool add_seg = true;

    // Extract points from seg and create a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : seg.points) {
      cloud->points.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }

    // Check the shape of the point cloud
    if (cloudLinearShapeCheck(cloud)) {
      add_seg = false;
      continue;
    }

    for (size_t i = 0; i < world_segments.size(); ++i) {
      Eigen::Vector3d world_seg_p1 = world_segments[i].t_min * world_segments[i].b + world_segments[i].a;
      Eigen::Vector3d world_seg_p2 = world_segments[i].t_max * world_segments[i].b + world_segments[i].a;

      Eigen::Vector3d test_seg_proj_p1 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p1);
      Eigen::Vector3d test_seg_proj_p2 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p2);

      if (((test_seg_proj_p1-test_seg_p1).norm() < epsilon) && 
          ((test_seg_proj_p2-test_seg_p2).norm() < epsilon) && 
          (seg.radius == world_segments[i].radius)){

        if (VERBOSE > NONE){
          ROS_INFO("Segments are close, checking for similarity");
        }

        Eigen::Vector3d new_world_seg_a = test_seg_proj_p1*(1-LEARNING_RATE) + test_seg_p1*LEARNING_RATE;
        Eigen::Vector3d new_world_seg_b = ((test_seg_proj_p2*(1-LEARNING_RATE) + test_seg_p2*LEARNING_RATE) - 
                                          (test_seg_proj_p1*(1-LEARNING_RATE) + test_seg_p1*LEARNING_RATE)).normalized();

        Eigen::Vector3d test_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p1);
        Eigen::Vector3d test_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p2);
        Eigen::Vector3d world_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p1);
        Eigen::Vector3d world_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p2);

        double test_seg_proj_t1 = (test_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double test_seg_proj_t2 = (test_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double world_seg_proj_t1 = (world_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double world_seg_proj_t2 = (world_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        
        if (!(std::min(test_seg_proj_t1, test_seg_proj_t2)>std::max(world_seg_proj_t1, world_seg_proj_t2)) ||
              (std::max(test_seg_proj_t1, test_seg_proj_t2)<std::min(world_seg_proj_t1, world_seg_proj_t2))){

          if (VERBOSE > NONE){
            ROS_INFO("The 2 segments are similar");
          }

          add_seg = false;

          world_segments[i].a = new_world_seg_a;
          world_segments[i].b = new_world_seg_b;

          std::vector<double> list_t = {test_seg_proj_t1, test_seg_proj_t2, world_seg_proj_t1, world_seg_proj_t2};
          world_segments[i].t_max = *std::max_element(list_t.begin(), list_t.end());
          world_segments[i].t_min = *std::min_element(list_t.begin(), list_t.end());
          world_segments[i].radius = seg.radius;
          world_segments[i].points.insert(world_segments[i].points.end(), seg.points.begin(), seg.points.end());

          break;
        }
      } else if (VERBOSE > NONE){
        ROS_INFO("Segments are not close");
      }
    }
    if (add_seg) {
      world_segments.push_back(seg);
    }
  }
}

void PtCdProcessing::structureOutput(){

  // initiate variables
  struct_segm.clear();

  size_t nb_segments = world_segments.size();
  intersection_matrix.resize(nb_segments);
  for (size_t i = 0; i < nb_segments; ++i) {
      intersection_matrix[i].resize(nb_segments, std::make_tuple(-1.0, -1.0));
  }

  for (size_t i = 0; i < world_segments.size(); ++i) {
    const segment& world_seg_1 = world_segments[i];
    bool hasIntersection = false;

    for (size_t j = i + 1; j < world_segments.size(); ++j) {
      const segment& world_seg_2 = world_segments[j];

      double epsilon = 2*sqrt(3)*LEAF_SIZE + 2*(world_seg_1.radius + world_seg_2.radius)/2;

      Eigen::Vector3d world_seg_1_p1 = world_seg_1.t_min * world_seg_1.b + world_seg_1.a;
      Eigen::Vector3d world_seg_2_p1 = world_seg_2.t_min * world_seg_2.b + world_seg_2.a;

      Eigen::Vector3d cross_prod = world_seg_2.b.cross(world_seg_1.b);
      if (cross_prod.norm() < 1e-6) {
        if (VERBOSE > INFO)
          ROS_WARN("Segments are parallel, skipping intersection check");
        continue;
      }
      cross_prod.normalize();

      Eigen::Vector3d RHS = world_seg_2_p1 - world_seg_1_p1;
      Eigen::Matrix3d LHS;
      LHS << world_seg_1.b, -world_seg_2.b, cross_prod;

      Eigen::Vector3d solution = LHS.colPivHouseholderQr().solve(RHS);
      
      double dist_intersection = abs(solution[2] * cross_prod.norm());


      if (((solution[0] >= world_seg_1.t_min && solution[0] <= world_seg_1.t_max) &&
          (solution[1] > world_seg_2.t_min && solution[1] < world_seg_2.t_max)) &&
          dist_intersection < epsilon) {

        Eigen::Vector3d intersection_point = world_seg_1_p1 + solution[0] * world_seg_1.b;

        // Create new segments for each segment at the intersection point
        segment new_seg_1 = world_seg_1;
        new_seg_1.t_max = solution[0];
        struct_segm.push_back(new_seg_1);

        segment new_seg_1_part2 = world_seg_1;
        new_seg_1_part2.t_min = solution[0];
        struct_segm.push_back(new_seg_1_part2);

        segment new_seg_2 = world_seg_2;
        new_seg_2.t_max = solution[1];
        struct_segm.push_back(new_seg_2);

        segment new_seg_2_part2 = world_seg_2;
        new_seg_2_part2.t_min = solution[1];
        struct_segm.push_back(new_seg_2_part2);

        struct_joints.push_back(intersection_point);

        intersection_matrix[i][j] = std::make_tuple(solution[0], solution[1]);
        intersection_matrix[j][i] = std::make_tuple(solution[1], solution[0]);

        hasIntersection = true;
      }
    }
    if (!hasIntersection) {
    // If no intersection is found, add the original segment to the output
      struct_segm.push_back(world_seg_1);
    }
  }
}

// Publish the points used, and the computed segments as cylinders in RViz
void PtCdProcessing::visualization() {
  // Create a pointcloud to hold the points used for Hough
  sensor_msgs::PointCloud2 output_hough;
  pcl::PointCloud<pcl::PointXYZ> pc_out;
  // Create a marker array to hold the cylinders
  visualization_msgs::MarkerArray markers;
    

  // Loop through the computed segments and create a cylinder for each segment
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

      // Check if an intersection exists between seg i and j
      if (t1 != -1.0 && t2 != -1.0) {
          Eigen::Vector3d intersection_point = world_segments[i].a + t1 * world_segments[i].b;

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

          double sphere_radius = 3/2*std::max(RADIUS_SIZES[0], RADIUS_SIZES[RADIUS_SIZES.size()-1]);
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
  marker_pub.publish(markers);
}