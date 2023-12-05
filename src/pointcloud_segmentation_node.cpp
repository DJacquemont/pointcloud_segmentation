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
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

#include "hough_3d_lines.h"

using namespace std::chrono;

enum verbose {NONE, INFO, WARN};

const double WINDOW_FILTERING_SIZE = 3.0;


/**
 * @brief Class processing the points and storing the output segments
 * 
 * @details This class subscribes to the point cloud topic published by the Autopilot package, 
 * and processes the point cloud using the Hough transform.
 */
class PtCdProcessing
{
  struct Intersection {
    int seg_idx;
    Eigen::Vector3d solution;
  };

  struct SharedData {
    sensor_msgs::PointCloud2 latestMsg;
    bool newDataAvailable = false;
  };

  struct Pose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  };

public:
  PtCdProcessing() : tfListener(tfBuffer)
  {
    running = true;
    processingThread = std::thread(&PtCdProcessing::processData, this);

    setParams();

    // Initialize subscribers and publishers
    tof_pc_sub = node.subscribe("/tof_pc", 1, &PtCdProcessing::pointcloudCallback, this);
    marker_pub = node.advertise<visualization_msgs::MarkerArray>("markers", 0);
    filtered_pc_pub = node.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 0);
    hough_pc_pub = node.advertise<sensor_msgs::PointCloud2>("hough_pointcloud", 0);
  }

  ~PtCdProcessing() {
    running = false;
    dataCondition.notify_one();
    if (processingThread.joinable()) {
        processingThread.join();
    }
  }

  void setParams();

  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

  void processData();

  int closestDronePose(const ros::Time& timestamp);

  void cloudFiltering(const pcl::PCLPointCloud2::Ptr& input_cloud, 
                      pcl::PointCloud<pcl::PointXYZ>::Ptr& output_filtered_cloud_XYZ);

  void drone2WorldSeg(std::vector<segment>& drone_segments);

  Eigen::Vector3d segPCA(const segment&);

  bool integrityCheck(const segment& drone_seg);

  void segFiltering(std::vector<segment>& drone_segments);

  void checkConnections(const int& idx_check_seg);

  // void intersectionSearch();

  void visualization();

  void clearMarkers();

private:
  ros::NodeHandle node;

  // Mutex and condition variable for thread safety
  bool running = false;
  SharedData sharedData;
  std::mutex dataMutex;
  std::condition_variable dataCondition;
  std::thread processingThread;

  // Subscribers and publishers
  ros::Subscriber tof_pc_sub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pc_pub;
  ros::Publisher hough_pc_pub;
  
  // TF2
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  Pose drone;

  // Structure
  std::vector<segment> world_segments;
  std::vector<segment> struct_segm;
  std::vector<std::vector<std::tuple<double, double>>> intersection_matrix;

  // Parameters
  enum verbose verbose_level;
  double floor_trim_height;
  double min_pca_coeff;
  double min_lr_coeff;
  int opt_minvotes;
  int opt_nlines;
  std::vector<double> radius_sizes;
  double leaf_size;
  double opt_dx;
  double rad_2_leaf_ratio;
};


/**
 * @brief Callback function receiving ToF images from the Autopilot package
 * 
 * @param msg Message containing the point cloud
 * @return ** void 
 */
void PtCdProcessing::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(dataMutex);
  sharedData.latestMsg = *msg;
  sharedData.newDataAvailable = true;
  dataCondition.notify_one();
}


/**
 * @brief Set the parameters
 * 
 * @return ** void
 */
void PtCdProcessing::setParams() {
  // Get parameters from the parameter server
  int verbose_level_int;
  if (!this->node.getParam("/verbose_level", verbose_level_int)) {
    ROS_ERROR("Failed to get param 'verbose_level'");
    verbose_level_int = 0;
  }
  verbose_level = static_cast<verbose>(verbose_level_int);

  if (!this->node.getParam("/floor_trim_height", floor_trim_height)) {
    ROS_ERROR("Failed to get param 'floor_trim_height'");
    floor_trim_height = 0.3;
  }

  if (!this->node.getParam("/min_pca_coeff", min_pca_coeff)) {
    ROS_ERROR("Failed to get param 'min_pca_coeff'");
    min_pca_coeff = 0.95;
  }

  if (!this->node.getParam("/min_lr_coeff", min_lr_coeff)) {
    ROS_ERROR("Failed to get param 'min_lr_coeff'");
    min_pca_coeff = 0.1;
  }

  if (!this->node.getParam("/rad_2_leaf_ratio", rad_2_leaf_ratio)) {
    ROS_ERROR("Failed to get param 'rad_2_leaf_ratio'");
    rad_2_leaf_ratio = 1.5;
  }

  if (!this->node.getParam("/opt_minvotes", opt_minvotes)) {
    ROS_ERROR("Failed to get param 'opt_minvotes'");
    opt_minvotes = 10;
  }

  if (!this->node.getParam("/opt_nlines", opt_nlines)) {
    ROS_ERROR("Failed to get param 'opt_nlines'");
    opt_nlines = 10;
  }

  XmlRpc::XmlRpcValue radius_sizes_tmp;
  radius_sizes.clear();
  if (this->node.getParam("/radius_sizes", radius_sizes_tmp)) {
    for (int32_t i = 0; i < radius_sizes_tmp.size(); ++i) {
      radius_sizes.push_back(static_cast<double>(radius_sizes_tmp[i]));
    }
  } else {
    ROS_ERROR("Failed to get param 'radius_sizes'");
    radius_sizes.push_back(static_cast<double>(0.05));
  }

  leaf_size = std::min(radius_sizes[0], radius_sizes[radius_sizes.size()-1]) / rad_2_leaf_ratio;
  opt_dx = sqrt(3) * leaf_size;

  ROS_INFO("Configuration:");
  ROS_INFO("  verbose_level: %d", verbose_level);
  ROS_INFO("  floor_trim_height: %f", floor_trim_height);
  ROS_INFO("  min_pca_coeff: %f", min_pca_coeff);
  ROS_INFO("  min_lr_coeff: %f", min_lr_coeff);
  ROS_INFO("  opt_minvotes: %d", opt_minvotes);
  ROS_INFO("  opt_nlines: %d", opt_nlines);
  ROS_INFO("  radius_sizes: %f, %f, %f", 
            radius_sizes[0], radius_sizes[1], radius_sizes[2]);
  ROS_INFO("  leaf_size: %f", leaf_size);
  ROS_INFO("  opt_dx: %f", opt_dx);
}



/**
 * @brief Process the latest point cloud
 * 
 * @return ** void
 */
void PtCdProcessing::processData() {
  while (running) {
    std::unique_lock<std::mutex> lock(dataMutex);
    dataCondition.wait(lock, [this] { return sharedData.newDataAvailable || !running; });

    // Process the latest message
    auto latestMsgCopy = sharedData.latestMsg;
    sharedData.newDataAvailable = false;

    lock.unlock();

    auto callStart = high_resolution_clock::now();

    // Find the closest pose in time to the point cloud's timestamp
    if (closestDronePose(latestMsgCopy.header.stamp)){
      ROS_WARN("Unable to find the closest pose in time to the point cloud's timestamp");
      return;
    }

    // Filtering pointcloud
    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_XYZ( new pcl::PointCloud<pcl::PointXYZ> );
    pcl_conversions::toPCL(latestMsgCopy, *cloud);
    cloudFiltering(cloud, filtered_cloud_XYZ);

    // segment extraction with the Hough transform
    std::vector<segment> drone_segments;
    // const double opt_dx = sqrt(3)*leaf_size; // optimal discretization step
    if (hough3dlines(*filtered_cloud_XYZ, drone_segments, opt_dx, radius_sizes, opt_minvotes, opt_nlines, verbose_level) 
        && verbose_level > INFO){
      ROS_WARN("Unable to perform the Hough transform");
    }

    // Transform the segments from the drone's frame to the world frame
    drone2WorldSeg(drone_segments);

    // Filter the segments that already exist in the world_segments
    segFiltering(drone_segments);

    // Identify intersections between segments, and ready the data for output
    // intersectionSearch();

    visualization();

    // Print all the segments
    if (verbose_level > INFO){
      ROS_INFO("Number of segments: %lu", world_segments.size());
      for (size_t i = 0; i < world_segments.size(); ++i) {
        ROS_INFO("Segment %lu: a = (%f, %f, %f), t_min = %f, t_max = %f", 
                  i, world_segments[i].a.x(), world_segments[i].a.y(), world_segments[i].a.z(), 
                  world_segments[i].t_min, world_segments[i].t_max);
      }
    }

    // Print the matrix of intersections
    if (verbose_level > INFO){
      ROS_INFO("Intersection matrix:");
      for (size_t i = 0; i < intersection_matrix.size(); ++i) {
        for (size_t j = 0; j < intersection_matrix.size(); ++j) {
          ROS_INFO("intersection_matrix[%lu][%lu] = (%f, %f)", i, j, 
                    std::get<0>(intersection_matrix[i][j]), std::get<1>(intersection_matrix[i][j]));
        }
      }
    }

    auto callEnd = high_resolution_clock::now();
    if (verbose_level > NONE){
      ROS_INFO("Callback execution time: %ld us",
                duration_cast<microseconds>(callEnd - callStart).count());
    }
  }
}


/**
 * @brief Finds the closest pose in time to the point cloud's timestamp
 * 
 * @param timestamp Timestamp of the point cloud
 * @return int 0 if successful, 1 otherwise
 */
int PtCdProcessing::closestDronePose(const ros::Time& timestamp) {
  try {
    geometry_msgs::TransformStamped transformStamped;
    transformStamped = tfBuffer.lookupTransform("mocap", "world", timestamp, ros::Duration(1.0));
    drone.position = Eigen::Vector3d(transformStamped.transform.translation.x,
                                      transformStamped.transform.translation.y,
                                      transformStamped.transform.translation.z);
    drone.orientation = Eigen::Quaterniond(transformStamped.transform.rotation.w,
                                            transformStamped.transform.rotation.x,
                                            transformStamped.transform.rotation.y,
                                            transformStamped.transform.rotation.z);
  }
  catch (tf2::TransformException &ex) {
    if (verbose_level > INFO){
      ROS_WARN("%s", ex.what());
    }
    return 1;
  }
  return 0;
}


// Filtering pointcloud
/**
 * @brief Filtering pointcloud using PCL thresholding and voxel grid
 * 
 * @param cloud cloud to be filtered
 * @param filtered_cloud_XYZ filtered cloud
 */
void PtCdProcessing::cloudFiltering(const pcl::PCLPointCloud2::Ptr& input_cloud, 
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_filtered_cloud_XYZ){

  pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2());

  // Apply PassThrough filter for 'x', 'y', and 'z' fields
  pcl::PassThrough<pcl::PCLPointCloud2> pass;

  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(0.0f, WINDOW_FILTERING_SIZE/2);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-WINDOW_FILTERING_SIZE/2, WINDOW_FILTERING_SIZE/2);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(-WINDOW_FILTERING_SIZE/2, WINDOW_FILTERING_SIZE/2);
  pass.filter(*filtered_cloud);

  // Applying VoxelGrid filter
  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(filtered_cloud);
  voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid.filter(*filtered_cloud);

  pcl::fromPCLPointCloud2(*filtered_cloud, *output_filtered_cloud_XYZ );

  // Publishing PCL filtered cloud
  sensor_msgs::PointCloud2 output_filtered;
  pcl_conversions::fromPCL(*filtered_cloud, output_filtered);
  filtered_pc_pub.publish(output_filtered);
}


/**
 * @brief Transform the segments from the drone's frame to the world frame
 * 
 * @param drone_segments segments in the drone's frame 
 */
void PtCdProcessing::drone2WorldSeg(std::vector<segment>& drone_segments){

  // Convert the quaternion to a rotation matrix
  Eigen::Matrix3d rotation_matrix = drone.orientation.toRotationMatrix();

  std::vector<segment> valid_segments;

  for (segment& drone_seg : drone_segments) {

    // Transform the segment in the world frame
    segment test_seg;
    test_seg.a = rotation_matrix * drone_seg.a + drone.position;
    test_seg.b = rotation_matrix * drone_seg.b;
    test_seg.t_min = drone_seg.t_min;
    test_seg.t_max = drone_seg.t_max;

    // Check if the segment is above the ground
    Eigen::Vector3d test_seg_p1 = test_seg.t_min * test_seg.b + test_seg.a;
    Eigen::Vector3d test_seg_p2 = test_seg.t_max * test_seg.b + test_seg.a;

    if (test_seg_p1.z() > floor_trim_height || test_seg_p2.z() > floor_trim_height){

      test_seg.t_values = drone_seg.t_values;
      test_seg.radius = drone_seg.radius;
      test_seg.points_size = drone_seg.points_size;
      test_seg.pca_coeff = drone_seg.pca_coeff;
      test_seg.pca_eigenvalues = drone_seg.pca_eigenvalues;

      for (const Eigen::Vector3d& point : drone_seg.points){
        test_seg.points.push_back(rotation_matrix * point + drone.position);
      }

      valid_segments.push_back(test_seg);
    }
  }

  drone_segments = valid_segments;
}


/**
 * @brief Computing the PCA of the segment's points
 * 
 * @param drone_seg segment in the drone's frame
 * @return Eigen::Vector3d Eigenvalues
 */
Eigen::Vector3d PtCdProcessing::segPCA(const segment& drone_seg) {

  // Extract points from drone_seg and create a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& point : drone_seg.points) {
    cloud->points.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
  }

  // Performing PCA using PCL
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cloud);

  // Eigenvalues are in descending order
  Eigen::Vector3d eigenvalues = pca.getEigenValues().cast<double>();
  
  return eigenvalues;
}


/**
 * @brief Check the integrity of the segment
 * 
 * @param drone_seg segment in the drone's frame
 * @return true if the segment is valid
 * @return false if the segment is invalid
 */
bool PtCdProcessing::integrityCheck(const segment& drone_seg){

  // check integrity of the segment
  bool seg_integrity = true;

  Eigen::Vector3d drone_seg_p1 = drone_seg.t_min * drone_seg.b + drone_seg.a;
  Eigen::Vector3d drone_seg_p2 = drone_seg.t_max * drone_seg.b + drone_seg.a;
  int div_number = static_cast<int>(std::max(3.0, (drone_seg_p2-drone_seg_p1).norm()) / (opt_dx*opt_minvotes));
  // int div_number = static_cast<int>(std::floor(3*drone_seg.t_values.size() / opt_minvotes));
  double div_length_t = (drone_seg.t_values.back() - drone_seg.t_values.front()) / div_number;
  int div_minpoints = static_cast<int>(std::floor((4.0/6.0) * drone_seg.t_values.size() / div_number));

  for (int i = 0; i < div_number; i++) {
    double start_range = drone_seg.t_values.front() + i * div_length_t;
    double end_range = (i == div_number - 1) ? drone_seg.t_values.back() : start_range + div_length_t;
    int count = 0;
    
    for (const double& t : drone_seg.t_values) {
      if (t >= start_range && t <= end_range) {
        ++count;
      }
    }

    if (count < div_minpoints) {
      return false;
    }
  }
  return true;
}



/**
 * @brief Function filtering already existing or invalid segments, fusing similar segments, 
 * and adding new segments to the world_segments
 * 
 * @param drone_segments Segments in the drone's frame
 */
void PtCdProcessing::segFiltering(std::vector<segment>& drone_segments){

  // std::cout << "world_segments before" << std::endl;
  // for (size_t i = 0; i < world_segments.size(); ++i) {
  //   std::cout << "world_segments[" << i << "].a = " << world_segments[i].a[0] << " " << world_segments[i].a[1] << " " << world_segments[i].a[2] << std::endl;
  //   std::cout << "world_segments[" << i << "].b = " << world_segments[i].b[0] << " " << world_segments[i].b[1] << " " << world_segments[i].b[2] << std::endl;
  //   std::cout << "world_segments[" << i << "].radius = " << world_segments[i].radius << std::endl;
  // }
  // std::cout << std::endl;

  std::vector<int> world_segments_2_check;

  for (const segment& drone_seg : drone_segments) {

    bool add_seg = true;

    Eigen::Vector3d test_seg_p1 = drone_seg.t_min * drone_seg.b + drone_seg.a;
    Eigen::Vector3d test_seg_p2 = drone_seg.t_max * drone_seg.b + drone_seg.a;
    
    Eigen::Vector3d eigenvalues = segPCA(drone_seg);
    double new_pca_coeff = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

    double length_seg = (test_seg_p2 - test_seg_p1).norm();
    int min_nb_points_seg = static_cast<int>(2.0*drone_seg.radius*length_seg/(rad_2_leaf_ratio*2*opt_dx*2*opt_dx));

    bool integrity = integrityCheck(drone_seg);

    if (new_pca_coeff < min_pca_coeff || drone_seg.points.size() < min_nb_points_seg || !integrity) {
      // if (new_pca_coeff < min_pca_coeff || drone_seg.points.size() < min_nb_points_seg) {
      add_seg = false;
      continue;
    }

    for (size_t i = 0; i < world_segments.size(); ++i) {
      Eigen::Vector3d world_seg_p1 = world_segments[i].t_min * world_segments[i].b + world_segments[i].a;
      Eigen::Vector3d world_seg_p2 = world_segments[i].t_max * world_segments[i].b + world_segments[i].a;

      Eigen::Vector3d test_seg_proj_p1 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p1);
      Eigen::Vector3d test_seg_proj_p2 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p2);

      double epsilon = drone_seg.radius + world_segments[i].radius + 2*(2*opt_dx);

      if (((test_seg_proj_p1-test_seg_p1).norm() < epsilon) && 
          ((test_seg_proj_p2-test_seg_p2).norm() < epsilon) && 
          (drone_seg.radius == world_segments[i].radius)){

        if (verbose_level > NONE){
          ROS_INFO("Segments are close, checking for similarity");
        }


        double lr_coeff = drone_seg.points.size()/(world_segments[i].points.size()+drone_seg.points.size());
        lr_coeff = (lr_coeff < min_lr_coeff) ? min_lr_coeff : lr_coeff;

        // double learning_rate = new_pca_coeff/(world_segments[i].pca_coeff*world_segments[i].num_pca+new_pca_coeff);
        // double learning_rate = (new_pca_coeff*drone_seg.points.size())/
        //             (world_segments[i].pca_coeff*world_segments[i].points.size()+new_pca_coeff*drone_seg.points.size());
        double learning_rate = (new_pca_coeff*lr_coeff)/
                                (world_segments[i].pca_coeff*(1-lr_coeff) + new_pca_coeff*lr_coeff);

        Eigen::Vector3d new_world_seg_a = test_seg_proj_p1 + learning_rate*(test_seg_p1 - test_seg_proj_p1);
        Eigen::Vector3d new_world_seg_b = (test_seg_proj_p2-test_seg_proj_p1) +
                                          learning_rate*((test_seg_p2-test_seg_proj_p2)-(test_seg_p1-test_seg_proj_p1));

        Eigen::Vector3d test_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p1);
        Eigen::Vector3d test_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p2);
        Eigen::Vector3d world_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p1);
        Eigen::Vector3d world_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p2);

        double test_seg_proj_t1 = (test_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double test_seg_proj_t2 = (test_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double world_seg_proj_t1 = (world_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        double world_seg_proj_t2 = (world_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        
        if (!((std::min(test_seg_proj_t1, test_seg_proj_t2)>std::max(world_seg_proj_t1, world_seg_proj_t2)) ||
              (std::max(test_seg_proj_t1, test_seg_proj_t2)<std::min(world_seg_proj_t1, world_seg_proj_t2)))){

          if (verbose_level > NONE){
            ROS_INFO("The 2 segments are similar");
          }

          add_seg = false;

          segment modif_seg = world_segments[i];

          modif_seg.a = new_world_seg_a;
          modif_seg.b = new_world_seg_b;

          


          // modifer la colonne intersection de world line studied
          for (size_t j = 0; j < intersection_matrix.size(); ++j) {
            double t1, t2;
            std::tie(t1, t2) = intersection_matrix[i][j];
            if (t1 != -1.0 && t2 != -1.0) {
              assert(((t1 == -1) && (t2 != -1)) || ((t1 != -1) && (t2 == -1)));

              if (i > j){
                double old_t_i = t1;
                Eigen::Vector3d old_p_i = old_t_i * world_segments[i].b + world_segments[i].a;
                Eigen::Vector3d new_p_i = find_proj(new_world_seg_a, new_world_seg_b, old_p_i);
                double new_t = (new_p_i.x() - new_world_seg_a.x()) / new_world_seg_b.x();




                intersection_matrix[i][j] = std::make_tuple(new_t, std::get<1>(intersection_matrix[i][j]));
              } else if (i < j){
                double old_t = t2;
                Eigen::Vector3d old_p = old_t * world_segments[j].b + world_segments[j].a;
                Eigen::Vector3d new_p = find_proj(new_world_seg_a, new_world_seg_b, old_p);
                double new_t = (new_p.x() - new_world_seg_a.x()) / new_world_seg_b.x();
                intersection_matrix[j][i] = std::make_tuple(std::get<0>(intersection_matrix[i][j]), new_t);
              }
            }
          }

          

          // cherche a savoir si t_min ou _max est a une intersection. Si c'est le cas alors le nouveau t ne pourra pas depasser la valeur de l'intersection
          // regarder colonne intersection de world line studied, et en deduire les t min et max
          // si le t min ou max est present, alors verifier que l'intersection est bien la bonne enregardant le world segment correspondant

          // check if t_min or t_max is at an intersection. If so, then the new t cannot exceed the value of the intersection

          std::vector<double> list_t = {test_seg_proj_t1, test_seg_proj_t2, world_seg_proj_t1, world_seg_proj_t2};
          modif_seg.t_max = *std::max_element(list_t.begin(), list_t.end());
          modif_seg.t_min = *std::min_element(list_t.begin(), list_t.end());
          for (size_t j = 0; j < intersection_matrix.size(); ++j) {
            if (std::get<0>(intersection_matrix[i][j]) != -1.0){
              if (std::get<0>(intersection_matrix[i][j]) == world_segments[i].t_min){
                modif_seg.t_min = std::get<0>(intersection_matrix[i][j]);
              }
              if (std::get<0>(intersection_matrix[i][j]) == world_segments[i].t_max){
                modif_seg.t_max = std::get<0>(intersection_matrix[i][j]);
              }
            }
          }
          modif_seg.radius = drone_seg.radius;
          modif_seg.points_size = modif_seg.points_size + drone_seg.points_size;
          modif_seg.points.insert(modif_seg.points.end(), drone_seg.points.begin(), drone_seg.points.end());
          // modif_seg.pca_coeff = (modif_seg.pca_coeff*modif_seg.num_pca+new_pca_coeff)/(modif_seg.num_pca+1);
          // modif_seg.pca_coeff = (modif_seg.pca_coeff*world_segments[i].points.size()+
          //                                               new_pca_coeff*drone_seg.points.size())/modif_seg.points.size();
          modif_seg.pca_coeff = modif_seg.pca_coeff*(1-lr_coeff) + new_pca_coeff*lr_coeff;
          // modif_seg.pca_coeff = (modif_seg.pca_coeff*new_pca_coeff)/
          //                   (modif_seg.pca_coeff*world_segments[i].points.size()+new_pca_coeff*drone_seg.points.size());
          // modif_seg.pca_coeff = (modif_seg.pca_coeff*new_pca_coeff)/(modif_seg.pca_coeff+new_pca_coeff);
          
          // modif_seg.pca_eigenvalues = (modif_seg.pca_eigenvalues*world_segments[i].points.size()+
          //                                               eigenvalues*drone_seg.points.size())/modif_seg.points.size();
          modif_seg.pca_eigenvalues = modif_seg.pca_eigenvalues*(1-lr_coeff) + eigenvalues*lr_coeff;

          world_segments[i] = modif_seg;

          world_segments_2_check.push_back(i);
        }
      } else if (verbose_level > NONE){
        ROS_INFO("Segments are not close");
      }
    }

    if (add_seg) {

      segment added = drone_seg; 
      added.pca_coeff = new_pca_coeff;
      added.pca_eigenvalues = eigenvalues;
      world_segments.push_back(added);

      // std::cout << "drone segment added " << std::endl;
      // std::cout << "drone_segment.a = " << added.a[0] << " " << added.a[1] << " " << added.a[2] << std::endl;
      // std::cout << "drone_segment.b = " << added.b[0] << " " << added.b[1] << " " << added.b[2] << std::endl;
      // std::cout << "drone_segment.radius = " << added.radius << std::endl << std::endl;

      world_segments_2_check.push_back(world_segments.size()-1);
    }
  }

  // world segments after
  // std::cout << "world_segments after" << std::endl;
  // for (size_t i = 0; i < world_segments.size(); ++i) {
  //   std::cout << "world_segments[" << i << "].a = " << world_segments[i].a[0] << " " << world_segments[i].a[1] << " " 
  //             << world_segments[i].a[2] << "t_min = " << world_segments[i].t_min << "t_max = " << world_segments[i].t_max << std::endl;
  // }
  // std::cout <<  std::endl;

  for (int idx : world_segments_2_check){
    checkConnections(idx);
  }

}

/**
 * @brief Search for intersections between segments, separates segments with intersections, and ready the data for 
 * output
 * 
 */
void PtCdProcessing::checkConnections(const int& idx_check_seg){

  std::vector<Intersection> intersections;

  segment* check_seg = &world_segments[idx_check_seg];

  bool has_intersection = false;

  Eigen::Vector3d check_seg_p1 = check_seg->t_min * check_seg->b + check_seg->a;

  for (size_t i = 0; i < world_segments.size(); ++i) {

    if (i == idx_check_seg){
      continue;
    }

    segment world_seg = world_segments[i];
    Eigen::Vector3d world_seg_p1 = world_seg.t_min * world_seg.b + world_seg.a;

    Eigen::Vector3d cross_prod = world_seg.b.cross(check_seg->b);
    if (cross_prod.norm() < 1e-2) {
      if (verbose_level > INFO){
        ROS_WARN("Segments are parallel, skipping intersection check");
      }
      continue;
    }
    cross_prod.normalize();

    Eigen::Vector3d RHS = world_seg_p1 - check_seg_p1;
    Eigen::Matrix3d LHS;
    LHS << check_seg->b, -world_seg.b, cross_prod;

    Eigen::Vector3d sol_intersection = LHS.colPivHouseholderQr().solve(RHS);
    double dist_intersection = abs(sol_intersection[2]);

    double epsilon = 2*opt_dx + check_seg->radius + world_seg.radius;
    if (((sol_intersection[0] > check_seg->t_min && sol_intersection[0] < check_seg->t_max) &&
        (sol_intersection[1] > world_seg.t_min && sol_intersection[1] < world_seg.t_max)) &&
        dist_intersection < epsilon) {
      
      Intersection intersection;
      intersection.seg_idx = i;
      intersection.solution = sol_intersection;

      // Find the correct insertion point
      auto it = std::lower_bound(intersections.begin(), intersections.end(), intersection, 
        [](const Intersection& a, const Intersection& b) {
          return a.solution[0] < b.solution[0];
        });

      // Insert the new intersection at the found position
      intersections.insert(it, intersection);

      has_intersection = true;
    }
  }

  // Resize the intersection_matrix to accommodate new segments
  int nb_new_segments = (intersections.size() > 0) ? intersections.size()-1 : 0;
  intersection_matrix.resize(world_segments.size() + nb_new_segments);
  for (auto& row : intersection_matrix) {
    row.resize(world_segments.size() + nb_new_segments, std::make_tuple(-1.0, -1.0));
  }

  // std::cout<<"world_segments.size() = "<<world_segments.size()<<std::endl;
  // std::cout<<"intersection_matrix.size() = "<<intersection_matrix.size()<<std::endl;


  if (has_intersection){

    segment original_seg = *check_seg;
    double total_length = original_seg.t_max - original_seg.t_min;

    for (size_t i = 0; i<intersections.size(); i++){

      Intersection intersection = intersections[i];
      if (intersection.solution[0] > original_seg.t_min  && intersection.solution[0] < original_seg.t_max){
        if (i == 0){
          check_seg->t_max = original_seg.t_min + intersection.solution[0];
          check_seg->pca_coeff = original_seg.pca_coeff * (check_seg->t_max - original_seg.t_min) / total_length;
          check_seg->pca_eigenvalues[0] = original_seg.pca_eigenvalues[0] 
                                          * (check_seg->t_max - original_seg.t_min) / total_length;
        }
        else{
          segment new_seg = original_seg;
          new_seg.t_min = original_seg.t_min + intersections[i-1].solution[0];
          new_seg.t_max = original_seg.t_min + intersection.solution[0];
          new_seg.pca_coeff = original_seg.pca_coeff * (new_seg.t_max - new_seg.t_min) / total_length;
          new_seg.pca_eigenvalues[0] = original_seg.pca_eigenvalues[0] 
                                          * (new_seg.t_max - new_seg.t_min) / total_length;

          world_segments.push_back(new_seg);
        }

        assert(idx_check_seg+i == intersection.seg_idx);
        if (idx_check_seg+i > intersection.seg_idx){
          intersection_matrix[idx_check_seg+i][intersection.seg_idx] = std::make_tuple(
          original_seg.t_min + intersection.solution[0], 
          world_segments[intersection.seg_idx].t_min + intersection.solution[1]);
        }
        else {
          intersection_matrix[intersection.seg_idx][idx_check_seg+i] = std::make_tuple(
          world_segments[intersection.seg_idx].t_min + intersection.solution[1], 
          original_seg.t_min + intersection.solution[0]);
        }

      }
    }
  }
}

// /**
//  * @brief Function filtering already existing or invalid segments, fusing similar segments, 
//  * and adding new segments to the world_segments
//  * 
//  * @param drone_segments Segments in the drone's frame
//  */
// void PtCdProcessing::segFiltering(std::vector<segment>& drone_segments){

//   std::vector<segment> new_world_segments = world_segments;

//   for (const segment& drone_seg : drone_segments) {

//     bool add_seg = true;

//     Eigen::Vector3d test_seg_p1 = drone_seg.t_min * drone_seg.b + drone_seg.a;
//     Eigen::Vector3d test_seg_p2 = drone_seg.t_max * drone_seg.b + drone_seg.a;
    
//     Eigen::Vector3d eigenvalues = segPCA(drone_seg);
//     double new_pca_coeff = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

//     double length_seg = (test_seg_p2 - test_seg_p1).norm();
//     int min_nb_points_seg = static_cast<int>(2.0*drone_seg.radius*length_seg/(rad_2_leaf_ratio*2*opt_dx*2*opt_dx));

//     bool integrity = integrityCheck(drone_seg);

//     if (new_pca_coeff < min_pca_coeff || drone_seg.points.size() < min_nb_points_seg || !integrity) {
//       // if (new_pca_coeff < min_pca_coeff || drone_seg.points.size() < min_nb_points_seg) {
//       add_seg = false;
//       continue;
//     }

//     for (size_t i = 0; i < world_segments.size(); ++i) {
//       Eigen::Vector3d world_seg_p1 = world_segments[i].t_min * world_segments[i].b + world_segments[i].a;
//       Eigen::Vector3d world_seg_p2 = world_segments[i].t_max * world_segments[i].b + world_segments[i].a;

//       Eigen::Vector3d test_seg_proj_p1 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p1);
//       Eigen::Vector3d test_seg_proj_p2 = find_proj(world_segments[i].a, world_segments[i].b, test_seg_p2);

//       double epsilon = drone_seg.radius + world_segments[i].radius + 2*(2*opt_dx);

//       if (((test_seg_proj_p1-test_seg_p1).norm() < epsilon) && 
//           ((test_seg_proj_p2-test_seg_p2).norm() < epsilon) && 
//           (drone_seg.radius == world_segments[i].radius)){

//         if (verbose_level > NONE){
//           ROS_INFO("Segments are close, checking for similarity");
//         }

//         double lr_coeff = drone_seg.points.size()/(world_segments[i].points.size()+drone_seg.points.size());
//         lr_coeff = (lr_coeff < min_lr_coeff) ? min_lr_coeff : lr_coeff;

//         // double learning_rate = new_pca_coeff/(world_segments[i].pca_coeff*world_segments[i].num_pca+new_pca_coeff);
//         // double learning_rate = (new_pca_coeff*drone_seg.points.size())/
//         //             (world_segments[i].pca_coeff*world_segments[i].points.size()+new_pca_coeff*drone_seg.points.size());
//         double learning_rate = (new_pca_coeff*lr_coeff)/
//                                 (world_segments[i].pca_coeff*(1-lr_coeff) + new_pca_coeff*lr_coeff);

//         Eigen::Vector3d new_world_seg_a = test_seg_proj_p1 + learning_rate*(test_seg_p1 - test_seg_proj_p1);
//         Eigen::Vector3d new_world_seg_b = (test_seg_proj_p2-test_seg_proj_p1) +
//                                           learning_rate*((test_seg_p2-test_seg_proj_p2)-(test_seg_p1-test_seg_proj_p1));

//         Eigen::Vector3d test_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p1);
//         Eigen::Vector3d test_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, test_seg_p2);
//         Eigen::Vector3d world_seg_proj_p1 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p1);
//         Eigen::Vector3d world_seg_proj_p2 = find_proj(new_world_seg_a, new_world_seg_b, world_seg_p2);

//         double test_seg_proj_t1 = (test_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
//         double test_seg_proj_t2 = (test_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
//         double world_seg_proj_t1 = (world_seg_proj_p1.x() - new_world_seg_a.x()) / new_world_seg_b.x();
//         double world_seg_proj_t2 = (world_seg_proj_p2.x() - new_world_seg_a.x()) / new_world_seg_b.x();
        
//         if (!((std::min(test_seg_proj_t1, test_seg_proj_t2)>std::max(world_seg_proj_t1, world_seg_proj_t2)) ||
//               (std::max(test_seg_proj_t1, test_seg_proj_t2)<std::min(world_seg_proj_t1, world_seg_proj_t2)))){

//           if (verbose_level > NONE){
//             ROS_INFO("The 2 segments are similar");
//           }

//           add_seg = false;

//           segment modif_seg = world_segments[i];

//           modif_seg.a = new_world_seg_a;
//           modif_seg.b = new_world_seg_b;

//           std::vector<double> list_t = {test_seg_proj_t1, test_seg_proj_t2, world_seg_proj_t1, world_seg_proj_t2};
//           modif_seg.t_max = *std::max_element(list_t.begin(), list_t.end());
//           modif_seg.t_min = *std::min_element(list_t.begin(), list_t.end());
//           modif_seg.radius = drone_seg.radius;
//           modif_seg.points_size = modif_seg.points_size + drone_seg.points_size;
//           modif_seg.points.insert(modif_seg.points.end(), drone_seg.points.begin(), drone_seg.points.end());
//           // modif_seg.pca_coeff = (modif_seg.pca_coeff*modif_seg.num_pca+new_pca_coeff)/(modif_seg.num_pca+1);
//           // modif_seg.pca_coeff = (modif_seg.pca_coeff*world_segments[i].points.size()+
//           //                                               new_pca_coeff*drone_seg.points.size())/modif_seg.points.size();
//           modif_seg.pca_coeff = modif_seg.pca_coeff*(1-lr_coeff) + new_pca_coeff*lr_coeff;
//           // modif_seg.pca_coeff = (modif_seg.pca_coeff*new_pca_coeff)/
//           //                   (modif_seg.pca_coeff*world_segments[i].points.size()+new_pca_coeff*drone_seg.points.size());
//           // modif_seg.pca_coeff = (modif_seg.pca_coeff*new_pca_coeff)/(modif_seg.pca_coeff+new_pca_coeff);
          
//           // modif_seg.pca_eigenvalues = (modif_seg.pca_eigenvalues*world_segments[i].points.size()+
//           //                                               eigenvalues*drone_seg.points.size())/modif_seg.points.size();
//           modif_seg.pca_eigenvalues = modif_seg.pca_eigenvalues*(1-lr_coeff) + eigenvalues*lr_coeff;

//           world_segments[i] = modif_seg;
//         }
//       } else if (verbose_level > NONE){
//         ROS_INFO("Segments are not close");
//       }
//     }

//     if (add_seg) {

//       integrityCheck(drone_seg);

//       segment added = drone_seg; 
//       added.pca_coeff = new_pca_coeff;
//       added.pca_eigenvalues = eigenvalues;
//       world_segments.push_back(added);
//     }
//   }
// }

// /**
//  * @brief Search for intersections between segments, separates segments with intersections, and ready the data for 
//  * output
//  * 
//  * @return ** void
//  */
// void PtCdProcessing::intersectionSearch(){

//   // initiate variables
//   struct_segm.clear();
//   std::vector<std::vector<std::tuple<double, double>>> empty_intersection_matrix;
//   intersection_matrix.swap(empty_intersection_matrix);

//   size_t nb_segments = world_segments.size();
//   intersection_matrix.resize(nb_segments);
//   for (size_t i = 0; i < nb_segments; ++i) {
//       intersection_matrix[i].resize(nb_segments, std::make_tuple(-1.0, -1.0));
//   }

//   for (size_t i = 0; i < world_segments.size(); ++i) {
//     const segment world_seg_1 = world_segments[i];
//     bool hasIntersection = false;

//     for (size_t j = i + 1; j < world_segments.size(); ++j) {
//       const segment world_seg_2 = world_segments[j];

//       double epsilon = 2*opt_dx + world_seg_1.radius + world_seg_2.radius;

//       Eigen::Vector3d world_seg_1_p1 = world_seg_1.t_min * world_seg_1.b + world_seg_1.a;
//       Eigen::Vector3d world_seg_2_p1 = world_seg_2.t_min * world_seg_2.b + world_seg_2.a;

//       Eigen::Vector3d cross_prod = world_seg_2.b.cross(world_seg_1.b);
//       if (cross_prod.norm() < 1e-4) {
//         if (verbose_level > INFO)
//           ROS_WARN("Segments are parallel, skipping intersection check");
//         continue;
//       }
//       cross_prod.normalize();

//       Eigen::Vector3d RHS = world_seg_2_p1 - world_seg_1_p1;
//       Eigen::Matrix3d LHS;
//       LHS << world_seg_1.b, -world_seg_2.b, cross_prod;

//       Eigen::Vector3d solution = LHS.colPivHouseholderQr().solve(RHS);
      
//       double dist_intersection = abs(solution[2]);

//       double tolerance_coef = 0.05;
//       double tol_seg_1 = tolerance_coef * (world_seg_1.t_max - world_seg_1.t_min);
//       double tol_seg_2 = tolerance_coef * (world_seg_2.t_max - world_seg_2.t_min);
//       if (((solution[0] >= (world_seg_1.t_min-tol_seg_1) && solution[0] <= (world_seg_1.t_max+tol_seg_1)) &&
//           (solution[1] >= (world_seg_2.t_min-tol_seg_2) && solution[1] <= (world_seg_2.t_max+tol_seg_2))) &&
//           dist_intersection < epsilon) {

//         // Create new segments for each segment at the intersection point
//         segment new_seg_1 = world_seg_1;
//         new_seg_1.t_max = solution[0];
//         struct_segm.push_back(new_seg_1);

//         segment new_seg_1_part2 = world_seg_1;
//         new_seg_1_part2.t_min = solution[0];
//         struct_segm.push_back(new_seg_1_part2);

//         segment new_seg_2 = world_seg_2;
//         new_seg_2.t_max = solution[1];
//         struct_segm.push_back(new_seg_2);

//         segment new_seg_2_part2 = world_seg_2;
//         new_seg_2_part2.t_min = solution[1];
//         struct_segm.push_back(new_seg_2_part2);

//         intersection_matrix[i][j] = std::make_tuple(world_seg_1.t_min + solution[0], world_seg_2.t_min + solution[1]);

//         hasIntersection = true;
//       }
//     }
//     if (!hasIntersection) {
//     // If no intersection is found, add the original segment to the output
//       struct_segm.push_back(world_seg_1);
//     }
//   }
// }


/**
 * @brief Publish the points used, and the computed segments as cylinders in RViz
 * 
 * @return ** void
 */
void PtCdProcessing::visualization() {

  clearMarkers();

  // Create a pointcloud to hold the points used for Hough
  pcl::PointCloud<pcl::PointXYZ> pc_out;

  // Create a marker array to hold the cylinders
  visualization_msgs::MarkerArray markers;

  int id_counter = 0;
    
  std::vector<segment> output_segments = world_segments;

  // Loop through the computed segments and create a cylinder for each segment
  for (size_t i = 0; i < output_segments.size(); i++) {
    for (const Eigen::Vector3d& point : output_segments[i].points){
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
    cylinder.id = id_counter++;
    cylinder.action = visualization_msgs::Marker::ADD;
    cylinder.pose.orientation.w = 1.0;
    cylinder.type = visualization_msgs::Marker::CYLINDER;

    // Set the cylinder's position (midpoint between p1 and p2)
    Eigen::Vector3d segment_p1 = output_segments[i].t_min * output_segments[i].b + output_segments[i].a;
    Eigen::Vector3d segment_p2 = output_segments[i].t_max * output_segments[i].b + output_segments[i].a;

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
    double cylinder_height = (segment_p2 - segment_p1).norm();
    double cylinder_radius = output_segments[i].radius;
    cylinder.scale.x = cylinder_radius * 2.0;
    cylinder.scale.y = cylinder_radius * 2.0;
    cylinder.scale.z = cylinder_height;
    cylinder.color.r = 1.0;
    cylinder.color.g = 0.0;
    cylinder.color.b = 0.0;
    cylinder.color.a = 0.5;

    // Add the cylinder marker to the marker array
    markers.markers.push_back(cylinder);


    // Create a text marker for displaying the segment number
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "mocap";
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "segment_text";
    text_marker.id = id_counter++;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.orientation.w = 1.0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.pose.position.x = (segment_p1.x() + segment_p2.x()) / 2.0;
    text_marker.pose.position.y = (segment_p1.y() + segment_p2.y()) / 2.0;
    text_marker.pose.position.z = (segment_p1.z() + segment_p2.z()) / 2.0 ;
    text_marker.text = std::to_string(i);
    text_marker.scale.z = 0.1;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    // Add the text marker to the marker array
    markers.markers.push_back(text_marker);
  }

  for (size_t i = 0; i < world_segments.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      double t1, t2;
      std::tie(t1, t2) = intersection_matrix[i][j];

      // Check if an intersection exists between drone_seg i and j
      if (t1 != -1.0 && t2 != -1.0) {
          
          Eigen::Vector3d intersection_point = world_segments[i].a + t1 * world_segments[i].b;

          visualization_msgs::Marker sphere;
          sphere.header.frame_id = "mocap";
          sphere.header.stamp = ros::Time::now();
          sphere.ns = "intersections";
          sphere.id = id_counter++;
          sphere.action = visualization_msgs::Marker::ADD;
          sphere.pose.orientation.w = 1.0;
          sphere.type = visualization_msgs::Marker::SPHERE;
          sphere.pose.position.x = intersection_point.x();
          sphere.pose.position.y = intersection_point.y();
          sphere.pose.position.z = intersection_point.z();

          double sphere_radius = 3/2*std::max(radius_sizes.front(), radius_sizes.back());
          sphere.scale.x = sphere_radius * 2.0;
          sphere.scale.y = sphere_radius * 2.0;
          sphere.scale.z = sphere_radius * 2.0;
          sphere.color.r = 0.0;
          sphere.color.g = 1.0;
          sphere.color.b = 0.0;
          sphere.color.a = 1.0;

          markers.markers.push_back(sphere);


          // Text marker for displaying segment numbers
          visualization_msgs::Marker text_marker;
          text_marker.header.frame_id = "mocap";
          text_marker.header.stamp = ros::Time::now();
          text_marker.ns = "intersection_text";
          text_marker.id = id_counter++;
          text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
          text_marker.action = visualization_msgs::Marker::ADD;
          text_marker.pose.position.x = intersection_point.x();
          text_marker.pose.position.y = intersection_point.y();
          text_marker.pose.position.z = intersection_point.z() + 0.1;
          text_marker.scale.z = 0.1;
          text_marker.color.a = 1.0;
          text_marker.color.r = 1.0;
          text_marker.color.g = 1.0;
          text_marker.color.b = 1.0;
          text_marker.text = "Intersection: " + std::to_string(i) + " & " + std::to_string(j);

          markers.markers.push_back(text_marker);
      }
    }
  }

  // Publishing points used Hough
  sensor_msgs::PointCloud2 output_hough;
  pcl::PCLPointCloud2::Ptr pts_hough (new pcl::PCLPointCloud2);
  pcl::toPCLPointCloud2(pc_out, *pts_hough);
  pcl_conversions::fromPCL(*pts_hough, output_hough);
  output_hough.header.frame_id = "mocap";
  hough_pc_pub.publish(output_hough);

  // Publish the marker array containing the cylinders
  marker_pub.publish(markers);
}

// Clear the markers in RViz
void PtCdProcessing::clearMarkers() {
  visualization_msgs::Marker clear_marker;
  clear_marker.action = visualization_msgs::Marker::DELETEALL;
  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(clear_marker);
  marker_pub.publish(marker_array);
}



int main(int argc, char *argv[])
{
  ROS_INFO("Pointcloud segmentation node starting");

  ros::init(argc, argv, "pointcloud_seg");

  initHoughSpace();

  PtCdProcessing SAPobject;

  ros::spin();

  return 0;
}