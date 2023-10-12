#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <eigen3/Eigen/Dense>


using Eigen::MatrixXf;


// struct{
//   int a;
//   int b;
// } ejfbwiefbws; 


// Utility functions prototype
int hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc);
double orthogonal_LSQ(const PointCloud &pc, Vector3d* a, Vector3d* b);


// Class caring for the node's communication (Subscription & Publication)
class PtCdProcessing
{
  public:
  // Class initialisation
  PtCdProcessing()
  {
    tof_pc_sub = node.subscribe("/tof_pc", 1000, &PtCdProcessing::pointcloudCallback, this);
    marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 10);
  }

  // Callback function receiving ToF images from the Autopilot package
  // and publishing marker lines to vizualise the computed lines in rviz
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

  // Method computing 3d lines with the Hough transform
  int hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc);

  private:
  ros::NodeHandle node;
  ros::Subscriber tof_pc_sub;
  ros::Publisher marker_pub;
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
    // 3dLines.clear()

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    ROS_INFO("Pointcloud Received : w %d, h %d, is_dense %d", 
                    pcl_cloud.width, pcl_cloud.height, pcl_cloud.is_dense);

    if (hough3dlines(pcl_cloud)){
      ROS_INFO("ERROR - Unable to perform the Hough transform");
    }

    // // Creating visualisation markers
    // visualization_msgs::Marker line_list;
    // line_list.header.frame_id = "world";
    // line_list.header.stamp = ros::Time::now();
    // line_list.ns = "points_and_lines";
    // line_list.action = visualization_msgs::Marker::ADD;
    // line_list.pose.orientation.w = 1.0;
    // line_list.type = visualization_msgs::Marker::LINE_LIST;
    // line_list.scale.x = 0.1;
    // line_list.scale.y = 0.1;
    // line_list.color.r = 1.0;
    // line_list.color.a = 1.0;

    // // Test: displaying 3 lines in rviz
    // for (uint32_t i = 0; i < 3; ++i)
    // {
    //   geometry_msgs::Point p;
    //   p.x = 0;
    //   p.y = i;
    //   p.z = 1;

    //   line_list.points.push_back(p);
    //   p.z += 1.0;
    //   line_list.points.push_back(p);
    // }

    // marker_pub.publish(line_list);
  }





//--------------------------------------------------------------------
// utility functions
//--------------------------------------------------------------------

// orthogonal least squares fit with libeigen
double orthogonal_LSQ(const PointCloud &pc, Vector3d* a, Vector3d* b){
  // rc = largest eigenvalue
  double rc = 0.0;

  // anchor point is mean value
  *a = pc.meanValue();

  // copy points to libeigen matrix
  Eigen::MatrixXf points = Eigen::MatrixXf::Constant(pc.points.size(), 3, 0);
  for (int i = 0; i < points.rows(); i++) {
    points(i,0) = pc.points.at(i).x;
    points(i,1) = pc.points.at(i).y;
    points(i,2) = pc.points.at(i).z;
  }

  // compute scatter matrix ...
  MatrixXf centered = points.rowwise() - points.colwise().mean();
  MatrixXf scatter = (centered.adjoint() * centered);

  // ... and its eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(scatter);
  Eigen::MatrixXf eigvecs = eig.eigenvectors();

  // we need eigenvector to largest eigenvalue
  // libeigen yields it as LAST column
  b->x = eigvecs(0,2); b->y = eigvecs(1,2); b->z = eigvecs(2,2);
  rc = eig.eigenvalues()(2);

  return (rc);
}


int PtCdProcessing::hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc){

  // ------------------ TEST IN PROGRESS ------------------ //
  // Creating visualisation marker
  visualization_msgs::Marker line_list;
  // ------------------ TEST IN PROGRESS ------------------ //

  PointCloud X;
  Vector3d point;

  for (size_t currentPoint = 0; currentPoint < pc.points.size(); currentPoint++)
	{
    float x = pc.points[currentPoint].x;
    float y = pc.points[currentPoint].y;
    float z = pc.points[currentPoint].z;

    // Cleaning the PointCloud from NaNs & Inf
    if (!(isnan(x) || isinf(x)) && !(isnan(y) ||
             isinf(y)) && !(isnan(z) && isinf(z))){
      point.x = x;
      point.y = y;
      point.z = z;

      X.points.push_back(point);
    }
  }

  // default parameter values
  double opt_dx = 0;
  int opt_nlines = 4;
  int opt_minvotes = 30;
  int opt_verbose = 0;

  // number of icosahedron subdivisions for direction discretization
  int granularity = 4;
  int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

  // bounding box of point cloud
  Vector3d minP, maxP, minPshifted, maxPshifted;
  // diagonal length of point cloud
  double d;

  // center cloud and compute new bounding box
  X.getMinMax3D(&minP, &maxP);
  d = (maxP-minP).norm();
  if (d == 0.0) {
    ROS_INFO("ERROR - all points in point cloud identical");
    return 1;
  }
  X.shiftToOrigin();
  X.getMinMax3D(&minPshifted, &maxPshifted);

  // estimate size of Hough space
  if (opt_dx == 0.0) {
    opt_dx = d / 64.0;
  }
  else if (opt_dx >= d) {
    ROS_INFO("ERROR - dx too large");
    return 1;
  }
  double num_x = floor(d / opt_dx + 0.5);
  double num_cells = num_x * num_x * num_directions[granularity];
  if (opt_verbose) {
    ROS_INFO("x'y' value range is %f in %.0f steps of width dx=%f",
           d, num_x, opt_dx);
    ROS_INFO("Hough space has %.0f cells taking %.2f MB memory space",
           num_cells, num_cells * sizeof(unsigned int) / 1000000.0);
  }

  // first Hough transform
  Hough* hough;
  try {
    hough = new Hough(minPshifted, maxPshifted, opt_dx, granularity);
  } catch (const std::exception &e) {
    
    ROS_INFO("ERROR - cannot allocate memory for %.0f Hough cells"
            " (%.2f MB)", num_cells, 
            (double(num_cells) / 1000000.0) * sizeof(unsigned int));
    return 2;
  }
  hough->add(X);
  
  // iterative Hough transform
  // (Algorithm 1 in IPOL paper)
  PointCloud Y;	// points close to line
  double rc;
  unsigned int nvotes;
  int nlines = 0;

  do {

    Vector3d a; // anchor point of line
    Vector3d b; // direction of line

    hough->subtract(Y); // do it here to save one call

    nvotes = hough->getLine(&a, &b);
    if (opt_verbose > 1) {
      Vector3d p = a + X.shift;
      ROS_INFO( "Highest number of Hough votes is %i for the following line:", nvotes);
      ROS_INFO( "a=(%f,%f,%f), b=(%f,%f,%f)", p.x, p.y, p.z, b.x, b.y, b.z);
    }

    X.pointsCloseToLine(a, b, opt_dx, &Y);

    rc = orthogonal_LSQ(Y, &a, &b);
    if (rc==0.0) break;

    X.pointsCloseToLine(a, b, opt_dx, &Y);
    nvotes = Y.points.size();
    if (nvotes < (unsigned int)opt_minvotes) break;

    rc = orthogonal_LSQ(Y, &a, &b);
    if (rc==0.0) break;

    a = a + X.shift;

    nlines++;

    ROS_INFO("npoints=%lu, a=(%f,%f,%f), b=(%f,%f,%f)",
              Y.points.size(), a.x, a.y, a.z, b.x, b.y, b.z);

    // ------------------ TEST IN PROGRESS ------------------ //

    line_list.header.frame_id = "world";
    line_list.header.stamp = ros::Time::now();
    line_list.ns = "points_and_lines";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.1;
    line_list.scale.z = 0.1;
    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    int t = 3;

    geometry_msgs::Point p1;
    p1.x = a.x;
    p1.y = a.y;
    p1.z = a.z;

    geometry_msgs::Point p2;
    p2.x = a.x+t*b.x;
    p2.y = a.y+t*b.y;
    p2.z = a.z+t*b.z;

    line_list.points.push_back(p1);
    line_list.points.push_back(p2);

    // ------------------ TEST IN PROGRESS ------------------ //

    X.removePoints(Y);

  } while ((X.points.size() > 1) && 
           ((opt_nlines == 0) || (opt_nlines > nlines)) && ros::ok());

  marker_pub.publish(line_list);

  // clean up
  delete hough;

  return 0;
}