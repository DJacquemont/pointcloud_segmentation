// This file is a direct adaptation of the hough3dlines.cpp file from the library hough-3d-lines 
// (libraries/hough-3d-lines/hough3dlines.cpp). 
// This adaptation allows for the use of the hough algorithm in a ROS node.

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/pca.h>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

/**
 * @brief Segment structure storing the line segment information
 * 
 */
struct segment {
  Eigen::Vector3d a, b;
  double t_min, t_max;
  double radius;
  std::vector<Eigen::Vector3d> points;
  int points_size;
  double pca_coeff;
  Eigen::Vector3d pca_eigenvalues;
};

/**
 * @brief Finds the projection of a point on a line
 * 
 * @param a anchor point of the line
 * @param b direction of the line
 * @param p point to project
 * @return Eigen::Vector3d projection of p on the line
 */
int find_t(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p, std::vector<double>& t_values, std::vector<double>& p_norm){

  if (b.x() == 0) {
    return 1;
  }

  // t is similar for x, y and z
  double t = (p.x() - a.x()) / b.x();

  if (t_values.empty()){
    t_values.push_back(t);
    p_norm.push_back((a + t * b).norm());

  } else {
    // Find the appropriate index to insert t while keeping the vector sorted
    auto it = std::upper_bound(t_values.begin(), t_values.end(), t);

    // Calculate the index at which t should be inserted
    int index = std::distance(t_values.begin(), it);

    // Insert t at the calculated index
    t_values.insert(it, t);

    // Calculate the norm of p corresponding to t and insert it at the same index
    p_norm.insert(p_norm.begin() + index, (a + t * b).norm());
  }
  return 0;
}

/**
 * @brief Finds the projection of a point on a line
 * 
 * @param a anchor point of the line
 * @param b direction of the line
 * @param p point to project
 * @return Eigen::Vector3d projection of p on the line
 */
Eigen::Vector3d find_proj(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p){

  Eigen::Vector3d p_A(a);
  Eigen::Vector3d p_B(a+b);
  Eigen::Vector3d p_proj = p_A + (p_B-p_A)*((p-p_A).transpose()*(p_B-p_A))/((p_B-p_A).transpose()*(p_B-p_A));

  return p_proj;
}


/**
 * @brief Computing the PCA of the segment's points
 * 
 * @param drone_seg segment in the drone's frame
 * @return Eigen::Vector3d Eigenvalues
 */
Eigen::Vector3d segPCA(const std::vector<Eigen::Vector3d>& points) {

  // Extract points from drone_seg and create a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (Eigen::Vector3d point : points) {
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
 * @brief orthogonal least squares fit with libeigen
 * 
 * @param pc point cloud
 * @param a anchor point of the line
 * @param b direction of the line
 * @return double largest eigenvalue of the scatter matrix
 */
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



/**
 * @brief Method computing 3d lines with the Hough transform
 * 
 * @param pc point cloud
 * @param computed_lines vector of segments
 * @param opt_dx discretization step
 * @param radius_sizes vector of radius sizes
 * @param opt_minvotes minimum number of votes for a line to be considered
 * @param opt_nlines maximum number of lines to be computed
 * @param VERBOSE verbosity level
 * @return int 0 if successful, 1 otherwise
 */
int hough3dlines(const pcl::PointCloud<pcl::PointXYZ>& pc, std::vector<segment>& computed_lines, 
                  const double opt_dx, const int granularity, const std::vector<double> radius_sizes, 
                  const int opt_minvotes, const int opt_nlines, const double min_pca_coeff, 
                  const double rad_2_leaf_ratio, const int VERBOSE){

  PointCloud X;
  Vector3d point;

  for (size_t currentPoint = 0; currentPoint < pc.points.size(); currentPoint++)
	{
    float x = pc.points[currentPoint].x;
    float y = pc.points[currentPoint].y;
    float z = pc.points[currentPoint].z;

    // Cleaning the PointCloud from NaNs & Inf
    if (!(isnan(x) || isinf(x)) && !(isnan(y) || isinf(y)) && !(isnan(z) || isinf(z))){
      point.x = x;
      point.y = y;
      point.z = z;

      X.points.push_back(point);
    }
  }

  // number of icosahedron subdivisions for direction discretization
  const int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

  // bounding box of point cloud
  Vector3d minP, maxP, minPshifted, maxPshifted;
  // diagonal length of point cloud
  double d;

  // center cloud and compute new bounding boxhough3dtransform
  X.getMinMax3D(&minP, &maxP);
  d = (maxP-minP).norm();
  if (d == 0.0) {
    ROS_WARN("All points in point cloud identical");
    return 1;
  }
  X.shiftToOrigin();
  X.getMinMax3D(&minPshifted, &maxPshifted);

  if (opt_dx >= d) {
    ROS_WARN("dx too large");
    return 1;
  }

  double num_x = floor(d / opt_dx + 0.5);
  double num_cells = num_x * num_x * num_directions[granularity];

  // first Hough transform
  Hough* hough;
  try {
    hough = new Hough(minPshifted, maxPshifted, opt_dx);
  } catch (const std::exception &e) {
    ROS_WARN("Cannot allocate memory for %.0f Hough cells"
            " (%.2f MB)", num_cells, 
            (double(num_cells) / 1000000.0) * sizeof(unsigned int));

    return 1;
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

    // find t of the line segment, and the corresponding norm of p
    std::vector<double> p_radius;
    std::vector<double> p_norm;
    std::vector<double> t_values;
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d a_eigen(a.x, a.y, a.z);
    Eigen::Vector3d b_eigen(b.x, b.y, b.z);
    
    for(std::vector<Vector3d>::iterator it = Y.points.begin(); it != Y.points.end(); it++){
      
      Vector3d point = *it + X.shift;
      Eigen::Vector3d point_eigen(point.x, point.y, point.z);
      Eigen::Vector3d p_proj = find_proj(a_eigen, b_eigen, point_eigen);

      p_radius.push_back((p_proj - point_eigen).norm());
      if (find_t(a_eigen, b_eigen, p_proj, t_values, p_norm)){
        // no solution for t
        return 1;
      }

      // saving points
      points.push_back(point_eigen);
    }

    // finds largest distance between 2 neighbors points on the line using p_norm
    double max_dist = 0;
    for (int i = 0; i < p_norm.size()-1; i++){
      double dist = std::abs(p_norm[i+1] - p_norm[i]);
      if (dist > max_dist){
        max_dist = dist;
      }
    }


    double radius = std::max(p_radius.front(), p_radius.back());
    double closest_radius = radius_sizes.front();
    double min_radius_diff = std::abs(radius - radius_sizes.front());
    double max_radius = std::abs(radius - radius_sizes.front());
    for (double r : radius_sizes) {
      double current_difference = std::abs(radius - r);
      if (current_difference < min_radius_diff) {
          min_radius_diff = current_difference;
          closest_radius = r;
      }
      if (r > max_radius) {
          max_radius = r;
      }
    }
    

    // add line to vector if conditions are met
    if (min_radius_diff < opt_dx && max_radius <= closest_radius && max_dist < 2*opt_dx){

      Eigen::Vector3d pca_eigenvalues = segPCA(points);
      double pca_coeff = pca_eigenvalues[0] / (pca_eigenvalues[0] + pca_eigenvalues[1] + pca_eigenvalues[2]);

      Eigen::Vector3d test_seg_p1 = t_values.front() * b_eigen + a_eigen;
      Eigen::Vector3d test_seg_p2 = t_values.back() * b_eigen + a_eigen;
      double length_seg = (test_seg_p2 - test_seg_p1).norm();
      int min_nb_points_seg = static_cast<int>(2.0*closest_radius*length_seg/(rad_2_leaf_ratio*2*opt_dx*2*opt_dx));

      if (pca_coeff > min_pca_coeff && points.size() > min_nb_points_seg){

        segment l;
        l.a = a_eigen;
        l.b = b_eigen;
        l.t_min = t_values.front();
        l.t_max = t_values.back();
        l.radius = closest_radius;      
        l.points = points;
        l.points_size = points.size();
        l.pca_coeff = pca_coeff;
        l.pca_eigenvalues = pca_eigenvalues;

        computed_lines.push_back(l);
      }
    }
    
    X.removePoints(Y);

  } while ((X.points.size() > 1) && 
           ((opt_nlines == 0) || (opt_nlines > nlines)) && ros::ok());

  // clean up
  delete hough;

  return 0;
}