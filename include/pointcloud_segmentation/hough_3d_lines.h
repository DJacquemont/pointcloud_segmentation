#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

const int opt_nlines = 5;

struct segment {
  // Eigen::Vector3d p1, p2;
  Eigen::Vector3d a, b;
  double t_min, t_max;
  double radius;
  std::vector<Eigen::Vector3d> points;
  double pca_coeff;
  int num_pca;
};

void find_t(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p, std::vector<double>& t_values, std::vector<double>& p_norm){

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
}

Eigen::Vector3d find_proj(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p){

  Eigen::Vector3d p_A(a);
  Eigen::Vector3d p_B(a+b);
  Eigen::Vector3d p_proj = p_A + (p_B-p_A)*((p-p_A).transpose()*(p_B-p_A))/((p_B-p_A).transpose()*(p_B-p_A));

  return p_proj;
}

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

// Method computing 3d lines with the Hough transform
int hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc, std::vector<segment>& computed_lines, 
                  double opt_dx, std::vector<double> radius_sizes, int opt_minvotes, int opt_nlines, int VERBOSE){

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

  // number of icosahedron subdivisions for direction discretization
  const int granularity = 4;
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
      find_t(a_eigen, b_eigen, p_proj, t_values, p_norm);

      // saving points
      points.push_back(point_eigen);
    }

    double radius = std::max(p_radius[0], p_radius[p_radius.size()-1]);

    double closest_radius = radius_sizes[0];
    double min_radius_diff = std::abs(radius - radius_sizes[0]);
    for (double r : radius_sizes) {
      double current_difference = std::abs(radius - r);
      if (current_difference < min_radius_diff) {
          min_radius_diff = current_difference;
          closest_radius = r;
      }
    }
    

    // check integrity of the segment
    bool seg_integrity = true;
    int div_number = std::max(3, static_cast<int>(std::floor(t_values.size() / opt_minvotes)));
    double div_length = (t_values.back() - t_values.front()) / div_number;
    int div_minpoints = static_cast<int>(std::ceil((3.0/4.0) * t_values.size() / div_number));

    for (int i = 0; i < div_number; i++) {
      double start_range = t_values.front() + i * div_length;
      double end_range = (i == div_number - 1) ? t_values.back() : start_range + div_length;
      int count = 0;
      
      for (const double& t : t_values) {
        if (t >= start_range && t <= end_range) {
          ++count;
        }
      }

      if (count < div_minpoints) {
        seg_integrity = false;
        break;
      }
    }

    // add line to vector
    if (min_radius_diff < opt_dx && seg_integrity){

      Eigen::Vector3d p1 = a_eigen + t_values.front() *b_eigen;
      Eigen::Vector3d p2 = a_eigen + t_values.back() *b_eigen;

      segment l;
      l.a = a_eigen;
      l.b = b_eigen;
      l.t_min = t_values.front();
      l.t_max = t_values.back();
      l.radius = closest_radius;      
      l.points = points;

      computed_lines.push_back(l);
    }
    
    X.removePoints(Y);

  } while ((X.points.size() > 1) && 
           ((opt_nlines == 0) || (opt_nlines > nlines)) && ros::ok());

  // clean up
  delete hough;

  return 0;
}