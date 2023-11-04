#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

struct line {
  Eigen::Vector3d p1;
  Eigen::Vector3d p2;
  Eigen::Vector3d a;
  Eigen::Vector3d b;
  double t_min;
  double t_max;
  double radius;
  std::vector<Eigen::Vector3d> points;
};

void find_t(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p, std::vector<double>& t_value, std::vector<double>& p_norm){

  // t is similar for x, y and z
  double t = (p.x() - a.x()) / b.x();

  if (t_value.empty()){
    t_value.push_back(t);
    p_norm.push_back((a + t * b).norm());

  } else {
    // Find the appropriate index to insert t while keeping the vector sorted
    auto it = std::upper_bound(t_value.begin(), t_value.end(), t);

    // Calculate the index at which t should be inserted
    int index = std::distance(t_value.begin(), it);

    // Insert t at the calculated index
    t_value.insert(it, t);

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
int hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc, std::vector<line>& computed_lines){

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
  double opt_dx = 0.15;
  int opt_nlines = 4;
  int opt_minvotes = 20;
  int opt_verbose = 0;

  // number of icosahedron subdivisions for direction discretization
  int granularity = 4;
  int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

  // bounding box of point cloud
  Vector3d minP, maxP, minPshifted, maxPshifted;
  // diagonal length of point cloud
  double d;

  // center cloud and compute new bounding boxhough3dtransform
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
    hough = new Hough(minPshifted, maxPshifted, opt_dx);
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

    if (opt_verbose)
      ROS_INFO("npoints=%lu, a=(%f,%f,%f), b=(%f,%f,%f)",
                Y.points.size(), a.x, a.y, a.z, b.x, b.y, b.z);

    // find t of the line segment, and the corresponding norm of p
    std::vector<double> p_radius;
    std::vector<double> p_norm;
    std::vector<double> t_value;
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d a_eigen(a.x, a.y, a.z);
    Eigen::Vector3d b_eigen(b.x, b.y, b.z);
    
    for(std::vector<Vector3d>::iterator it = Y.points.begin(); it != Y.points.end(); it++){
      
      Vector3d point = *it + X.shift;
      Eigen::Vector3d point_eigen(point.x, point.y, point.z);
      Eigen::Vector3d p_proj = find_proj(a_eigen, b_eigen, point_eigen);
      p_radius.push_back((p_proj - point_eigen).norm());
      find_t(a_eigen, b_eigen, p_proj, t_value, p_norm);

      // saving points
      points.push_back(point_eigen);
      }

    std::sort(p_radius.begin(), p_radius.end());
    double radius = p_radius[round(9*p_radius.size()/10)];

    double maxDifference = std::abs(p_norm[1] - p_norm[0]);
    for (size_t i = 1; i < p_norm.size() - 1; ++i) {
        double difference = std::abs(p_norm[i + 1] - p_norm[i]);
        if (difference > maxDifference)
            maxDifference = difference;
    }

    // add line to vector
    if (radius > 0.05 && maxDifference < 0.2){

      Eigen::Vector3d p1 = a_eigen + t_value[0]*b_eigen;
      Eigen::Vector3d p2 = a_eigen + t_value[t_value.size()-1]*b_eigen;

      line l;
      l.p1 = p1;
      l.p2 = p2;
      l.a = a_eigen;
      l.b = b_eigen;
      l.t_min = t_value[0];
      l.t_max = t_value[t_value.size()-1];
      l.radius = radius;
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