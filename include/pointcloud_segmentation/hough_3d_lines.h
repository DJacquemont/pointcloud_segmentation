#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

struct line {
  Vector3d p1;
  Vector3d p2;
  Vector3d a;
  Vector3d b;
  double radius;
};

double find_t(Vector3d a, Vector3d b, Vector3d p, bool mode){
  double t;
  double t1 = (p.x - a.x) / b.x;
  double t2 = (p.y - a.y) / b.y;
  double t3 = (p.z - a.z) / b.z;

  if (mode)
    t = std::max(std::max(t1, t2), t3);
  else
    t = std::min(std::min(t1, t2), t3);
  return t;
}

Vector3d find_proj(Vector3d a, Vector3d b, Vector3d p, std::vector<double> &point_dist){
  Vector3d p_A(a.x, a.y, a.z);
  Vector3d p_B(a.x+b.x, a.y+b.y, a.z+b.z);
  Vector3d p_proj = p_A + (((p-p_A)*(p_B-p_A))*(p_B-p_A))/((p_B-p_A)*(p_B-p_A));

  point_dist.push_back((p_proj - p).norm());

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
int hough3dlines(pcl::PointCloud<pcl::PointXYZ>& pc, std::vector<line>& computed_lines, pcl::PointCloud<pcl::PointXYZ> &pc_out){

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

    
    // find t_min and t_max length of the line segment
    std::vector<double> point_dist;
    Vector3d p_proj = find_proj(a, b, Y.points[0] + X.shift, point_dist);
    double t_min = find_t(a, b, p_proj, 0);
    double t_max = find_t(a, b, p_proj, 1);

    for(std::vector<Vector3d>::iterator it = Y.points.begin(); it != Y.points.end(); it++){
      Vector3d p_proj = find_proj(a, b, *it + X.shift, point_dist);
      double t_min_test = find_t(a, b, p_proj, 0);
      if (t_min > t_min_test)
        t_min = t_min_test;

      double t_max_test = find_t(a, b, p_proj, 1);
      if (t_max < t_max_test)
        t_max = t_max_test;

      // Rviz markers points
      Vector3d p_hough = *it + X.shift;
      pcl::PointXYZ p_pcl;
      p_pcl.x = p_hough.x;
      p_pcl.y = p_hough.y;
      p_pcl.z = p_hough.z;
      pc_out.points.push_back(p_pcl);
      }

    // find radius
    double radius = double(std::accumulate(point_dist.begin(), point_dist.end(), 0.0)) / point_dist.size();
    
    // add line to vector
    if (radius > 0.05){
      Vector3d p1 = a + t_min*b;
      Vector3d p2 = a + t_max*b;

      line l;
      l.p1 = p1;
      l.p2 = p2;
      l.a = a;
      l.b = b;
      l.radius = radius;

      computed_lines.push_back(l);
    }
    
    X.removePoints(Y);

  } while ((X.points.size() > 1) && 
           ((opt_nlines == 0) || (opt_nlines > nlines)) && ros::ok());

  // clean up
  delete hough;

  return 0;
}