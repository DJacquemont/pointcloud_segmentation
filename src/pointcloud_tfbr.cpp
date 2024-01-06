#include <ros/ros.h>
#include "std_msgs/String.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>


/**
 * @brief Callback function broadcasting a TF transform mocap -> world to display 
 * the ToF point-cloud from the drone's perpective (world frame taking the drone's 
 * pose & orientation in the mocap frame).
 * 
 * This is not conventional as the world frame is supposed to be static.
 * 
 * @param[in] msg 
 */
void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  
  transformStamped.header.stamp = msg->header.stamp;
  transformStamped.header.frame_id = "mocap";
  transformStamped.child_frame_id = "world";
  transformStamped.transform.translation.x = msg->pose.position.x;
  transformStamped.transform.translation.y = msg->pose.position.y;
  transformStamped.transform.translation.z = msg->pose.position.z;
  transformStamped.transform.rotation.x = msg->pose.orientation.x;
  transformStamped.transform.rotation.y = msg->pose.orientation.y;
  transformStamped.transform.rotation.z = msg->pose.orientation.z;
  transformStamped.transform.rotation.w = msg->pose.orientation.w;

  br.sendTransform(transformStamped);
}


int main(int argc, char* argv[]){

  ROS_INFO("Pointcloud TF broadcaster node starting");

  ros::init(argc, argv, "pointcloud_tfbr");

  ros::NodeHandle node;

  ros::Subscriber pose_sub = node.subscribe("/mavros/local_position/pose", 0, poseCallback);

  ros::spin();

  return 0;
}
