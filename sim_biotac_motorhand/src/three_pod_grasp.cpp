#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/SwitchController.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <sr_robot_msgs/RobotTeachMode.h>

#include <dynamic_reconfigure/Config.h>
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>

bool stopTrajectoryController() {
  ros::NodeHandle nh;

  ros::ServiceClient listClient =
      nh.serviceClient<controller_manager_msgs::ListControllers>(
          "controller_manager/list_controllers");
  ros::ServiceClient switchClient =
      nh.serviceClient<controller_manager_msgs::SwitchController>(
          "controller_manager/switch_controller");
  controller_manager_msgs::ListControllers srv;

  listClient.call(srv);
  for (int i = 0; i < srv.response.controller.size(); i++) {
    if (srv.response.controller[i].name == "rh_trajectory_controller") {
      controller_manager_msgs::SwitchController switchSrv;
      switchSrv.request.stop_controllers.push_back("rh_trajectory_controller");
      switchSrv.request.strictness = 1;
      switchClient.call(switchSrv);
      if (switchSrv.response.ok == true) {
        ROS_WARN("Stopped trajectory controller");
        return true;
      }
    }
  }
  return false;
}

void switchThreeJointsToEffort() {
  ros::NodeHandle nh;
  ros::ServiceClient switchClient =
      nh.serviceClient<controller_manager_msgs::SwitchController>(
          "controller_manager/switch_controller");
  ros::ServiceClient loadClient =
      nh.serviceClient<controller_manager_msgs::LoadController>(
          "controller_manager/load_controller");

  controller_manager_msgs::LoadController loadSrv;
  loadSrv.request.name = "/sh_rh_ffj3_effort_controller";
  loadClient.call(loadSrv);
  loadSrv.request.name = "/sh_rh_mfj3_effort_controller";
  loadClient.call(loadSrv);
  loadSrv.request.name = "/sh_rh_thj5_effort_controller";
  loadClient.call(loadSrv);

  controller_manager_msgs::SwitchController switchSrv;
  switchSrv.request.stop_controllers.push_back(
      "/sh_rh_ffj3_position_controller");
  switchSrv.request.stop_controllers.push_back(
      "/sh_rh_mfj3_position_controller");
  switchSrv.request.stop_controllers.push_back(
      "/sh_rh_thj5_position_controller");
  switchSrv.request.start_controllers.push_back(
      "/sh_rh_ffj3_effort_controller");
  switchSrv.request.start_controllers.push_back(
      "/sh_rh_mfj3_effort_controller");
  switchSrv.request.start_controllers.push_back(
      "/sh_rh_thj5_effort_controller");
  switchSrv.request.strictness = 1;
  switchClient.call(switchSrv);
}

void adjustPid() {
  dynamic_reconfigure::ReconfigureRequest srv_req;
  dynamic_reconfigure::ReconfigureResponse srv_resp;
  dynamic_reconfigure::DoubleParameter double_param;
  dynamic_reconfigure::Config conf;

  double_param.name = "p";
  double_param.value = 200;
  conf.doubles.push_back(double_param);

  double_param.name = "i";
  double_param.value = 0;
  conf.doubles.push_back(double_param);

  double_param.name = "d";
  double_param.value = 0;
  conf.doubles.push_back(double_param);

  srv_req.config = conf;

  ros::service::call("/sh_rh_ffj3_position_controller/pid/set_parameters",
                     srv_req, srv_resp);
  ros::service::call("/sh_rh_mfj3_position_controller/pid/set_parameters",
                     srv_req, srv_resp);
  ros::service::call("/sh_rh_thj4_position_controller/pid/set_parameters",
                     srv_req, srv_resp);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ThreePodGrasp");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(2);
  spinner.start();

  moveit::planning_interface::MoveGroupInterface hand("right_hand");
  moveit::planning_interface::MoveGroupInterface arm("right_arm");

  ros::Publisher pub_ff3 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_ffj3_effort_controller/command", 10);
  ros::Publisher pub_ff0 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_ffj0_effort_controller/command",10);
  ros::Publisher pub_mf3 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_mfj3_effort_controller/command", 10);
  ros::Publisher pub_mf0 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_mfj0_effort_controller/command",10);
  ros::Publisher pub_th5 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_thj5_effort_controller/command", 10);
  ros::Publisher pub_th2 = nh.advertise<std_msgs::Float64>(
      "/sh_rh_thj2_effort_controller/command",10);

  std::vector<double> joint_state;

  hand.setGoalJointTolerance(0.03);

  // Close hand
  joint_state = hand.getCurrentJointValues();
  // joint_state[3] = 1;     //ffj3
  // joint_state[4] = 0.7;     //ffj0
  joint_state[12] = 1.1; // mfj3
  joint_state[13] = 0.7; // mfj0
  joint_state[19] = 0.3; // thj5
  joint_state[20] = 1.4; // thj4
  joint_state[22] = 0.8; // thj2

  hand.setJointValueTarget(joint_state);
  if (!hand.move()) {
    ROS_ERROR("Failed to close hand");
    return -1;
  }

  //adjustPid();

/*
  // Switch to effort controllers
    // Whole hand
    ros::ServiceClient client =
    nh.serviceClient<sr_robot_msgs::RobotTeachMode>("/teach_mode");
    sr_robot_msgs::RobotTeachMode srv;
    srv.request.teach_mode = 1;
    srv.request.robot = "right_hand";
    if (!client.call(srv))
      ROS_ERROR("Failed switching controllers");

    // Only three finger
    switchThreeJointsToEffort();
    ros::Duration(1).sleep();


  // Publish effort  
    ROS_INFO("Publish effort");
    std_msgs::Float64 msg;
   
    // Fist finger
    msg.data = 0.3;
    pub_ff3.publish(msg);
    //pub_ff0.publish(msg);

    // Middle finger
    msg.data = 0.2;
    pub_mf3.publish(msg);
    // pub_mf0.publish(msg);

    // Thumb   
    msg.data = 0.5;
    pub_th5.publish(msg);
    // pub_th2.publish(msg);
*/


/*
  // Lift movement
    arm.setMaxVelocityScalingFactor(0.01);
    arm.setMaxAccelerationScalingFactor(0.01);

    geometry_msgs::PoseStamped current_pose = arm.getCurrentPose();
    geometry_msgs::Pose pose = current_pose.pose;
    pose.position.z += 0.05;
    arm.setPoseTarget(pose);

    if(!arm.move()) {
      ROS_ERROR("Failed to lift arm");
      return -1;
    }
*/

  return 0;
}
