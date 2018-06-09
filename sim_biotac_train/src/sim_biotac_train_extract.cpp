#include <sim_biotac/sim_biotac.h>

#include <apriltags_ros/AprilTagDetectionArray.h>
#include <eigen_conversions/eigen_msg.h>
#include <fstream>
#include <geometry_msgs/WrenchStamped.h>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sr_robot_msgs/BiotacAll.h>
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv) {

    if(argc < 2) {
        puts("usage: sim_biotac_adjust <bagfile>");
        return -1;
    }

    std::string bagname = argv[1];

    YAML::Node yaml = YAML::LoadFile(bagname + ".opt.yaml");
    double biotacRadius = yaml["biotac"]["radius"].as<double>();
    Eigen::Vector3d tagToProbe = Eigen::Vector3d(yaml["probe"]["position"]["x"].as<double>(), yaml["probe"]["position"]["y"].as<double>(), yaml["probe"]["position"]["z"].as<double>());
    auto tagToBiotac = makeTransform(0.000, -0.002, -0.009, 0.000, 0.000, -0.707, 0.707);
    tagToBiotac.translation() = Eigen::Vector3d(yaml["biotac"]["position"]["x"].as<double>(), yaml["biotac"]["position"]["y"].as<double>(), yaml["biotac"]["position"]["z"].as<double>());

    std::cout << biotacRadius << std::endl;
    std::cout << tagToBiotac.translation().format(Eigen::IOFormat()) << std::endl;
    std::cout << tagToProbe.format(Eigen::IOFormat()) << std::endl;

    rosbag::Bag bag;
    bag.open(bagname, rosbag::bagmode::Read);

    rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>({"/tag_detections", "wireless_ft/wrench_3", "rh/tactile", "ft_contact"})));

    auto probeTag = 9;
    auto biotacTag = 1;

    std::vector<std::string> headers = {
        "%time",
        "field.position.x",
        "field.position.y",
        "field.position.z",
        "field.wrench.force.x",
        "field.wrench.force.y",
        "field.wrench.force.z",
        "field.tactiles0.pac0",
        "field.tactiles0.pac1",
        "field.tactiles0.pdc",
        "field.tactiles0.tac",
        "field.tactiles0.tdc",
        "field.tactiles0.electrodes0",
        "field.tactiles0.electrodes1",
        "field.tactiles0.electrodes2",
        "field.tactiles0.electrodes3",
        "field.tactiles0.electrodes4",
        "field.tactiles0.electrodes5",
        "field.tactiles0.electrodes6",
        "field.tactiles0.electrodes7",
        "field.tactiles0.electrodes8",
        "field.tactiles0.electrodes9",
        "field.tactiles0.electrodes10",
        "field.tactiles0.electrodes11",
        "field.tactiles0.electrodes12",
        "field.tactiles0.electrodes13",
        "field.tactiles0.electrodes14",
        "field.tactiles0.electrodes15",
        "field.tactiles0.electrodes16",
        "field.tactiles0.electrodes17",
        "field.tactiles0.electrodes18",
    };

    std::vector<Eigen::VectorXd> data;

    ROS_INFO("reading bag");
    bool tagsInitialized = false;
    bool forceInitialized = false;
    bool contactInitialized = false;
    Eigen::Affine3d probeTagPose;
    Eigen::Affine3d biotacTagPose;
    KDL::Wrench wrench;
    Eigen::Vector3d contactPoint, contactForce;
    for(auto &message : view) {

        if(auto wrenchStamped = message.instantiate<geometry_msgs::WrenchStamped>()) {
            forceInitialized = true;
            tf::wrenchMsgToKDL(wrenchStamped->wrench, wrench);
        }
        if(auto detections = message.instantiate<apriltags_ros::AprilTagDetectionArray>()) {
            if(detections->detections.size() == 2) {
                tagsInitialized = true;
                for(auto &detection : detections->detections) {
                    Eigen::Affine3d tagPose;
                    tf::poseMsgToEigen(detection.pose.pose, tagPose);
                    if(detection.id == probeTag) {
                        probeTagPose = tagPose;
                    }
                    if(detection.id == biotacTag) {
                        biotacTagPose = tagPose;
                    }
                }
            }
        }
        if(!tagsInitialized) continue;
        if(!forceInitialized) continue;
        if(auto biotac = message.instantiate<sr_robot_msgs::BiotacAll>()) {
            auto tactile = biotac->tactiles[0];
            Eigen::Vector3d position0 = (biotacTagPose * tagToBiotac).inverse() * (probeTagPose * tagToProbe);
            Eigen::Vector3d position = position0;
            if(position.x() < 0.0) {
                double f = biotacRadius / Eigen::Vector2d(position.y(), position.z()).norm();
                position.y() *= f;
                position.z() *= f;
            } else {
                position *= biotacRadius / position.norm();
            }

            Eigen::Vector3d force(wrench.force.x(), wrench.force.y(), wrench.force.z());
            auto r = ((biotacTagPose * tagToBiotac).inverse() * probeTagPose);
            r.translation() = Eigen::Vector3d::Zero();
            force = r * force;

            if(force.norm() > 0.1 && (position - position0).norm() > 0.005) {
                continue;
            }

            position = position0;

            Eigen::VectorXd row(31);
            size_t i = 0;
            row[i++] = message.getTime().toNSec();
            row[i++] = position.x();
            row[i++] = position.y();
            row[i++] = position.z();
            row[i++] = force.x();
            row[i++] = force.y();
            row[i++] = force.z();
            row[i++] = tactile.pac0;
            row[i++] = tactile.pac1;
            row[i++] = tactile.pdc;
            row[i++] = tactile.tac;
            row[i++] = tactile.tdc;
            for(auto &e : tactile.electrodes) {
                row[i++] = e;
            }
            data.push_back(row);
        }
    }

    ROS_INFO("writing csv");
    std::ofstream stream(bagname + ".csv");
    for(auto &header : headers) {
        if(header != headers.front()) stream << ",";
        stream << header;
    }
    stream << "\n";
    writecsv(stream, data);
}
