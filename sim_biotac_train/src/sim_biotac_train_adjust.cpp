#include <sim_biotac/sim_biotac.h>

#include <apriltags_ros/AprilTagDetectionArray.h>
#include <eigen_conversions/eigen_msg.h>
#include <fstream>
#include <geometry_msgs/WrenchStamped.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv) {

    if(argc < 2) {
        puts("usage: sim_biotac_adjust <bagfile>");
        return -1;
    }

    std::string bagname = argv[1];

    double probeSphereDiameter = 0.004;
    double probeSphereRadius = probeSphereDiameter * 0.5f;

    double biotacRadius = 0.007;

    auto probeTag = 9;
    auto tagToProbe = Eigen::Vector3f(0.000, 0.051, 0.038);
    auto biotacTag = 1;
    auto tagToBiotac = makeTransform(0.000, -0.002, -0.009, 0.000, 0.000, -0.707, 0.707).cast<float>();

    auto tagToProbe0 = tagToProbe;
    auto tagToBiotac0 = tagToBiotac;

    ROS_INFO("reading");
    std::vector<apriltags_ros::AprilTagDetectionArray::ConstPtr> tagDetections;
    {
        rosbag::Bag bag;
        bag.open(bagname, rosbag::bagmode::Read);
        rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>({"/tag_detections", "wireless_ft/wrench_3"})));
        bool inContactPrev = false;
        bool forceInitialized = false;
        Eigen::Vector3d force = Eigen::Vector3d::Zero();
        Eigen::Vector3d forcePrev = Eigen::Vector3d::Zero();
        int progressLast = 0;
        for(auto &message : view) {
            int progress = (int)round((message.getTime() - view.getBeginTime()).toSec() * 100.0 / (view.getEndTime() - view.getBeginTime()).toSec());
            if(progress != progressLast) {
                ROS_INFO("reading %i%%", progress);
                progressLast = progress;
            }
            if(auto wrench = message.instantiate<geometry_msgs::WrenchStamped>()) {
                forcePrev = force;
                tf::vectorMsgToEigen(wrench->wrench.force, force);
                forceInitialized = true;
            }
            if(!forceInitialized) continue;
            if(auto detections = message.instantiate<apriltags_ros::AprilTagDetectionArray>()) {
                bool inContact = (force.norm() > 0.3);
                if(!inContact && inContactPrev) {
                    tagDetections.push_back(detections);
                }
                inContactPrev = inContact;
            }
        }
    }

    ROS_INFO("sample count %i", (int)tagDetections.size());

    struct Solution {
        Eigen::Vector3f tagToProbe;
        Eigen::Affine3f tagToBiotac;
        float biotacRadius;
    };

    std::vector<Eigen::Vector3f> contactPoints;

    struct Sample {
        Eigen::Affine3f probeTagPose;
        Eigen::Affine3f biotacTagPose;
        Eigen::Affine3f probeTagToBiotacTag;
    };

    std::vector<Sample> samples;
    for(auto detections : tagDetections) {
        if(detections->detections.size() != 2) {
            continue;
        }
        Sample sample;
        for(auto &detection : detections->detections) {
            Eigen::Affine3d tagPose;
            tf::poseMsgToEigen(detection.pose.pose, tagPose);
            if(detection.id == probeTag) {
                sample.probeTagPose = tagPose.cast<float>();
            }
            if(detection.id == biotacTag) {
                sample.biotacTagPose = tagPose.cast<float>();
            }
        }
        sample.probeTagToBiotacTag = sample.biotacTagPose.inverse() * sample.probeTagPose;
        samples.push_back(sample);
    }

    auto transform = [&](Solution solution) {
        contactPoints.resize(samples.size());
        Eigen::Vector3f tagToProbe = solution.tagToProbe;
        Eigen::Affine3f tagToBiotac = solution.tagToBiotac;
        Eigen::Affine3f tagToBiotacInverse = tagToBiotac.inverse();
#pragma omp parallel for
        for(size_t i = 0; i < samples.size(); i++) {
            auto &sample = samples[i];
            Eigen::Vector3f probePosition = (tagToBiotacInverse * (sample.probeTagToBiotacTag * tagToProbe));
            contactPoints[i] = probePosition;
        }
    };

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> ndist;
    std::uniform_real_distribution<float> udist;

    std::vector<float> errors;

    auto evaluate = [&](const Solution &solution, int start = 0, int stepsize = 1) {
        auto biotacRadius = solution.biotacRadius;
        Eigen::Vector3f tagToProbe = solution.tagToProbe;
        Eigen::Affine3f tagToBiotac = solution.tagToBiotac;
        Eigen::Affine3f tagToBiotacInverse = tagToBiotac.inverse();
        float radius = biotacRadius + probeSphereRadius;
        errors.resize(samples.size());
#pragma omp parallel for
        for(size_t i = 0; i < samples.size(); i++) {
            auto &sample = samples[i];
            Eigen::Vector3f probePosition = (tagToBiotacInverse * (sample.probeTagToBiotacTag * tagToProbe));
            if(probePosition.x() < 0.0) {
                probePosition.x() = 0.0;
            }
            double e = probePosition.norm() - radius;
            errors[i] = std::fabs(e);
        }
        float error = 0.0;
        for(size_t i = 0; i < samples.size(); i++) {
            error += errors[i];
        }
        return error / samples.size();
    };

    Solution currentSolution;
    currentSolution.tagToProbe = tagToProbe;
    currentSolution.tagToBiotac = tagToBiotac;
    currentSolution.biotacRadius = biotacRadius;
    transform(currentSolution);
    double currentError = evaluate(currentSolution);

    {
        std::ofstream f(bagname + ".opt.samples.0.csv");
        f << "index,x,y,z"
          << "\n";
        transform(currentSolution);
        for(size_t i = 0; i < contactPoints.size(); i++) {
            auto &p = contactPoints[i];
            f << i << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
        }
    }

    {
        std::ofstream logfile(bagname + ".opt.log.csv");
        logfile << "step,error" << std::endl;
        size_t steps = 1000;
        for(size_t step = 0; step < steps; step++) {
            for(size_t i = 0; i < 16; i++) {
                Solution newSolution = currentSolution;
                double d = udist(rng) * udist(rng) * 0.0002;
                newSolution.tagToProbe += Eigen::Vector3f(ndist(rng), ndist(rng), ndist(rng)) * d;
                newSolution.tagToBiotac.translation() += Eigen::Vector3f(ndist(rng), ndist(rng), ndist(rng)) * d;
                double newError = evaluate(newSolution);
                if(newError < currentError) {
                    currentSolution = newSolution;
                    currentError = newError;
                }
            }
            ROS_INFO("step:%i error:%f", (int)step, currentError);
            logfile << step << "," << currentError << std::endl;
            transform(currentSolution);
        }
    }

    {
        std::ofstream f(bagname + ".opt.samples.csv");
        f << "index,x,y,z"
          << "\n";
        transform(currentSolution);
        for(size_t i = 0; i < contactPoints.size(); i++) {
            auto &p = contactPoints[i];
            f << i << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
        }
    }

    {
        std::ofstream resultfile(bagname + ".opt.yaml");
        YAML::Emitter yaml;
        {
            yaml << YAML::BeginMap;

            yaml << YAML::Key << "biotac";
            {
                yaml << YAML::BeginMap;

                yaml << YAML::Key << "radius";
                yaml << YAML::Value << currentSolution.biotacRadius;

                yaml << YAML::Key << "position";
                {
                    yaml << YAML::BeginMap;
                    yaml << YAML::Key << "x";
                    yaml << YAML::Value << currentSolution.tagToBiotac.translation().x();
                    yaml << YAML::Key << "y";
                    yaml << YAML::Value << currentSolution.tagToBiotac.translation().y();
                    yaml << YAML::Key << "z";
                    yaml << YAML::Value << currentSolution.tagToBiotac.translation().z();
                    yaml << YAML::EndMap;
                }

                yaml << YAML::EndMap;
            }

            yaml << YAML::Key << "probe";
            {
                yaml << YAML::BeginMap;

                yaml << YAML::Key << "position";
                {
                    yaml << YAML::BeginMap;
                    yaml << YAML::Key << "x";
                    yaml << YAML::Value << currentSolution.tagToProbe.x();
                    yaml << YAML::Key << "y";
                    yaml << YAML::Value << currentSolution.tagToProbe.y();
                    yaml << YAML::Key << "z";
                    yaml << YAML::Value << currentSolution.tagToProbe.z();
                    yaml << YAML::EndMap;
                }

                yaml << YAML::EndMap;
            }

            yaml << YAML::EndMap;
        }
        ROS_INFO("\n%s", yaml.c_str());
        resultfile << yaml.c_str();
    }
}
