#pragma once

#include <Eigen/Dense>
#include <functional>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include <vector>
//#include <xmlrpcpp/base64.h>
#include <b64/cencode.h>
#include <b64/cdecode.h>
#include <yaml-cpp/yaml.h>

class NeuralNetwork {

    Eigen::VectorXf yamlToVector(const YAML::Node &yaml) {
        Eigen::VectorXf vector;
        vector.resize(yaml.size());
        for(size_t i = 0; i < vector.size(); i++) {
            vector[i] = yaml[i].as<float>();
        }
        return vector;
    }

    base64<char> b46;
    Eigen::VectorXf yamlToWeightVector(const YAML::Node &yaml) {
        Eigen::VectorXf vector;
        auto text = yaml.as<std::string>();
        std::string data;
        int i = 0;
        b46.get(text.begin(), text.end(), std::back_insert_iterator<std::string>(data), i);
        if(data.size() != data.size() / 4 * 4) {
            throw 0;
        }
        vector.resize(data.size() / 4);
        memcpy(vector.data(), data.data(), data.size());
        return vector;
    }

    struct Layer {
        std::string name;
        std::vector<std::string> inputNames;
        std::vector<std::shared_ptr<Layer>> inputLayers;
        std::vector<Eigen::VectorXf> inputs;
        Eigen::VectorXf output;
        std::vector<Eigen::MatrixXf> weights;
        virtual void run() {}
    };

    std::unordered_map<std::string, std::shared_ptr<Layer>> layerMap;
    std::vector<std::shared_ptr<Layer>> layerList;
    Eigen::VectorXf _inputCenter, _inputScale, _outputCenter, _outputScale;
    std::shared_ptr<Layer> inputLayer, outputLayer;

public:
    const Eigen::VectorXf &inputCenter() { return _inputCenter; }
    const Eigen::VectorXf &inputScale() { return _inputScale; }
    const Eigen::VectorXf &outputCenter() { return _outputCenter; }
    const Eigen::VectorXf &outputScale() { return _outputScale; }

    void load(const std::string &filename) {
        ROS_INFO("loading network %s", filename.c_str());
        YAML::Node yaml = YAML::LoadFile(filename);
        size_t layerCount = yaml["layers"]["layers"].size();
        _inputCenter = yamlToVector(yaml["normalization"]["input"]["center"]);
        _inputScale = yamlToVector(yaml["normalization"]["input"]["scale"]);
        _outputCenter = yamlToVector(yaml["normalization"]["output"]["center"]);
        _outputScale = yamlToVector(yaml["normalization"]["output"]["scale"]);
        for(size_t layerIndex = 0; layerIndex < layerCount; layerIndex++) {
            YAML::Node yamlLayer = yaml["layers"]["layers"][layerIndex];
            std::string layerType = yamlLayer["class_name"].as<std::string>();
            std::shared_ptr<Layer> layer;
            if(layerType == "InputLayer") {
                struct Input : Layer {};
                layer = std::make_shared<Input>();
            }
            if(layerType == "Dense") {
                enum class Activation {
                    linear = 0,
                    relu,
                    sigmoid,
                };
                struct Dense : Layer {
                    Activation activation = Activation::linear;
                    bool useBias = false;
                    void run() {
                        output = weights[0] * inputs[0];
                        if(useBias) {
                            output += weights[1];
                        }
                        switch(activation) {
                        case Activation::relu:
                            for(size_t i = 0; i < output.size(); i++) {
                                output[i] = std::max(0.0f, output[i]);
                            }
                            break;
                        case Activation::sigmoid:
                            for(size_t i = 0; i < output.size(); i++) {
                                output[i] = 1.0 / (1.0 + std::exp(-output[i]));
                            }
                            break;
                        }
                    }
                };
                auto dense = std::make_shared<Dense>();
                dense->useBias = yamlLayer["config"]["use_bias"].as<bool>();
                std::string activation = yamlLayer["config"]["activation"].as<std::string>();
                if(activation == "relu") dense->activation = Activation::relu;
                if(activation == "sigmoid") dense->activation = Activation::sigmoid;
                layer = dense;
            }
            if(layerType == "Multiply") {
                struct Multiply : Layer {
                    void run() {
                        output = inputs[0];
                        for(size_t i = 1; i < inputs.size(); i++) {
                            output = output.cwiseProduct(inputs[i]);
                        }
                    }
                };
                layer = std::make_shared<Multiply>();
            }
            if(layerType == "Add") {
                struct Add : Layer {
                    void run() {
                        output = inputs[0];
                        for(size_t i = 1; i < inputs.size(); i++) {
                            output += inputs[i];
                        }
                    }
                };
                layer = std::make_shared<Add>();
            }
            if(!layer) {
                ROS_FATAL("unknown layer type %s", layerType.c_str());
                throw 0;
            }
            auto yamlWeights = yaml["weights"][layerIndex];
            std::vector<Eigen::MatrixXf> layerWeights;
            for(size_t i = 0; i < yamlWeights.size(); i++) {
                try {
                    layerWeights.push_back(yamlToWeightVector(yamlWeights[i]));
                    continue;
                } catch(const YAML::BadConversion &e) {
                }
                Eigen::MatrixXf weights(yamlWeights[i].size(), yamlToWeightVector(yamlWeights[i][0]).size());
                for(size_t row = 0; row < weights.rows(); row++) {
                    auto r = yamlToWeightVector(yamlWeights[i][row]);
                    if(r.size() != weights.row(row).size()) {
                        ROS_INFO("%i %i", (int)r.size(), (int)weights.cols());
                        throw 0;
                    }
                    weights.row(row) = r;
                }
                layerWeights.push_back(weights.transpose());
            }
            layer->weights = layerWeights;
            for(size_t i = 0; i < yamlLayer["inbound_nodes"][0].size(); i++) {
                layer->inputNames.push_back(yamlLayer["inbound_nodes"][0][i][0].as<std::string>());
            }
            layer->name = yamlLayer["name"].as<std::string>();
            layerMap[layer->name] = layer;
            layerList.push_back(layer);
        }
        for(auto layer : layerList) {
            for(auto n : layer->inputNames) {
                layer->inputLayers.push_back(layerMap[n]);
            }
        }
        for(auto layer : layerList) {
            ROS_INFO("%s", layer->name.c_str());
            for(auto n : layer->inputNames) {
                ROS_INFO("  input %s", n.c_str());
            }
        }
        inputLayer = layerMap[yaml["layers"]["input_layers"][0][0].as<std::string>()];
        outputLayer = layerMap[yaml["layers"]["output_layers"][0][0].as<std::string>()];
        ROS_INFO("ready");
    }

    void run(const Eigen::VectorXf &input, Eigen::VectorXf &output) {
        inputLayer->output = input;
        inputLayer->output = inputLayer->output - _inputCenter;
        inputLayer->output = inputLayer->output.cwiseProduct(_inputScale.cwiseInverse());
        for(auto &layer : layerList) {
            layer->inputs.resize(layer->inputLayers.size());
            for(size_t i = 0; i < layer->inputLayers.size(); i++) {
                layer->inputs[i] = layer->inputLayers[i]->output;
            }
            layer->run();
        }
        output = outputLayer->output;
        output = output.cwiseProduct(_outputScale);
        output = output + _outputCenter;
    }
};
