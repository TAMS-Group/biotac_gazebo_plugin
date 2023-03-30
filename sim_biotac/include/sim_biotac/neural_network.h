#pragma once

#include <Eigen/Dense>
#include <functional>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include <vector>
//#include <xmlrpcpp/base64.h>
//replace xmlrpcpp with libb64
#include <b64/encode.h>
#include <b64/decode.h>

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

    //base64<char> b46;
    base64::decoder E;

//    Eigen::VectorXf yamlToWeightVector(const YAML::Node &yaml) {
//        Eigen::VectorXf vector;
//        ROS_INFO_STREAM("before as................................................................ ");
//        ROS_INFO_STREAM("yaml content "<<yaml);
//        const std::string text = yaml.as<std::string>();
//        ROS_INFO_STREAM("yaml content2 "<<yaml);
//        char* plaintext_out = new char[10000];
//        ROS_INFO_STREAM("input text "<<text<<" and text size is "<<text.size());
//        //int i = 0;
//        //b46.get(text.begin(), text.end(), std::back_insert_iterator<std::string>(data), i);
//        int plainlength;
//        plainlength = E.decode(text.c_str(), text.size(), plaintext_out);

//        ROS_INFO_STREAM( "length of weight : " << plainlength);
//        for(int i = 0; i < plainlength; i++)
//            ROS_INFO_STREAM( "convert data is : " <<i<<" , "<< (int)*(plaintext_out + i));
//        std::string data;
//        data  = plaintext_out;

//        ROS_INFO_STREAM( "updated ..................data and size are : " <<data<<" , "<<data.size());

//        if(data.size() != data.size() / 4 * 4) {
//            throw 0;
//        }
//        vector.resize(data.size() / 4);
//        memcpy(vector.data(), data.data(), data.size());
//        delete [] plaintext_out;
//        return vector;
//    }

    Eigen::VectorXf yamlToWeightVector(const std::string text) {
        Eigen::VectorXf vector;
        char* plaintext_out = new char[10000];
        int plainlength;
//        std::string data = "";
        plainlength = E.decode(text.c_str(), text.size(), plaintext_out);

//        ROS_INFO_STREAM( "length of weight : " << plainlength);
//        for(int i = 0; i < plainlength; i++){
//            ROS_INFO_STREAM( "convert data is : " <<i<<" , "<< (int)*(plaintext_out + i));
//            data = data + plaintext_out[i];
//        }

        std::string data(reinterpret_cast<const char*>(plaintext_out), sizeof(plaintext_out)/sizeof(plaintext_out[0]));

//        ROS_INFO_STREAM( "updated ..................data and size are : " <<data<<" , "<<data.size());

        if(data.size() != data.size() / 4 * 4) {
            throw 0;
        }
        vector.resize(data.size() / 4);
        memcpy(vector.data(), data.data(), data.size());
        delete [] plaintext_out;
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
            ROS_INFO_STREAM("layerCount is "<<layerCount<<" and current layerIndex is "<<layerIndex <<"layer type is "<<layerType);
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
            YAML::Node yamlWeights = yaml["weights"][layerIndex];
            std::vector<Eigen::MatrixXf> layerWeights;
            std::string full_weight_txt,single_col_weight_txt,cur_col_weight_txt;

            for(size_t i = 0; i < yamlWeights.size(); i++) {
//                ROS_INFO_STREAM("yamlWeights.size() is "<<yamlWeights.size());
                try {
//                    ROS_INFO_STREAM("before full text print");
                    full_weight_txt = yamlWeights[i].as<std::string>();
//                    ROS_INFO_STREAM("full text "<<full_weight_txt);
                    Eigen::MatrixXf full_weight_mat = yamlToWeightVector(full_weight_txt);
                    layerWeights.push_back(full_weight_mat);

//                    layerWeights.push_back(yamlToWeightVector(yamlWeights[i]));
                    continue;
                } catch(const YAML::BadConversion &e) {
//                    ROS_INFO_STREAM("bad convert..........................");
                }
//                ROS_INFO_STREAM("yamlWeights[i].size() "<<yamlWeights[i].size());
//                ROS_INFO_STREAM("yamlWeights[i][0] "<<yamlWeights[i][0]);
                single_col_weight_txt = yamlWeights[i][0].as<std::string>();

                Eigen::VectorXf single_weight_vec = yamlToWeightVector(single_col_weight_txt);
//                ROS_INFO_STREAM("vec size is "<<single_weight_vec.size());
                Eigen::MatrixXf weights(yamlWeights[i].size(), single_weight_vec.size());

//                Eigen::MatrixXf weights(yamlWeights[i].size(), yamlToWeightVector(yamlWeights[i][0]).size());
                for(size_t row = 0; row < weights.rows(); row++) {
                    cur_col_weight_txt = yamlWeights[i][row].as<std::string>();
//                    ROS_INFO_STREAM("cur_col_weight_txt is "<<cur_col_weight_txt);
                    auto r = yamlToWeightVector(cur_col_weight_txt);
                    if(r.size() != weights.row(row).size()) {
                        ROS_INFO("%i %i", (int)r.size(), (int)weights.cols());
                        throw 0;
                    }
                    ROS_INFO_STREAM("row is "<<r);
                    weights.row(row) = r;
                }
                layerWeights.push_back(weights.transpose());
            }
            ROS_INFO("weight value is loaded successfully");

            layer->weights = layerWeights;
            for(size_t i = 0; i < yamlLayer["inbound_nodes"][0].size(); i++) {
                layer->inputNames.push_back(yamlLayer["inbound_nodes"][0][i][0].as<std::string>());
            }
            layer->name = yamlLayer["name"].as<std::string>();
            layerMap[layer->name] = layer;
            layerList.push_back(layer);
        }
        ROS_INFO("loading data finished.................................");
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
