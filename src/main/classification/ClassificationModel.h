#ifndef Classification_Model_H
#define Classification_Model_H

// libsvm
#include "svm.h"
// Standard
#include <vector>
#include <iostream>

class ClassificationModel {
protected:
    std::map<std::string, cv::Mat> normalFeatures;
    std::map<std::string, cv::Mat> anomalyFeatures;

    ClassificationModel(const std::map<std::string, cv::Mat>& normalFeatures,
                        const std::map<std::string, cv::Mat>& anomalyFeatures)
        : normalFeatures(normalFeatures), anomalyFeatures(anomalyFeatures) {}

public:

    virtual std::vector<std::vector<svm_node>> toSvmNodes(const std::map<std::string, cv::Mat>& featureMap) = 0;
    virtual bool classify(std::vector<svm_node>& nodes) = 0;
    virtual bool result() const = 0;
    virtual ~ClassificationModel() = default;

};

class SVM : public ClassificationModel {

public:

    svm_model* model;

    SVM(const std::map<std::string, cv::Mat>& normalFeatures, const std::map<std::string, cv::Mat>& anomalyFeatures)
        : ClassificationModel(normalFeatures, anomalyFeatures) {

               std::vector<double> labels;
    std::vector<std::vector<svm_node>> nodes;

    auto addSamples = [&](const std::map<std::string, cv::Mat>& featureMap, double label) {
        int numCells = featureMap.begin()->second.total();
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            labels.push_back(label);
            std::vector<svm_node> node;
            int featureIdx = 1;
            for (auto& [_, mat] : featureMap) {
                cv::Mat flat;
                mat.reshape(1, mat.total()).convertTo(flat, CV_64F);
                node.push_back({featureIdx++, flat.at<double>(cellIdx)});
            }
            node.push_back({-1, 0}); // sentinel
            nodes.push_back(std::move(node));
        }
    };

    addSamples(normalFeatures,  +1.0);
    addSamples(anomalyFeatures, -1.0);

    int n = labels.size();
    std::vector<svm_node*> x_ptrs(n);
    for (int i = 0; i < n; ++i)
        x_ptrs[i] = nodes[i].data();

    std::cout << "CHECK SAMPLE STABILITY\n";
    for (int i = 0; i < std::min(5, (int)nodes.size()); ++i) {
        std::cout << nodes[i][0].value << "\n";
    }

    svm_problem prob;
    prob.l = n;
    prob.y = labels.data();
    prob.x = x_ptrs.data();

    svm_parameter param{};
    param.svm_type    = C_SVC;
    param.kernel_type = RBF;
    param.C           = 1.0;
    param.gamma       = 1.0 / n; 
    param.eps         = 1e-3;
    param.cache_size  = 100;
    param.nr_weight   = 0;
    std::cout << "n: " << 1.0 / n << "\n";
    const char* err = svm_check_parameter(&prob, &param);
    if (err) throw std::runtime_error(std::string("SVM param error: ") + err);

    model = svm_train(&prob, &param);

    }

    /*
     * classify
     * DESCRITPION
     * PARAMS
    */ 
    bool classify(std::vector<svm_node>& nodes) override {

        // Predict a new point
        // std::cout << "nodes: " << nodes.data() << "\n";
        double prediction = svm_predict(model, nodes.data());
        //svm_free_and_destroy_model(&model);

        // std::cout << "Predicted class: " << prediction << "\n"; // +1 or -1

        if (prediction == +1)
            return true;
        else 
            return false;

    }

    /*
     * result
     * DESCRITPION
     * PARAMS
    */ 
    bool result() const override { 

        std::cout << "ERROR: NOT IMPLMENTED\n";
        return true;
    }

    /*
     * toSVMNodes
     * Converts a map of cv::Mat features into libsvm-ready nodes
     */
    std::vector<std::vector<svm_node>> toSvmNodes(const std::map<std::string, cv::Mat>& featureMap) {

        int numCells = featureMap.begin()->second.total();
        std::vector<std::vector<svm_node>> nodes(numCells);

        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            int featureIdx = 1;
            for (auto& [_, mat] : featureMap) {
                // Flatten mat and grab the value for this cell
                cv::Mat flat = mat.reshape(1, mat.total());
                flat.convertTo(flat, CV_64F);
                double val = flat.at<double>(cellIdx);
                nodes[cellIdx].push_back({featureIdx++, val});
            }
            nodes[cellIdx].push_back({-1, 0}); // sentinel
        }
        return nodes;
    }

};

#endif
