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

    virtual bool classify(cv::Mat cell) = 0;
    virtual bool result() const = 0;
    virtual ~ClassificationModel() = default;

};

class SVM : public ClassificationModel {

public:

    svm_model* model;

    SVM(const std::map<std::string, cv::Mat>& normalFeatures,
                         const std::map<std::string, cv::Mat>& anomalyFeatures)
        : ClassificationModel(normalFeatures, anomalyFeatures) {

        // Training data: 4 points, 2 features each
        // Labels: +1 or -1
        std::vector<double> labels = {+1, +1, -1, -1};
        std::vector<std::vector<double>> features = {
            {2.0, 3.0}, {3.0, 3.5}, {-1.0, -1.5}, {-2.0, -2.0}
        };

        int n = labels.size();

        // Build libsvm node structure
        std::vector<std::vector<svm_node>> nodes(n);
        for (int i = 0; i < n; ++i) {
            for (double val : features[i])
                nodes[i].push_back({(int)(&val - &features[i][0]) + 1, val});
            nodes[i].push_back({-1, 0}); // sentinel — required by libsvm
        }

        // Fill the training problem
        svm_problem prob;
        prob.l = n;
        prob.y = labels.data();
        std::vector<svm_node*> x_ptrs(n);
        for (int i = 0; i < n; ++i) x_ptrs[i] = nodes[i].data();
        prob.x = x_ptrs.data();

        // Configure SVM: C-SVC with RBF kernel
        svm_parameter param{};
        param.svm_type    = C_SVC;
        param.kernel_type = RBF;
        param.C           = 1.0;   // regularisation — tune this
        param.gamma       = 0.5;   // RBF bandwidth — tune this
        param.eps         = 1e-3;
        param.cache_size  = 100;   // MB
        param.nr_weight   = 0;

        model = svm_train(&prob, &param);

    }

    /*
     * classify
     * DESCRITPION
     * PARAMS
    */ 
    bool classify(cv::Mat cell) override {

        // Predict a new point
        std::vector<svm_node> test = {{1, 2.5}, {2, 3.0}, {-1, 0}};
        double prediction = svm_predict(model, test.data());
        std::cout << "Predicted class: " << prediction << "\n"; // +1 or -1

        svm_free_and_destroy_model(&model);
        return true;
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
};

#endif
