#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace turboreg {

// Enum for metric types
enum class MetricType {
    INLIER_COUNT,   // IN
    MAE,            // Mean Absolute Error
    MSE             // Mean Square Error
};



// Utility function to convert string to MetricType enum
MetricType string_to_metric_type(const std::string& metric_str);

// ModelSelection class for point cloud registration and model selection
class ModelSelection {
public:
    // Constructor: Initialize with metric type (IN, MAE, MSE) and inlier threshold
    ModelSelection(const std::string& metric_str, float inlier_threshold);
    ModelSelection(MetricType metric, float inlier_threshold);

    // Method to calculate the best matching clique based on the given metric
    torch::Tensor calculate_best_clique(
        const torch::Tensor& cliques_wise_trans, 
        const torch::Tensor& kpts_src, 
        const torch::Tensor& kpts_dst);

private:
    MetricType metric_type;  // Metric type (IN, MAE, MSE)
    float inlier_threshold;  // Inlier threshold
};

}


#endif // MODEL_SELECTION_H