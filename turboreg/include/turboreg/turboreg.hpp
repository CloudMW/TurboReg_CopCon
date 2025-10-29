#ifndef TURBOREG_HPP
#define TURBOREG_HPP

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include "rigid_transform.hpp"
#include "model_selection.hpp"
#include <unordered_map>

namespace turboreg
{
    torch::Tensor coplanar_constraint(const torch::Tensor &cliques_tensor, const torch::Tensor &kpts_src, const torch::Tensor &kpts_dst
        ,float threshold);

    class TurboRegGPU
    {
    public:
        TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str);
        RigidTransform runRegCXX(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);
        torch::Tensor runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);

    private:
        int max_N;               // Maximum number of correspondences
        float tau_length_consis; // \tau
        float radiu_nms;  // Radius for avoiding the instability of the solution
        int num_pivot;    // Number of pivot points, K_1
        float tau_inlier; // Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
        bool hard = true; // Flag for hard thresholding. NOTE: just using hard compatibility graph.
        MetricType eval_metric;
    };

}

#endif // TURBOREG_HPP
