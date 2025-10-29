#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <turboreg/turboreg.hpp>

torch::Tensor turboreg::coplanar_constraint(const torch::Tensor &cliques_tensor, const torch::Tensor &kpts_src, const torch::Tensor &kpts_dst, float threshold)
{
    // cliques_tensor: [N, 3] - 点对组索引
    // kpts_src, kpts_dst: [M, 3] - 源点云和目标点云坐标

    int64_t N = cliques_tensor.size(0);
    int64_t M = kpts_src.size(0);

    // 保存原始设备信息
    auto original_device = kpts_src.device();

    // 将输入数据移到CPU（PCL只支持CPU）
    auto kpts_src_cpu = kpts_src.cpu();
    auto kpts_dst_cpu = kpts_dst.cpu();
    auto cliques_cpu = cliques_tensor.cpu();

    // 转换为PCL点云
    auto tensor_to_cloud = [](const torch::Tensor &pts) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pts已经在CPU上，直接访问
        auto accessor = pts.accessor<float, 2>();

        cloud->points.resize(pts.size(0));
        for (int i = 0; i < pts.size(0); ++i) {
            cloud->points[i].x = accessor[i][0];
            cloud->points[i].y = accessor[i][1];
            cloud->points[i].z = accessor[i][2];
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    };

    auto src_cloud = tensor_to_cloud(kpts_src_cpu);
    auto dst_cloud = tensor_to_cloud(kpts_dst_cpu);

    // 计算法向量
    auto compute_normals = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        ne.setKSearch(10); // 使用10个最近邻点
        
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        ne.compute(*normals);
        return normals;
    };

    auto src_normals = compute_normals(src_cloud);
    auto dst_normals = compute_normals(dst_cloud);

    // 将法向量转换为tensor [M, 3] - 先在CPU上创建
    auto normals_to_tensor = [&](pcl::PointCloud<pcl::Normal>::Ptr normals) {
        // 在CPU上创建tensor
        torch::Tensor normal_tensor = torch::zeros({M, 3}, torch::kFloat32);
        auto accessor = normal_tensor.accessor<float, 2>();
        
        for (int i = 0; i < M; ++i) {
            accessor[i][0] = normals->points[i].normal_x;
            accessor[i][1] = normals->points[i].normal_y;
            accessor[i][2] = normals->points[i].normal_z;
        }
        return normal_tensor;
    };

    auto src_normals_tensor = normals_to_tensor(src_normals);
    auto dst_normals_tensor = normals_to_tensor(dst_normals);

    // 获取每个clique对应的法向量 [N, 3, 3]
    // 注意：cliques_cpu 和 normals tensor 都在CPU上
    auto src_norms = src_normals_tensor.index_select(0, cliques_cpu.flatten()).view({N, 3, 3});
    auto dst_norms = dst_normals_tensor.index_select(0, cliques_cpu.flatten()).view({N, 3, 3});

    // 提取三个点的法向量
    auto src_n0 = src_norms.index({torch::indexing::Slice(), 0}); // [N, 3]
    auto src_n1 = src_norms.index({torch::indexing::Slice(), 1});
    auto src_n2 = src_norms.index({torch::indexing::Slice(), 2});

    auto dst_n0 = dst_norms.index({torch::indexing::Slice(), 0});
    auto dst_n1 = dst_norms.index({torch::indexing::Slice(), 1});
    auto dst_n2 = dst_norms.index({torch::indexing::Slice(), 2});

    // 计算法向量相似度（余弦相似度的绝对值）
    auto src_sim01 = torch::abs((src_n0 * src_n1).sum(-1)); // [N]
    auto src_sim02 = torch::abs((src_n0 * src_n2).sum(-1));
    auto src_sim12 = torch::abs((src_n1 * src_n2).sum(-1));

    auto dst_sim01 = torch::abs((dst_n0 * dst_n1).sum(-1));
    auto dst_sim02 = torch::abs((dst_n0 * dst_n2).sum(-1));
    auto dst_sim12 = torch::abs((dst_n1 * dst_n2).sum(-1));

    // 将6个相似度堆叠 [N, 6]
    auto all_sims = torch::stack({src_sim01, src_sim02, src_sim12,
                                   dst_sim01, dst_sim02, dst_sim12}, -1);

    // 获取每组的最小相似度 [N]
    auto min_sims = std::get<0>(all_sims.min(-1));

    // 过滤：保留最小相似度小于阈值的组
    auto mask = min_sims < threshold;

    auto filtered_cliques = cliques_cpu.index({mask});

    // 将结果移回原始设备（如果原来在XPU/CUDA上）
    if (original_device.is_cuda() || original_device.is_xpu()) {
        filtered_cliques = filtered_cliques.to(original_device);
    }

    return filtered_cliques;
}
