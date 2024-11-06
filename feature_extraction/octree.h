#ifndef OCTREE_H
#define OCTREE_H

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class OctreeNode {
public:
    std::array<int, 6> region; // [y_start, y_end, x_start, x_end, z_start, z_end]
    bool isLeaf;
    std::vector<OctreeNode*> children;
    int avg;

    OctreeNode(int y_start, int y_end, int x_start, int x_end, int z_start, int z_end);
    ~OctreeNode();
};
int isHomogeneous(const std::vector<std::vector<std::vector<int>>>& volume,
                  int y_start, int y_end, int x_start, int x_end, int z_start, int z_end, int threshold);

OctreeNode* buildOctree(const std::vector<std::vector<std::vector<int>>>& volume,
                        int y_start, int y_end, int x_start, int x_end, int z_start, int z_end,
                        int maxDepth, int threshold);

void saveMatAsTensor(const cv::Mat& mat, const std::string& file_path);

void retriveImage(OctreeNode* node, cv::Mat& reconstruction, int z_target);

std::vector<std::vector<std::vector<int>>> loadVolumeFromImages(const std::string& folderPath, int height, int width, int depth);
#endif
