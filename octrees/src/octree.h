#ifndef OCTREE_H
#define OCTREE_H

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

class OctreeNode {
public:
    std::array<int, 6> region; // [x_start, x_end, y_start, y_end, z_start, z_end]
    bool isLeaf;
    std::vector<OctreeNode*> children;
    double value;

    OctreeNode(int x_start, int x_end, int y_start, int y_end, int z_start, int z_end);
    ~OctreeNode();

    void pool(const std::vector<std::vector<std::vector<int>>>* volume, double *values,
              int x_start, int x_end, int y_start, int y_end, int z_start, int z_end);
};

OctreeNode* buildOctree(const std::vector<std::vector<std::vector<int>>> *volume,
                        int x_start, int x_end, int y_start, int y_end, int z_start, int z_end,
                        int maxDepth, int threshold);

void retriveImage(OctreeNode* node, cv::Mat& reconstruction, int z_target);

std::vector<std::vector<std::vector<int>>> loadVolumeFromImages(const std::string& folderPath, int height, int width, int depth);
std::vector<std::vector<std::vector<int>>> loadVolumeFromArray(int *array, int height, int width, int depth);
#endif
