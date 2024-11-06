#include "octree.h"
#include <iostream>
#include <cassert>
#include <filesystem>

OctreeNode::OctreeNode(int y_start, int y_end, int x_start, int x_end, int z_start, int z_end)
    : region({ y_start, y_end, x_start, x_end, z_start, z_end }), isLeaf(true), avg(0) {}

OctreeNode::~OctreeNode() {
    for (auto* child : children) {
        delete child;
    }
}


int isHomogeneous(const std::vector<std::vector<std::vector<int>>>& volume,
                  int y_start, int y_end, int x_start, int x_end, int z_start, int z_end, int threshold) {
    if (y_start >= y_end || x_start >= x_end || z_start >= z_end) return -1;

    int min_val = volume[y_start][x_start][z_start];
    int max_val = min_val;

    int sum = 0;
    int nr = 0;

    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            int value = volume[y][x][z_start];
            nr++;
            sum += value;
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
            if ((max_val - min_val) > threshold) return -1;
        }
    }
    if (nr == 0) return -1;
    return sum / nr;
}

OctreeNode* buildOctree(const std::vector<std::vector<std::vector<int>>>& volume,
                        int y_start, int y_end, int x_start, int x_end, int z_start, int z_end,
                        int maxDepth, int threshold) {
    OctreeNode* node = new OctreeNode(y_start, y_end, x_start, x_end, z_start, z_end);

    if ((x_end - x_start <= 1) && (y_end - y_start <= 1) && (z_end - z_start <= 1)) {
        node->isLeaf = true;
        node->avg = volume[y_start][x_start][z_start];
        return node;
    }

    if (z_end - z_start > 1) {
        int z_mid = (z_start + z_end) / 2;
        node->children.reserve(2);
        node->children.push_back(buildOctree(volume, y_start, y_end, x_start, x_end, z_start, z_mid, maxDepth - 1, threshold));
        node->children.push_back(buildOctree(volume, y_start, y_end, x_start, x_end, z_mid, z_end, maxDepth - 1, threshold));
        node->isLeaf = false;
        return node;
    }

    node->avg = isHomogeneous(volume, y_start, y_end, x_start, x_end, z_start, z_end, threshold);
    if (maxDepth == 0 || node->avg != -1) {
        node->isLeaf = true;
        return node;
    }

    node->isLeaf = false;
    int x_mid = (x_start + x_end) / 2;
    int y_mid = (y_start + y_end) / 2;
    node->children.reserve(4);
    node->children.push_back(buildOctree(volume, y_start, y_mid, x_start, x_mid, z_start, z_end, maxDepth - 1, threshold));
    node->children.push_back(buildOctree(volume, y_mid, y_end, x_start, x_mid, z_start, z_end, maxDepth - 1, threshold));
    node->children.push_back(buildOctree(volume, y_start, y_mid, x_mid, x_end, z_start, z_end, maxDepth - 1, threshold));
    node->children.push_back(buildOctree(volume, y_mid, y_end, x_mid, x_end, z_start, z_end, maxDepth - 1, threshold));

    return node;
}

void saveMatAsTensor(const cv::Mat& mat, const std::string& file_path) {
    
    torch::Tensor tensor = torch::from_blob(mat.data, {mat.rows, mat.cols}, torch::kInt32);
    tensor = tensor.clone();
    torch::save(tensor, file_path);
}

void retriveImage(OctreeNode* node, cv::Mat& reconstruction, int z_target) {
    if (!node) return;
    if (!(node->region[4] <= z_target && z_target < node->region[5])) return;

    if (node->isLeaf) {
        for (int y = node->region[0]; y < node->region[1]; ++y) {
            for (int x = node->region[2]; x < node->region[3]; ++x) {
                reconstruction.at<int>(y, x) = node->avg;
            }
        }
        return;
    }

    for (OctreeNode* child : node->children) {
        retriveImage(child, reconstruction, z_target);
    }
}

namespace fs = std::filesystem;

std::vector<std::vector<std::vector<int>>> loadVolumeFromImages(const std::string& folderPath, int height, int width, int depth) {
    std::vector<std::vector<std::vector<int>>> volume(height, std::vector<std::vector<int>>(width, std::vector<int>(depth, 0)));

    // Collect filenames for volume_1 only
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        const std::string filename = entry.path().filename().string();
        
        // Check if the file matches "volume_1" and has the correct format
        if (filename.find("volume_1_slice_") != std::string::npos && filename.find(".png") != std::string::npos) {
            filenames.push_back(entry.path().string());
        }
    }

    // Sort filenames based on slice number extracted from filename
    std::sort(filenames.begin(), filenames.end(), [](const std::string& a, const std::string& b) {
        int slice_a = std::stoi(a.substr(a.find("slice_") + 6, a.find(".png") - (a.find("slice_") + 6)));
        int slice_b = std::stoi(b.substr(b.find("slice_") + 6, b.find(".png") - (b.find("slice_") + 6)));
        return slice_a < slice_b;
    });

    int sliceIndex = 0;
    for (const auto& filename : filenames) {
        if (sliceIndex >= depth) break;
        cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error loading image: " << filename << std::endl;
            continue;
        }

        // Ensure the image dimensions match the expected width and height
        if (img.rows != height || img.cols != width) {
            std::cerr << "Image dimensions do not match expected dimensions: " << filename << std::endl;
            std::cerr << img.rows << img.cols << std::endl;
            continue;
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                volume[i][j][sliceIndex] = img.at<uchar>(i, j);
            }
        }
        sliceIndex++;
    }

    if (sliceIndex < depth) {
        std::cerr << "Warning: Expected " << depth << " images, but only found " << sliceIndex << " images for volume_1." << std::endl;
    }

    return volume;
}