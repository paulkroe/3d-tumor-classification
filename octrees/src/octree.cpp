#include "octree.h"
#include <iostream>
#include <cassert>
#include <filesystem>

OctreeNode::OctreeNode(int x_start, int x_end, int y_start, int y_end, int z_start, int z_end)
    : region({ x_start, x_end, y_start, y_end, z_start, z_end }), isLeaf(false), value(0) {}

OctreeNode::~OctreeNode() {
    for (auto* child : children) {
        delete child;
    }
}

void pool(const std::vector<std::vector<std::vector<int>>>* volume, double *values,
          int x_start, int x_end, int y_start, int y_end, int z_start, int z_end) {
    
    values[0] = 0;
    values[1] = (*volume)[x_start][y_start][z_start];
    values[2] = (*volume)[x_start][y_start][z_start];
    int value;
    
    for (int x = x_start; x < x_end; ++x) {
        for (int y = y_start; y < y_end; ++y) {
            for (int z = z_start; z < z_end; ++z) {
                value = (*volume)[x][y][z];
                values[0] += value;
                if (values[1] > value) values[1] = value;
                if (values[2] < value) values[2] = value;
            }
        }
    }
    values[0] /= (x_end - x_start) * (y_end - y_start) * (z_end - z_start);
}

OctreeNode* buildOctree(const std::vector<std::vector<std::vector<int>>> *volume,
                        int x_start, int x_end, int y_start, int y_end, int z_start, int z_end,
                        int maxDepth, int threshold) {

    double values[3];
    OctreeNode* node = new OctreeNode(x_start, x_end, y_start, y_end, z_start, z_end);

    pool(volume, values, x_start, x_end, y_start, y_end, z_start, z_end);
    node->value = values[0];

    // std::cout<<values[0]<<" "<<values[1]<<" "<<values[2]<<"\n";
    // max - min <= threshold or max depth reached -> no split
    if ((values[2] - values[1] <= threshold) || (maxDepth == 0)) {
        node->isLeaf = true;
        node->value = values[2];
        // std::cout<<"here"<<x_start<<" "<<y_start<<" "<<z_start<<"\n";
    } else { // branch
        int x_mid = (x_start + x_end) / 2;
        int y_mid = (y_start + y_end) / 2;
        int z_mid = (z_start + z_end) / 2;

        node->children.reserve(((x_end - x_start > 1) + 1) *
                               ((y_end - y_start > 1) + 1) *
                               ((z_end - z_start > 1) + 1));

        for (int i = 0; i < 8; i++) {
            int new_x_start = (i & 1) ? x_mid : x_start;
            int new_x_end   = (i & 1) ? x_end : x_mid;

            int new_y_start = (i & 2) ? y_mid : y_start;
            int new_y_end   = (i & 2) ? y_end : y_mid;

            int new_z_start = (i & 4) ? z_mid : z_start;
            int new_z_end   = (i & 4) ? z_end : z_mid;

            if ((new_x_end - new_x_start) > 0 && 
                (new_y_end - new_y_start) > 0 && 
                (new_z_end - new_z_start) > 0)
            {
                node->children.push_back(buildOctree(volume, new_x_start, new_x_end,
                                                     new_y_start, new_y_end,
                                                     new_z_start, new_z_end,
                                                     maxDepth - 1, threshold));
            }
        }
    }

    return node;
}

void retriveImage(OctreeNode* node, cv::Mat& reconstruction, int z_target) {
    if (!node) return;
    if (!(node->region[4] <= z_target && z_target < node->region[5])) return;

    if (node->isLeaf) {
        for (int x = node->region[0]; x < node->region[1]; ++x) {
            for (int y = node->region[2]; y < node->region[3]; ++y) {
                reconstruction.at<int>(x, y) = node->value;
            }
        }
        return;
    }

    for (OctreeNode* child : node->children) {
        retriveImage(child, reconstruction, z_target);
    }
}


std::vector<std::vector<std::vector<int>>> loadVolumeFromArray(int* array, int height, int width, int depth) {
    
    std::vector<std::vector<std::vector<int>>> volume(height, std::vector<std::vector<int>>(width, std::vector<int>(depth, 0)));

    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < depth; ++k) {
                volume[i][j][k] = array[index++];
            }
        }
    }
    return volume;
}

namespace fs = std::filesystem;
std::vector<std::vector<std::vector<int>>> loadVolumeFromImages(const std::string& folderPath, int height, int width, int depth) {
    
    std::vector<std::vector<std::vector<int>>> volume(height, std::vector<std::vector<int>>(width, std::vector<int>(depth, 0)));
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        const std::string filename = entry.path().filename().string();
        filenames.push_back(entry.path().string());
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

        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
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