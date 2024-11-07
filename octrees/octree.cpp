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
    
    // TODO: what happens if this is not a square
    double values[3];
    OctreeNode* node = new OctreeNode(x_start, x_end, y_start, y_end, z_start, z_end);

    pool(volume, values, x_start, x_end, y_start, y_end, z_start, z_end);
    node->value = values[0];
    
    // max - min < threshold or max depth reached -> no split
    if ((values[2] - values[1] < threshold) || (maxDepth == 0)) {
        node->isLeaf = true;
    } else { // branch
        
        int x_mid = (x_start + x_end) / 2;
        int y_mid = (y_start + y_end) / 2;
        int z_mid = (z_start + z_end) / 2;

        // it might be that we don't need 8 children because the input volume might not be a perfect cube
        int leaf_mask = (z_end - z_start > 1) * 4 + (y_end - y_start > 1) * 2 + (x_end - x_start > 1);
        node->children.reserve(((x_end - x_start > 1) + 1) * ((y_end - y_start > 1) + 1) * ((z_end - z_start > 1) + 1));
        
        for (int i = 0; i < 8; i++) {

            int new_x_start = (i & 1) ? x_mid : x_start;
            int new_x_end = (i & 1) ? x_end : x_mid;
            
            int new_y_start = (i & 2) ? y_mid :y_start;
            int new_y_end = (i & 2) ? y_end : y_mid;

            int new_z_start = (i & 4) ? z_mid :z_start;
            int new_z_end = (i & 4) ? z_end : z_mid;
            if (
                ((i & 4) && !(4 & leaf_mask)) || // should not split in x dimension
                ((i & 2) && !(2 & leaf_mask)) || // should not split in y dimension
                ((i & 1) && !(1 & leaf_mask))    // should not split in z dimension
            )
                continue;

            node->children.push_back(buildOctree(volume, new_x_start, new_x_end, new_y_start, new_y_end, new_z_start, new_z_end, maxDepth - 1, threshold));
        }

    }

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

/*
void isHomogeneous(const std::vector<std::vector<std::vector<int>>>& volume) {
    int x_start = this->region[0];
    int x_end = this->region[1];
    
    int y_start = this->region[2];
    int y_end = this->region[3];

    int z_start = this->region[4];
    int z_end = this->region[5];

    int min_val = volume[y_start][x_start][z_start];
    int max_val = min_val;

    double sum = 0;
    int nr = 0;

    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            int value = volume[y][x][z_start];
            nr++;
            sum += value;
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
            if ((max_val - min_val) > threshold) {
                this->avg = -1;
                this->isLeaf = false;
            }
        }
    }
   
    this->isLeaf = true;
    this->avg sum / nr;
    
}
*/
