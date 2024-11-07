#include "octree.h"
#include <iostream>
#include <cassert>


int main() {
    std::string folderPath = "../archive/data";
    int height = 369;
    int width = 369;
    int depth = 155;
    auto volume = loadVolumeFromImages(folderPath, width, height, depth);

    int maxDepth = 25;
    int threshold = 50;
    OctreeNode* root = buildOctree(&volume, 0, volume.size(), 0, volume[0].size(), 0, volume[0][0].size(), maxDepth, threshold);

    int z_target = 50;
    cv::Mat reconstruction(height, width, CV_32S, cv::Scalar(0));
    
    saveMatAsTensor(reconstruction, "reconstruction.pt");

    retriveImage(root, reconstruction, z_target);

    cv::Mat outputImage;
    reconstruction.convertTo(outputImage, CV_8U);

    cv::imwrite("test.png", outputImage);

    delete root;
    return 0;
}