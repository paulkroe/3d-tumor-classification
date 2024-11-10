#include <iostream>
#include <string>
#include "../src/octree.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <z_target>" << std::endl;
        return 1;
    }

    int z_target = std::stoi(argv[1]);

    std::string folderPath = "../../archive/data/volume_1";
    int height = 369;
    int width = 369;
    int depth = 155;
    auto volume = loadVolumeFromImages(folderPath, width, height, depth);
    int maxDepth = 9;
    int threshold = 0;
    OctreeNode* root = buildOctree(&volume, 0, volume.size(), 0, volume[0].size(), 0, volume[0][0].size(), maxDepth, threshold);

    cv::Mat reconstruction(height, width, CV_32S, cv::Scalar(0));
    
    saveMatAsTensor(reconstruction, "reconstruction.pt");

    retriveImage(root, reconstruction, z_target);

    cv::Mat outputImage;
    reconstruction.convertTo(outputImage, CV_8U);

    cv::imwrite("test.png", outputImage);

    delete root;
    return 0;
}
