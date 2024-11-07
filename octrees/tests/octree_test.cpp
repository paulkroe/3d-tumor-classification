#include "../src/octree.h"  // Adjust path based on octree.h's actual location
#include <iostream>
#include <cassert>

int main() {
    std::string folderPath = "../../archive/data";
    int height = 369;
    int width = 369;
    int depth = 155;
    auto volume = loadVolumeFromImages(folderPath, width, height, depth);

    int maxDepth = 8;
    int threshold = 0;
    OctreeNode* root = buildOctree(&volume, 0, volume.size(), 0, volume[0].size(), 0, volume[0][0].size(), maxDepth, threshold);

    for (int z = 0; z < depth; z++) {
        cv::Mat reconstruction(height, width, CV_32S, cv::Scalar(0));
        retriveImage(root, reconstruction, z);
        cv::Mat outputImage;
        reconstruction.convertTo(outputImage, CV_8U);

        // Verify the reconstruction matches the original volume slice at depth z
        bool match = true;
        for (int x = 0; x < width && match; x++) {
            for (int y = 0; y < height && match; y++) {
                if (outputImage.at<int>(y, x) != volume[x][y][z]) {
                    break;
                    match = false;
                }
            }
        }
        assert(match && "Reconstructed image does not match the original slice at depth z.");
        std::cout<<"Passed layer: "<<z<<"\n";
    }

    delete root;
    return 0;
}
