#include <iostream>
#include <string>
#include "octree.h" // Include your octree header file

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <z_target>" << std::endl;
        return 1;
    }

    int z_target = std::stoi(argv[1]);

    std::string folderPath = "../../archive/data";
    int height = 369;
    int width = 369;
    int depth = 71;
    auto volume = loadVolumeFromImages(folderPath, width, height, depth);

    /*
    int offset = 25;
    int shift = 12;
    */
    int maxDepth = 9;
    int threshold = 0;
    OctreeNode* root = buildOctree(&volume, 0, volume.size(), 0, volume[0].size(), 0, volume[0][0].size(), maxDepth, threshold);
    /* OctreeNode* root = buildOctree(&volume, offset, offset+shift, offset, offset+shift, offset, offset+shift, maxDepth, threshold);

    for (int i = 0; i < shift; i++)
        for (int j = 0; j < shift; j++)
            for (int l = 0; l < shift; l++)
                std::cout<<volume[offset][offset][offset];
    */

    cv::Mat reconstruction(height, width, CV_32S, cv::Scalar(0));
    
    saveMatAsTensor(reconstruction, "reconstruction.pt");

    retriveImage(root, reconstruction, z_target);

    cv::Mat outputImage;
    reconstruction.convertTo(outputImage, CV_8U);

    cv::imwrite("test.png", outputImage);

    delete root;
    return 0;
}
