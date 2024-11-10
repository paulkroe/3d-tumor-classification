#include <fstream>
#include <vector>
#include <../src/octree.h>

// Example octree traversal - replace this with your actual traversal function
void ExampleOctreeTraversal() {
    std::vector<OctreeNode> octreeData;
    OctreeNode node = OctreeNode(0, 5, 0, 5, 0, 5);
    node.value = 1;
    octreeData.push_back(node);
    // saveOctree(octreeData, "octree_data.csv");
}

// called by a thread to collect the data in a subtree
void collectData(OctreeNode* octree, std::stringstream& buffer) {
    if(!octree) return;
    if(octree->isLeaf) {
        for (int i = 0; i < 6; i++)
            buffer << octree->region[i] << ",";
        buffer << octree->value << "\n";
    } else {
        for (auto& child : octree->children) {
            collectData(child, buffer);
        }
    }
}

void exportOctreeToCSV(OctreeNode* root, const std::string& filename) {
    std::vector<std::thread> threads;
    std::vector<std::stringstream> buffers(std::thread::hardware_concurrency());

    for (size_t i = 0; i < buffers.size() && i < 8; i++) {
        threads.emplace_back([&, i]() {
            collectData(root->children[i], buffers[i]);
        });
    }

    for (auto& t:threads) t.join();

    std::ofstream file(filename);
    file << "x_start,x_end,y_start,y_end,z_start,z_end,value\n";

    for (auto& buffer : buffers) {
        file<<buffer.str();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]  << " <volume>"  << " <max_depth>" << " <threshold>" << std::endl;
        return 1;
    }


    std::string folderPath = "../../archive/data/volume_" + std::string(argv[1]);
    int height = 369;
    int width = 369;
    int depth = 155;
    auto volume = loadVolumeFromImages(folderPath, width, height, depth);
    int maxDepth = std::stoi(argv[2]);
    int threshold = std::stoi(argv[3]);
    OctreeNode* root = buildOctree(&volume, 0, volume.size(), 0, volume[0].size(), 0, volume[0][0].size(), maxDepth, threshold);
    exportOctreeToCSV(root, "octree_data.csv");
}