#include "..\Read.h"


int main() {
    double t0 = elapsed();

    // this is typically the fastest one.
    const char* index_key = "IVF4096,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";


    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        
        std::vector<Node> nodes;

        insertToNodes(xt,d,nt,nodes);

        // 输出第一个向量的内容
        printf("First vector (size %ld):\n", d);
        for (size_t i = 0; i < 3; i++) {
           nodes[i].print();
        }
        printf("%.3f",nodes[0].computeDistance(nodes[1]));
        
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        
        delete[] xt;
    }
}