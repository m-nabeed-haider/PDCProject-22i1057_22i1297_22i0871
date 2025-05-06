#include "sssp.h"
#include <cstring>
#include <stdexcept>

Graph::~Graph() {
    delete[] xadj;
    delete[] adjncy;
    delete[] weight;
}

SSSPTree::SSSPTree(size_t n) : size(n) {
    Dist = new float[n];
    Parent = new int[n];
}

SSSPTree::~SSSPTree() {
    delete[] Dist;
    delete[] Parent;
}