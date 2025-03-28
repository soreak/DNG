#pragma once
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#ifdef _WIN32
    #include <io.h>  // Windows 等效头文件
#else
    #include <unistd.h>
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

#include "Node.h"

#ifndef READ_H
    #define READ_H
#endif 

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f;
#ifdef _WIN32
    fopen_s(&f, fname, "rb");  // 使用 fopen_s 替代 fopen
#else
    f = fopen(fname, "rb");
#endif
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);

#ifdef _WIN32
    struct _stat64i32 st;  // 使用 Windows 的 _stat64i32
    _fstat64i32(_fileno(f), &st);  // 使用 _fstat64i32 替代 fstat
#else
    struct stat st;
    fstat(fileno(f), &st);
#endif

    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}


int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    using namespace std::chrono;
    static auto start = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    return duration<double>(now - start).count();
}

void insertToNodes(float* xt, size_t d_out, size_t n_out, std::vector<Node>& nodes) {
    for (size_t i = 0; i < n_out; i++) {
        Node new_node(d_out, i);  
        for (size_t j = 0; j < d_out; j++) {
            new_node.features[j] = xt[i * d_out + j];  // xt[i] 是第i个向量的起始位置
        }
        nodes.push_back(new_node); 
    }
}
