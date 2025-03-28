#pragma once
#include <stdexcept>
#include <math.h>
#include <vector>


class L2Float {
public:
    inline static float compare(const void *pVect1v, const void *pVect2v, size_t qty_ptr) {
        if (qty_ptr == 0) {
            return 0.0f;
        }

        float *a = (float *) pVect1v;
        float *b = (float *) pVect2v;
        size_t size = qty_ptr;

        float diff0, diff1, diff2, diff3;
        const float* last = a + size;
        const float* unroll_group = last - 3;

        float result = 0;
        while (a < unroll_group) {
            diff0 = a[0] - b[0];
            diff1 = a[1] - b[1];
            diff2 = a[2] - b[2];
            diff3 = a[3] - b[3];
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
            b += 4;
        }

        // Process last 0-3 elements
        while (a < last) {
            diff0 = *a++ - *b++;
            result += diff0 * diff0;
        }

        return result;
    }
};
