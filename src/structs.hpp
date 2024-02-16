#pragma once

#include <vector>
#include <stdint.h>

typedef std::vector<uint8_t> imageVector_t;

typedef struct {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int score;
    int type;
} bbox_t;

typedef struct cfg_struct{
    int mask_size;
    std::vector<uint8_t> mask;
    int rotation;
    int det_thresh;
    int iou_thresh;
    int fp_change;
    int fp_count;
    int jpeg_quality;
} cfg_struct_t;
