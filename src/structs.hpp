#pragma once

#include <vector>
#include <variant>
#include <stdint.h>


typedef std::vector<uint8_t> image_vector_t;

typedef struct {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int score;
    int type;
} bbox_t;

typedef std::vector<bbox_t> bbox_vector_t;

typedef struct{
    int mask_size;
    int mask_thresh;
    std::vector<uint8_t> mask;
    int rotation;
    int det_thresh;
    int iou_thresh;
    int fp_change;
    int fp_count;
    int jpeg_quality;
    int min_width;
    int min_height;
    bool min_as_area;
} cfg_struct_t;

typedef struct {
    int status;
    image_vector_t image;
    bbox_vector_t bboxes;
} web_task_msg_t;
