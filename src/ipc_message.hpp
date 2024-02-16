#pragma once

#include <vector>
#include "libs/base/ipc_message_buffer.h"
#include "structs.hpp"

//message type
//pointer to data
    //copy and then use them
//gate number to use when accessing the underlying data

//data were pushed, go read them

//bbox vector
//M7 -> M4

//image
//M4 -> M7

//config
//M4 -> M7
//M4 -> M4

//Send image size M7 -> M4
//configure M4
//Send ACK M4 -> M7


enum class IpcMsgType : uint8_t {
    CONFIG,       //M7 -> M4
    ACK,
};

enum IpcMsgGate {
    PRINT,
    IMAGE_BEFORE,
    IMAGE_AFTER,
    JPEG,
    BBOX,
    OTHER
};


//I am too lazy to do it properly so this will have to do for now
struct IpcMessageStruct {
    IpcMsgType type; //data type
    image_vector_t *image_b;
    image_vector_t *image_a;
    image_vector_t *jpeg;
    bbox_vector_t *bboxes;
    int image_size;
    uint8_t gate; //gate for thredsafe access to data
} __attribute__((packed));

static_assert(sizeof(IpcMessageStruct) <= coralmicro::kIpcMessageBufferDataSize);
