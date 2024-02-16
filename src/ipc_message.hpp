#pragma once

#include "libs/base/ipc_message_buffer.h"
#include "vector"

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


enum class IpcMsgType : uint8_t {
    BBOX,   //M7 -> M4
    IMAGE,  //M4 -> M7, M4 -> M4
    CONFIG, //M4 -> M7
    ACK,    //either but M4 does not care
    IMAGE_SIZE //M7 -> M4 right after start
};

enum IpcMsgGate {
    PRINT,
    IMAGE,
    CONFIG,
    BBOX,
    OTHER
};


struct IpcMessageStruct {
    IpcMsgType type;       //data type
    std::vector<int> *data; //pointer to data (stored lineary)
    uint8_t item_cnt = 1;   //how many items are stored in the vector
    uint8_t gate;           //gate for thredsafe access to data
} __attribute__((packed));

static_assert(sizeof(IpcMessageStruct) <= coralmicro::kIpcMessageBufferDataSize);
