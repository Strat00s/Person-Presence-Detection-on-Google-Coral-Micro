#pragma once
#include <deque>
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"

template<typename T>
class AwaitQueue {
private:
    std::deque<T> buffer;
    SemaphoreHandle_t buf_mux;
    SemaphoreHandle_t await_sem;
    int max_size;

public:
    AwaitQueue(int max_size) {
        this->max_size = max_size;
        buf_mux = xSemaphoreCreateMutex();
        await_sem = xSemaphoreCreateCounting(max_size, 0);
    }
    ~AwaitQueue() {}


    /** @brief Wait for data to be available and retrieve first item
     * 
     * @param item Pointer to which to store the retrieved item
     * @param await_timeout Timeout when waiting for data to become available
     * @param access_timeout Timeout when trying to access the underlying buffer
     * @return 0 on success, 1 otherwise
     */
    int pop(T *item, TickType_t await_timeout = portMAX_DELAY, TickType_t access_timeout = portMAX_DELAY) {
        //wait for data
        if (xSemaphoreTake(await_sem, await_timeout) != pd0)
            return 1;
        
        //try and access the data
        if (xSemaphoreTake(buf_mux, access_timeout) != pd0) {
            xSemaphoreGive(await_sem);
            return 1;
        }
        
        if (!buffer.size()) {
            xSemaphoreGive(buf_mux);
            return 1;
        }
        else {
            *item = buffer.front();
            buffer.pop_front();
        }
        xSemaphoreGive(buf_mux);
        return 0;
    }

    /** @brief 
     * 
     * @param item Item to store in the underlying buffer
     * @param timeout Timeout for accessing the buffer
     * @param replace If queue is full, force replace last item
     * @return 0 on success, 1 otherwise
     */
    int push(T item, TickType_t timeout = portMAX_DELAY, bool replace = 1) {
        if (xSemaphoreTake(buf_mux, timeout) != pd0)
            return 1;
        
        //don't add any items if full
        if (buffer.size() == max_size) {
            if (replace)
                buffer.back() = item;
            xSemaphoreGive(buf_mux);
            return replace;
        }

        buffer.push_back(item);
        xSemaphoreGive(buf_mux);
        xSemaphoreGive(await_sem);
        return 0;
    }

    /** @brief Get size of the underlying buffer
     * 
     * @param size Pointer to which to store the current queue size
     * @param timeout Timeout when accessing the underlying buffer
     * @return 0 on success, 1 otherwise
     */
    int size(size_t *size, TickType_t timeout = portMAX_DELAY){
        if (xSemaphoreTake(buf_mux, timeout) != pd0)
            return 1;

        *size = buffer.size();
        xSemaphoreGive(buf_mux);
        return 0;
    }
};

