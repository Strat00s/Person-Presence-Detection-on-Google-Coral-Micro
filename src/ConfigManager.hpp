/**
 * @file ConfigManager.hpp
 * @author Lukáš Baštýř (l.bastyr@seznam.cz)
 * @brief Manager for global configuration management.
 *
 * @copyright Copyright (c) 2024
 * 
 */


#include <vector>
#include <string>

#include "libs/base/filesystem.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"

using namespace std;


#define DEFAULT_IMAGE_SIZE   320
#define DEFAULT_MASK_SIZE    32
#define DEFAULT_ROTATION     0
#define DEFAULT_DET_THRESH   25
#define DEFAULT_IOU_THRESH   25
#define DEFAULT_FP_CHANGE    0
#define DEFAULT_FP_COUNT     0
#define DEFAULT_MASK_THRESH  70
#define DEFAULT_MIN_WIDTH    0
#define DEFAULT_MIN_HEIGHT   0
#define DEFAULT_MIN_AS_AREA  false


typedef struct{
    int mask_size;
    int mask_thresh;
    int rotation;
    int det_thresh;
    int iou_thresh;
    int fp_change;
    int fp_count;
    int min_width;
    int min_height;
    bool min_as_area;
    std::vector<uint8_t> mask;
} cfg_struct_t;


class ConfigManager {
private:
    /* data */
    cfg_struct_t config = {0};
    SemaphoreHandle_t *config_mux;

public:
    ConfigManager(SemaphoreHandle_t *config_mux) {
        this->config_mux = config_mux;
    } 
    //~ConfigManager();

    /** @brief Initialize config do default values.*/
    void init() {
        xSemaphoreTake(*config_mux, portMAX_DELAY);
        config.mask_size    = DEFAULT_MASK_SIZE;
        config.mask_thresh  = DEFAULT_MASK_THRESH;
        config.rotation     = DEFAULT_ROTATION;
        config.det_thresh   = DEFAULT_DET_THRESH;
        config.iou_thresh   = DEFAULT_IOU_THRESH;
        config.fp_change    = DEFAULT_FP_CHANGE;
        config.fp_count     = DEFAULT_FP_COUNT;
        config.min_width    = DEFAULT_MIN_WIDTH;
        config.min_height   = DEFAULT_MIN_HEIGHT;
        config.min_as_area  = DEFAULT_MIN_AS_AREA;
        config.mask.clear();
        config.mask.resize(config.mask_size * config.mask_size, 0);
        xSemaphoreGive(*config_mux);
    }

    /** @brief Get const pointer to the underlying config. Not thread safe!*/
    const cfg_struct_t* getConfig() {
        return &config;
    }

    /** @brief Save config to file.
     * 
     * @param path Path to the config file
     * @return True on success, false otherwise.
     */
    bool save(const char *path) {
        vector<uint8_t> raw = toBinary();
        return LfsWriteFile(path, raw.data(), raw.size());
    }

    /** @brief Update and save config.
     * 
     * @param new_config New configuration to set and save
     * @param path Path to config file where to store new config
     * @return True on success, false otherwise.
     */
    bool update(const cfg_struct_t &new_config, const char *path) {
        xSemaphoreTake(*config_mux, portMAX_DELAY);
        config = new_config;
        xSemaphoreGive(*config_mux);

        vector<uint8_t> raw = toBinary();
        return LfsWriteFile(path, raw.data(), raw.size());
    }

    /** @brief Load config from file in path.
     *
     * @param path Path to the config file
     * @return 0 on success.
     * 1 if reading failed.
     * 2 if conversion from binary to config failed.
     */
    int load(const char *path) {
        vector<uint8_t> raw;
        
        if (!LfsReadFile(path, &raw))
            return 1;
        
        return fromBinary(raw) ? 0 : 1;
    }

    /** @brief Convert config to binary format.*/
    vector<uint8_t> toBinary() {
        vector<uint8_t> raw_data;

        xSemaphoreTake(*config_mux, portMAX_DELAY);
        raw_data.push_back(config.mask_size);
        raw_data.push_back(config.mask_thresh);
        raw_data.push_back(config.rotation);
        raw_data.push_back(config.det_thresh);
        raw_data.push_back(config.iou_thresh);
        raw_data.push_back(config.fp_change);
        raw_data.push_back(config.fp_count);
        raw_data.push_back(config.min_width);
        raw_data.push_back(config.min_height);
        raw_data.push_back(config.min_as_area);
        //raw_data.insert(raw_data.end(), config.mask.begin(), config.mask.end());
        for (size_t i = 0; i < config.mask_size * config.mask_size; i++) {
            raw_data.push_back(config.mask[i]);
        }
        xSemaphoreGive(*config_mux);

        return raw_data;
    }
    
    /** @brief Load config from binary format.*/
    bool fromBinary(vector<uint8_t> raw) {
        //raw config is smaller than minimun size
        if (raw.size() < 14)
            return false;
    
        size_t mask_size = raw[0] * raw[0];

        //reported mask size does not match the raw config size
        if (raw.size() != 10 + mask_size)
            return false;

        //copy the data
        xSemaphoreTake(*config_mux, portMAX_DELAY);
        config.mask_size   = raw[0];
        config.mask_thresh = raw[1];
        config.rotation    = raw[2];
        config.det_thresh  = raw[3];
        config.iou_thresh  = raw[4];
        config.fp_change   = raw[5];
        config.fp_count    = raw[6];
        config.min_width   = raw[7];
        config.min_height  = raw[8];
        config.min_as_area = raw[9];
        config.mask        = vector<uint8>(raw.begin() + 10, raw.begin() + 10 + mask_size);
        xSemaphoreGive(*config_mux);


        return true;
    }
};

