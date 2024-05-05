add_executable_m7(my_project
    main.cpp
    DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/web/page.html>/page.html
    ${CMAKE_CURRENT_SOURCE_DIR}/web/responses/ok.txt>/ok.txt
    ${PROJECT_SOURCE_DIR}/models/my_models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite>/model.tflite
)

target_link_libraries(my_project
    libs_base-m7_freertos
    libs_jpeg_m7
    libs_base-m7_http_server
    libs_base-m7_ethernet
)
