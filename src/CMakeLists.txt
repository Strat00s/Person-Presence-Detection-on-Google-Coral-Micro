# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


add_executable_m7(my_project
    main.cpp
    M4_EXECUTABLE
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
