/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/mobilenet/output_handler.h"
#include <iostream>

void HandleOutput(tflite::ErrorReporter* error_reporter, int best_label,
                  float max_value) {
//  TF_LITE_REPORT_ERROR(error_reporter, "label: %d, value: %f\n",
//                       best_label,
//                       static_cast<double>(max_value));
  std::cout << "label: " << best_label << " value: " << max_value << "\n";
}
