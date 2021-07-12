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

#include "tensorflow/lite/micro/examples/mobilenet/main_functions.h"
#include "CImg.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
  cimg_usage("A simple image classification.\n\nUsage : mobilenet -i image");
  const char *const file_i = cimg_option("-i",(char*)0,"Input image");
  cimg_library::CImg<> img;

  if (!file_i)
    return 1;
  else img.assign(file_i);

  int w = 128;
  int h = 128;
  int c = 3;
  img.norm().resize(w, h);

  float image_data[w * h * c];
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      for (int k = 0; k < c; ++k)
        {
          const float val = img(j, i, k);
          image_data[i * w * c + j *c + k] = val / 128.0f - 1.0f;
        }

  setup();
  while (true) {
    input_data = image_data;
    loop();
  }
  return 0;
}
