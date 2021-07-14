#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import sys
import PIL as p
import PIL.Image

model_path = "model.tflite"
image_path = "keyboard.jpg"
if len(sys.argv) > 1:
  model_path = sys.argv[1]
if len(sys.argv) > 2:
  image_path = sys.argv[2]

img = PIL.Image.open(image_path)
img = img.resize((128,128))

interpreter = tf.lite.Interpreter(model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

for input_data in input_details:
  input_shape = input_data["shape"]
  input_dtype = input_data["dtype"]
  input_index = input_data["index"]

  input_data = np.array(img, input_dtype).reshape(input_shape)
  interpreter.set_tensor(input_index, input_data)

interpreter.invoke()

output_details = interpreter.get_output_details()

output_tensor = interpreter.get_tensor(output_details[0]["index"])

print(output_tensor)
print("label: ", np.argmax(output_tensor), "value: ", np.max(output_tensor))

