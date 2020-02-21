import tensorflow as tf

"tf2.0 compat tf1.0 version"

model_file = " "
input_array = [" "]
output_array = [" "]
save_file = model_file.split(".")[0] + ".tflite"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(model_file, input_array , output_array)
tflite_model = converter.convert()
open(save_file ,"wb").write(tflite_model)

save_fp16_file = model_file.split(".")[0]+"_fp16"+".tflite"
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
open(save_fp16_file,"wb").write(tflite_fp16_model)

