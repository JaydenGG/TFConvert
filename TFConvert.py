from PIL import Image
import tensorflow as tf
import os
import numpy as np

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f .endswith('.jpg')]

imlist=get_imlist(r"./dataset")

def representative_dataset_gen():
    for _ in range(10):
        im=Image.open(imlist[_])
        im=im.resize((769,769))
        inputnp=np.array(im)
        inputnp=inputnp.astype(np.float32)
        inputnp -=127.5
        inputnp /=128.0
        inputarray=inputnp[np.newaxis,:,:,:]
        print(inputarray.shape)
        yield [inputarray]

converter = tf.lite.TFLiteConverter.from_frozen_graph("frozen_graph.pb",input_arrays=["input_0"],output_arrays=["Cast"])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types=[tf.lite.constants.FLOAT16]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
open("model_weight_latency.tflite","wb").write(tflite_quant_model)

