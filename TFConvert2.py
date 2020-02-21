from PIL import Image
import tensorflow as tf
import os
import numpy as np
import pdb

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f .endswith('.jpg')]

imlist=get_imlist(r"./dataset")
#def representative_dataset_gen():
#    for i in range(10):
#        batch_list=[]
#        for j in range(1):
#            im=Image.open(imlist[i*30+j])
#            #im=im.resize((769,769))
#            inputnp=np.array(im)
#            inputnp=inputnp.astype(np.float32)
#            #input_list=list(im.getdata())
#            #input_list_float=[float(x) for x in input_list]
#            inputnp -=127.5
#            inputnp /=128.0
#            input_list=inputnp.tolist()
#            #inputarray=inputnp[np.newaxis,:,:,:]
#            batch_list.append(input_list)
#        batch_np=np.array(batch_list)
#        batch_np=batch_np.astype(np.float32)
#        print(batch_np.shape)
#        yield [batch_np]

def representative_dataset_gen():
        for _ in range(10):
            im=Image.open(imlist[_])
            im=im.resize((160,160))
            inputnp=np.array(im)
            inputnp=inputnp.astype(np.float32)
            inputnp -= 127.5
            inputnp /= 128.0
            inputarray = inputnp[np.newaxis,:,:,:]
            yield[inputarray]
#converter = tf.lite.TFLiteConverter.from_frozen_graph("frozen_graph.pb",input_arrays=["input_0"],output_arrays=["Cast"])
converter=tf.lite.TFLiteConverter.from_saved_model('inception_v3')
#model=tf.saved_model.load('inception_v3')
#concrete_func = model.signatures[
#          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#concrete_func = model.signatures['serving_default']
#concrete_func.inputs[0].set_shape([1, 299, 299, 3])
#converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types=[tf.float16]
converter.representative_dataset = representative_dataset_gen
#pdb.set_trace()
tflite_quant_model = converter.convert()
open("inception_v3.tflite","wb").write(tflite_quant_model)

