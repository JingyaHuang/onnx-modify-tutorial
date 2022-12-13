import numpy as np
import onnx
import onnxruntime as ort

model_name = 'merged'
model_path =  model_name + ".onnx"

onnx_dtype = onnx.TensorProto.INT32
onnx_control_dtype = onnx.TensorProto.BOOL
shape = [1]

onnx_input0 = onnx.helper.make_tensor_value_info("INPUT0", onnx_dtype, shape)
onnx_input1 = onnx.helper.make_tensor_value_info("INPUT1", onnx_dtype, shape)
onnx_input2 = onnx.helper.make_tensor_value_info("INPUT2", onnx_dtype, shape)

onnx_operation = onnx.helper.make_tensor_value_info("OP", onnx_control_dtype,
                                                    shape)

onnx_output = onnx.helper.make_tensor_value_info("OUTPUT", onnx_dtype, shape)
add_1 = onnx.helper.make_node('Add', ['INPUT0', 'INPUT1'], ['OUTPUT0'])
add_2 = onnx.helper.make_node('Add', ['INPUT2', 'OUTPUT0'], ['OUTPUT1'])
add_3 = onnx.helper.make_node('Add', ['OUTPUT1', 'OUTPUT1'], ['OUTPUT'])
const = onnx.helper.make_node('Constant', [], ['OUTPUT'], value=onnx.numpy_helper.from_array(np.array([1]).astype(np.int32)))

then_body = onnx.helper.make_graph([add_1, add_2, add_3], 'then', [],
                                   [onnx_output])
else_body = onnx.helper.make_graph([const], 'else', [],
                                   [onnx_output])

if_node = onnx.helper.make_node('If',
                                inputs=['OP'],
                                outputs=['OUTPUT'],
                                then_branch=then_body,
                                else_branch=else_body)

graph_proto = onnx.helper.make_graph([if_node], model_name,
                                     [onnx_input0, onnx_input1, onnx_input2, onnx_operation], [onnx_output])

model_def = onnx.helper.make_model(graph_proto, producer_name="if-example")

# Check spec
onnx.save(model_def, model_path)

# Inference with ONNX Runtime
input_data = {
    'INPUT0': np.asarray([1], dtype=np.int32),
    'INPUT1': np.asarray([1], dtype=np.int32),
    'INPUT2': np.asarray([5], dtype=np.int32),
    'OP': np.asarray([True], dtype=np.bool)
}

onnx_sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
pred = onnx_sess.run(None, input_data)
print(pred)