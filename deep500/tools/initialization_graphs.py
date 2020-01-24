import onnx
from typing import List, Tuple, Optional
from deep500.utils.onnx_interop.onnx_objects import AttributeType
from deep500.utils.onnx_interop.onnx_objects import Operation


def constant_of_shape(shape: List[int], value, name: str, dtype: AttributeType):
    """Produces Graph producing constant output Tensor of given shape and value. name MUST match weight input"""
    shape_tensor = onnx.helper.make_tensor("x", AttributeType.INTS.value, [len(shape)], shape)
    const_tensor = onnx.helper.make_tensor("value", dtype.value, [1], [value])

    const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['x'],
        value=shape_tensor,
    )

    upscale_node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['x'],
        outputs=['y'],
        value=const_tensor,
    )

    out = onnx.helper.make_tensor_value_info('y', elem_type=dtype.value, shape=shape)
    graph = onnx.helper.make_graph(nodes=[const_node, upscale_node], name=name, inputs=[], outputs=[out])

    return graph


def random_uniform(shape: List[int], range: Tuple[float], name: str, dtype: AttributeType, seed=None):
    """Produces Graph producing Uniform random output Tensor of given shape and value range.
    name MUST match weight input"""

    if seed is None:
        upscale_node = onnx.helper.make_node(
            'RandomUniform',
            inputs=[],
            outputs=['y'],
            shape=shape,
            high=range[1],
            low=range[0]
        )
    else:
        upscale_node = onnx.helper.make_node(
            'RandomUniform',
            inputs=[],
            outputs=['y'],
            shape=shape,
            high=range[1],
            low=range[0],
            seed=seed
        )

    out = onnx.helper.make_tensor_value_info('y', elem_type=dtype.value, shape=shape)
    graph = onnx.helper.make_graph(nodes=[upscale_node], name=name, inputs=[], outputs=[out])

    return graph


def random_normal(shape: List[int], mean_std: Tuple[float], name: str, dtype: AttributeType, seed: Optional[float]):
    """Produces Graph producing Normal random output Tensor of given shape with mean and std as given in args.
    name MUST match weight input"""

    if seed is None:
        upscale_node = onnx.helper.make_node(
            'RandomNormal',
            inputs=[],
            outputs=['y'],
            shape=shape,
            scale=mean_std[1],
            mean=mean_std[0]
        )
    else:
        upscale_node = onnx.helper.make_node(
            'RandomNormal',
            inputs=[],
            outputs=['y'],
            shape=shape,
            scale=mean_std[1],
            mean=mean_std[0],
            seed=seed
        )

    out = onnx.helper.make_tensor_value_info('y', elem_type=dtype.value, shape=shape)
    graph = onnx.helper.make_graph(nodes=[upscale_node], name=name, inputs=[], outputs=[out])

    return graph


# g = constant_of_shape([2, 5, 5], 3.1415, 'input.0.weight', tensor_type=AttributeType.FLOAT)
# g = random_uniform([2, 5, 5], 3.1415, 'input.0.weight', tensor_type=AttributeType.FLOAT)

g = random_normal([2, 5, 5], (0.0,1.0), 'input.0.weight', dtype=AttributeType.FLOAT, seed=69.0)

model = onnx.helper.make_model(g)
onnx.checker.check_model(model)
onnx.save_model(model, 'init.onnx')



from deep500.frameworks import pytorch as d5fw
import deep500 as d5

path = 'C:/Users/fdoki/PycharmProjects/my_d500/deep500/tools/init.onnx'
init = d5.parser.load_and_parse_model(path)
executorD = d5fw.from_model(init, device=d5.CPUDevice())
executorD.inference(executorD.visitor.initializers)