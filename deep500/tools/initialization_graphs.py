import onnx
from typing import List, Tuple, Optional
from deep500.utils.onnx_interop.onnx_objects import AttributeType


class InitializationGraph:
    def __init__(self):
        self.nodes = []
        self.outputs = []
        self.intermediates = []
        self.initialization_bindings = {}

    def _get_intermediate(self):
        var = "inter." + str(len(self.intermediates))
        self.intermediates.append(var)
        return var

    def _get_output_name(self, name: str):
        if any(k == name for k in self.initialization_bindings.keys()):
            raise Exception('need unique tensor name')

        output = self._get_intermediate()
        self.initialization_bindings[name] = output

        return output

    def make_graph(self, name=''):
        graph = onnx.helper.make_graph(nodes=self.nodes, name=name, inputs=[], outputs=self.outputs)
        return graph

    def get_bindings(self):
        return self.initialization_bindings

    def add_node(self, init_type: str, *parameters):
        supported_types = {
            'ConstantOfShape': self._constant_of_shape,
            'RandomUniform': self._random_uniform,
            'RandomNormal': self._random_normal
        }

        func = supported_types[init_type]
        func(*parameters)
        return

    def _constant_of_shape(self, shape: List[int], value, name: str, dtype: AttributeType):
        """adds nodes producing constant output Tensor of given shape and value. name MUST match weight name"""

        output = self._get_output_name(name)
        intermediate = self._get_intermediate()

        shape_tensor = onnx.helper.make_tensor('shape', AttributeType.INTS.value, [len(shape)], shape)

        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[intermediate],
            value=shape_tensor,
            name=name + '.node'
        )

        const_tensor = onnx.helper.make_tensor('value', dtype.value, [1], [value])

        upscale_node = onnx.helper.make_node(
            'ConstantOfShape',
            inputs=[intermediate],
            outputs=[output],
            value=const_tensor,
            name=name + '.node'
        )

        out = onnx.helper.make_tensor_value_info(output, elem_type=dtype.value, shape=shape)
        self.nodes += [const_node, upscale_node]
        self.outputs.append(out)
        return

    # todo: revise pytorch visitor concerning seed
    def _random_uniform(self, shape: List[int], range: Tuple[float], name: str, dtype: AttributeType, seed=None):
        """adds nodes producing Uniform random output Tensor of given shape and value range.
        name MUST match weight name"""

        output = self._get_output_name(name)
        self._assure_name_unique(name)

        if seed is None:
            upscale_node = onnx.helper.make_node(
                'RandomUniform',
                inputs=[],
                outputs=[output],
                shape=shape,
                high=range[1],
                low=range[0]
            )
        else:
            upscale_node = onnx.helper.make_node(
                'RandomUniform',
                inputs=[],
                outputs=[output],
                shape=shape,
                high=range[1],
                low=range[0],
                seed=seed
            )

        out = onnx.helper.make_tensor_value_info(output, elem_type=dtype.value, shape=shape)

        self.nodes.append(upscale_node)
        self.outputs.append(out)
        return


    def _random_normal(self, shape: List[int], mean_std: Tuple[float], name: str, dtype: AttributeType, seed=None):
        """adds node producing Normal random output Tensor of given shape with mean and std as given in args.
        name MUST match weight name"""

        output = self._get_output_name(name)

        if seed is None:
            upscale_node = onnx.helper.make_node(
                'RandomNormal',
                inputs=[],
                outputs=[output],
                shape=shape,
                scale=mean_std[1],
                mean=mean_std[0]
            )
        else:
            upscale_node = onnx.helper.make_node(
                'RandomNormal',
                inputs=[],
                outputs=[output],
                shape=shape,
                scale=mean_std[1],
                mean=mean_std[0],
                seed=seed
            )

        out = onnx.helper.make_tensor_value_info(output, elem_type=dtype.value, shape=shape)

        self.nodes.append(upscale_node)
        self.outputs.append(out)
        return

# # example code
# g = InitializationGraph()
#
# g.add_node('ConstantOfShape', [1,2,3], 3.1415, "weight.3", AttributeType.FLOAT)
# g.add_node('RandomNormal', [1,2,3], (0.0,1.0), "weight.1", AttributeType.FLOAT)
# g.add_node('RandomUniform', [1,2,3], (5.0,0.0), 'weight.0', AttributeType.FLOAT)
#
# graph = g.make_graph()
# model = onnx.helper.make_model(graph)
# onnx.save(model,'test_init.onnx')