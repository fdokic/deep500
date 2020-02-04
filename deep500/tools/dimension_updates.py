from typing import Dict, List, Optional, Text, Union
from onnx import ModelProto
from onnx.tools import update_model_dims


def update_dimensions(model, input_dim=None, output_dim=None):  # type: (ModelProto, Optional[Dict[int, Union[int, Text]]], Optional[Dict[int, Union[int, Text]]]) -> ModelProto
    """
    This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    The difference with update_inputs_outputs_dims is that only the dimensions that should be changed need
    to be provided. This simplifies making models symbolic (e.g., with respect to batch or length parameters) or
    instantiating them.

        Example. if we have the following shape for inputs and outputs:
                shape(input_1) = (128, 'l', 64, 64)
                shape(input_2) = (128, 'w')
                and shape(output)  = (128, 'c', 5)
                The parameters can be provided as:
                input_dim = {
                    0: 'b',
                    1: 20
                }
                output_dim = {
                    0: -1
                }
                producing the following result:
                shape(input_1) = ('b', 20, 64, 64)
                shape(input_2) = ('b', 20)
                and shape(output)  = ('output.0', 'c', 5)

    :param model: Onnx Protobuf model to be modified
    :param input_dim: Dictionary of zero-based dimension indices and corresponding values (int or string) to be set. -1 for unique string
    :param output_dim: Dictionary of zero-based dimension indices and corresponding values (int or string) to be set. -1 for unique string
    :return: Model with modified inputs and outputs
    """
    input_dim = input_dim or {}
    output_dim = output_dim or {}
    if not (bool(input_dim) and bool(output_dim)):
        return model

    # Separate graph inputs from initializers from graph.input set
    inputs = set(i.name for i in model.graph.input)
    initializers = set(i.name for i in model.graph.initializer)
    graph_inputs = inputs - initializers

    outputs = model.graph.output

    # Create complete dictionary of input dimensions. Only Graph inputs are modified as specified in input_dim
    inp = {}
    for i in inputs:
        # Gather input i, inputs only contains strings
        input = [j for j in model.graph.input if j.name == i][0]
        dimensions = input.type.tensor_type.shape.dim

        ls = []  # type: List[Union[int, Text]]

        if len(input_dim) == 0:
            break

        for d in dimensions:
            if d.HasField('dim_param'):
                ls.append(d.dim_param)
            if d.HasField('dim_value'):
                ls.append(d.dim_value)

        # Only change graph_input dimensions
        if input.name in graph_inputs:
            # Throw error in case specified dimension is too large (does not exist in inputs)
            if len(ls) <= max(input_dim.keys()):
                ValueError('Input {} has only {} dimensions, less than {} that are given'.format(input.name, len(ls),
                                                                                                 max(input_dim.keys())))
            for k in input_dim.keys():
                ls[k] = input_dim[k]

        inp[input.name] = ls

    # Create complete dictionary of output dimensions. All outputs are considered to be Graph outputs
    out = {}
    for o in outputs:
        dimensions = o.type.tensor_type.shape.dim

        ls = []

        if len(output_dim) == 0:
            break

        for d in dimensions:
            if d.HasField('dim_param'):
                ls.append(d.dim_param)
            if d.HasField('dim_value'):
                ls.append(d.dim_value)

        if len(ls) <= max(output_dim.keys()):
            ValueError('Input {} has only {} Dimensions, less than {} that are given'.format(o.name, len(ls),
                                                                                             max(output_dim.keys())))
        for k in output_dim.keys():
            ls[k] = output_dim[k]

        out[o.name] = ls

    return update_model_dims.update_inputs_outputs_dims(model, inp, out)