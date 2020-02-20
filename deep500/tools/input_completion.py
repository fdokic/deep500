from onnx import ModelProto, ValueInfoProto, TensorShapeProto, TypeProto


def complete_inputs(model: ModelProto):
    """
    :param model: ModelProto using Initialization Graph, with potentially incomplete model.graph.input
    this method adds missing initializer inputs by inferring them from
    TrainingInfo.initializer_binding and Initializer.output
    :return: ModelProto where model.graph.input contains initializer inputs specified through initializer_binding
    """

    missing_inputs = []
    inputs = model.graph.input
    # complete inputs for initialization graph
    for training in model.training_info:
        bindings = training.initialization_binding
        for output in training.initialization.output:
            input_name = [b.key for b in bindings if b.value == output.name].pop()
            if input_name not in [i.name for i in inputs]:
                value = ValueInfoProto()
                value.CopyFrom(output)
                value.name = input_name
                missing_inputs.append(value)

    # comlete inputs for initializers
    for i in model.graph.initializer:
        if i.name in [j.name for j in model.graph.input]:
            continue
        type = i.data_type
        shape = i.dims
        name = i.name

        tensor_type = TypeProto.Tensor()
        type_proto = TypeProto()
        shape_proto = TensorShapeProto()
        for s in shape:
            d = TensorShapeProto.Dimension()
            d.dim_value = s
            shape_proto.dim.extend([d])

        tensor_type.shape.CopyFrom(shape_proto)
        tensor_type.elem_type = type

        type_proto.tensor_type.CopyFrom(tensor_type)

        value = ValueInfoProto()
        value.name = name
        value.type.CopyFrom(type_proto)
        missing_inputs.append(value)

    model.graph.input.extend(missing_inputs)
    return model
