from onnx import ModelProto, ValueInfoProto


def complete_inputs(model: ModelProto):
    """
    :param model: ModelProto using Initialization Graph, with potentially incomplete model.graph.input
    this method adds missing initializer inputs by inferring them from
    TrainingInfo.initializer_binding and Initializer.output
    :return: ModelProto where model.graph.input contains initializer inputs specified through initializer_binding
    """

    missing_inputs = []
    inputs = model.graph.input
    for training in model.training_info:
        bindings = training.initialization_binding
        for output in training.initialization.output:
            input_name = [b.key for b in bindings if b.value == output.name].pop()
            if input_name not in [i.name for i in inputs]:
                value = ValueInfoProto()
                value.CopyFrom(output)
                value.name = input_name
                missing_inputs.append(value)

    model.graph.input.extend(missing_inputs)
    return model
