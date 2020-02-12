import abc
import enum
import struct
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np

"""
This file represents the framework independent intermediate representation of Deep500. 
The object is created by the ONNX parser. You can traverse this structure by using the `accept`
method and providing an appropriate visitor. 
"""


class Element(abc.ABC):
    """
    Base class for accept
    """

    @abc.abstractmethod
    def accept(self, visitor, network):
        pass


class AttributeType(Enum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 11

    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12


class OnnxAttribute(abc.ABC):
    def __init__(self, name: str, doc_string: Optional[str] = None):
        self.name = name
        self.doc_string = doc_string

    @abc.abstractmethod
    def get_value(self):
        pass

    @staticmethod
    def create_from_attribute(attribute):
        attr_type = AttributeType(attribute.type)
        if attr_type is AttributeType.FLOAT:
            return OnnxFloatAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.INT:
            return OnnxIntAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.STRING:
            return OnnxStringAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.INTS:
            return OnnxIntsAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.FLOATS:
            return OnnxFloatsAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.STRINGS:
            return OnnxStringsAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.TENSOR:
            return OnnxTensorAttribute.create_from_attribute(attribute)
        elif attr_type is AttributeType.SPARSE_TENSOR:
            return OnnxSparseTensorAttribute.create_from_attribute(attribute)
        raise Exception("Did not recognize attribute type: {}".format(attr_type))


class OnnxFloatAttribute(OnnxAttribute):
    def __init__(self, name: str, value: float, doc_string: Optional[str] = None, ):
        super(OnnxFloatAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, attribute.f, attribute.doc_string)

    def get_value(self) -> float:
        return self.value


class OnnxFloatsAttribute(OnnxAttribute):
    def __init__(self, name: object, value: object, doc_string: object = None) -> object:
        super(OnnxFloatsAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, list(attribute.floats), attribute.doc_string)

    def get_value(self) -> List[float]:
        return self.value


class OnnxIntsAttribute(OnnxAttribute):
    def __init__(self, name: str, value: List[int], doc_string: Optional[str] = None):
        super(OnnxIntsAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, list(attribute.ints), attribute.doc_string)

    def get_value(self) -> List[int]:
        return self.value


class OnnxStringsAttribute(OnnxAttribute):
    def __init__(self, name: str, value: List[str], doc_string: Optional[str] = None):
        super(OnnxStringsAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, list(attribute.strings), attribute.doc_string)

    def get_value(self) -> List[str]:
        return self.value


class OnnxIntAttribute(OnnxAttribute):
    def __init__(self, name: str, value: int, doc_string: Optional[str] = None):
        super(OnnxIntAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, attribute.i, attribute.doc_string)

    def get_value(self) -> int:
        return self.value


class OnnxStringAttribute(OnnxAttribute):
    def __init__(self, name: str, value: str, doc_string: Optional[str] = None):
        super(OnnxStringAttribute, self).__init__(name, doc_string)
        self.value = value

    @classmethod
    def create_from_attribute(cls, attribute):
        return cls(attribute.name, attribute.s.decode('utf-8'), attribute.doc_string)

    def get_value(self) -> str:
        return self.value


class OnnxNode(Element, metaclass=abc.ABCMeta):
    def __init__(self, input: List[str], output: List[str],
                 name: Optional[str], op_type: Optional[str], domain: Optional[str]
                 , attributes: Dict[str, OnnxAttribute], doc_string: Optional[str]):
        self.input = input
        self.output = output
        self.name = name
        self.op_type = op_type  # Operation.get_operation(node.op_type)
        self.domain = domain
        self.attributes = attributes
        self.doc_string = doc_string

    @classmethod
    def create_from_node(cls, node):
        input = list(node.input)
        output = list(node.output)
        name = node.name
        op_type = node.op_type  # Operation.get_operation(node.op_type)
        domain = node.domain
        attributes = dict([(attr.name, OnnxAttribute.create_from_attribute(attr))
                                                     for attr in node.attribute])
        doc_string = node.doc_string
        return cls(input, output, name, op_type, domain, attributes, doc_string)

    def accept(self, visitor, network):
        # This method is only called from children operations so that common visitor functionality can be
        # encapsulated in the called visit method
        visitor.visit_node(self, network)


class Operation(OnnxNode, metaclass=abc.ABCMeta):
    def __init__(self, input: List[str], output: List[str],
                 name: Optional[str], op_type: Optional[str], domain: Optional[str]
                 , attributes: Dict[str, OnnxAttribute], doc_string: Optional[str]):
        super(Operation, self).__init__(input, output, name, op_type, domain, attributes, doc_string)

    @classmethod
    def create_node(cls, op_type, node):
        from deep500.utils.onnx_interop.generated_operators import ONNX_OPERATIONS
        op = ONNX_OPERATIONS.get(op_type.lower())
        if op is None:
            raise Exception('please provide the operation type before using it')
        return op.create_from_node(node)


class TensorDataType(enum.Enum):
    UNDEFINED = 0

    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9

    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15

    @classmethod
    def from_numpy(cls, np_type: np.dtype) -> 'TensorDataType':
        if np_type == np.float32:
            return TensorDataType.FLOAT
        if np_type == np.int64:
            return TensorDataType.INT64
        if np_type == np.int32:
            return TensorDataType.INT32
        if np_type == np.int16:
            return TensorDataType.INT16
        if np_type == np.int8:
            return TensorDataType.INT8
        if np_type == np.uint8:
            return TensorDataType.UINT8
        if np_type == np.uint16:
            return TensorDataType.UINT16
        if np_type == np.string_:
            return TensorDataType.STRING
        if np_type == np.bool_:
            return TensorDataType.BOOL
        if np_type == np.float16:
            return TensorDataType.FLOAT16
        if np_type == np.uint32:
            return TensorDataType.UINT32
        if np_type == np.uint64:
            return TensorDataType.UINT64
        if np_type == np.float64:
            return TensorDataType.DOUBLE
        if np_type == np.complex64:
            return TensorDataType.COMPLEX64
        if np_type == np.complex128:
            return TensorDataType.COMPLEX128
        raise Exception('We did not find corresponding type for numpy dtype:{}'.format(np_type))

    def to_numpy(self):
        if self == TensorDataType.FLOAT:
            return np.float32
        if self == TensorDataType.INT64:
            return np.int64
        if self == TensorDataType.INT32:
            return np.int32
        if self == TensorDataType.INT16:
            return np.int16
        if self == TensorDataType.INT8:
            return np.int8
        if self == TensorDataType.UINT8:
            return np.uint8
        if self == TensorDataType.UINT16:
            return np.uint16
        if self == TensorDataType.STRING:
            return np.string_
        if self == TensorDataType.BOOL:
            return np.bool
        if self == TensorDataType.FLOAT16:
            return np.float16
        if self == TensorDataType.UINT32:
            return np.uint32
        if self == TensorDataType.UINT64:
            return np.uint64
        if self == TensorDataType.DOUBLE:
            return np.float64
        if self == TensorDataType.COMPLEX64:
            return np.complex64
        if self == TensorDataType.COMPLEX128:
            return np.complex128
        raise Exception('We did not find corresponding type for type:{}'.format(self))




class OnnxTensorShape:
    def __init__(self, shape):
        self.shape = shape

    @classmethod
    def create_from_onnx_shape(cls, onnx_shape):
        shape = [each_dim.dim_value if each_dim.dim_value is not None else each_dim.dim_param
                 for each_dim in onnx_shape.dim]
        return cls(shape)


class OnnxType:
    def __init__(self):
        pass

    @classmethod
    def create_type(cls, typez):
        if hasattr(typez, 'tensor_type'):
            return OnnxTensorType.create_from_onnx_type(typez.tensor_type)
        if hasattr(typez, 'sequence_type'):
            return OnnxSequenceType.create_from_onnx_type(typez.sequence_type)
        if hasattr(typez, 'map_type'):
            return OnnxMapType.create_from_onnx_type(typez.map_type)
        raise Exception('There should not be other types check youre model generator!')


class OnnxTensorType(OnnxType):
    def __init__(self, elem_type: TensorDataType, shape: OnnxTensorShape):
        super(OnnxTensorType, self).__init__()
        self.type = elem_type
        self.shape = shape

    @classmethod
    def create_from_onnx_type(cls, tensor_type):
        elem_type = TensorDataType(tensor_type.elem_type)
        tensor_shape = OnnxTensorShape.create_from_onnx_shape(tensor_type.shape)
        return cls(elem_type, tensor_shape)


class OnnxSequenceType(OnnxType):
    def __init__(self, elem_type: OnnxType):
        super(OnnxSequenceType, self).__init__()
        self.elem_type = elem_type

    @classmethod
    def create_from_onnx_type(cls, sequence_type):
        elem_type = OnnxType.create_type(sequence_type.elem_type)
        return cls(elem_type)


class OnnxMapType(OnnxType):
    def __init__(self, key_type: TensorDataType, value_type: OnnxType):
        super(OnnxMapType, self).__init__()
        self.key_type = key_type
        self.value_type = value_type

    @classmethod
    def create_from_onnx_type(cls, onnx_type):
        key_type = TensorDataType(onnx_type.key_type)
        value_type = OnnxType.create_type(onnx_type.value_type)
        return cls(key_type, value_type)


class OnnxTensor:
    def __init__(self, dims: Tuple[int, ...], data: np.ndarray, segment: Optional[Tuple[int, int]], name: Optional[str],
                 doc_string: Optional[str]):
        self.dims = dims
        self.segment = segment
        self.name = name
        self.doc_string = doc_string
        self.data = data
        self.data.resize(self.dims)

    def get_data(self):
        return self.data

    @staticmethod
    def create_from_onnx_tensor(tensor):
        tensor_type = TensorDataType(tensor.data_type)
        dims = tuple(tensor.dims)
        segment = None if tensor.segment is None \
            else (tensor.segment.begin, tensor.segment.end)
        name = tensor.name
        doc_string = tensor.doc_string

        if tensor_type is TensorDataType.FLOAT \
                or tensor_type is TensorDataType.FLOAT16:
            return OnnxFloatTensor.create_from_onnx_tensor_(tensor, dims, segment, name, doc_string)
        if tensor_type is TensorDataType.INT64:
            return OnnxInt64Tensor.create_from_onnx_tensor_(tensor, dims, segment, name, doc_string)
        if tensor_type is TensorDataType.STRING:
            return OnnxStringTensor.create_from_onnx_tensor_(tensor, dims, segment, name, doc_string)
        if tensor_type is TensorDataType.DOUBLE:
            return OnnxDoubleTensor.create_from_onnx_tensor_(tensor, dims, segment, name, doc_string)
        raise Exception('Tensor type: {} is not supported at the moment'
                        .format(tensor_type))


class OnnxTensorAttribute(OnnxAttribute):
    def __init__(self, name: str, tensor: OnnxTensor, doc_string: Optional[str] = None):
        super(OnnxTensorAttribute, self).__init__(name, doc_string)
        self.value = tensor

    @classmethod
    def create_from_attribute(cls, attribute):
        tensor = OnnxTensor.create_from_onnx_tensor(attribute.t)
        return cls(attribute.name, tensor, attribute.doc_string)

    def get_value(self):
        return self.value.get_data()

    
class OnnxSparseTensor:
    def __init__(self, values_tensor: OnnxTensor, indices_tensor: OnnxTensor, dims: int, name: Optional[str],
                 doc_string: Optional[str]):
        self.dims = dims
        self.values = values_tensor
        self.indices = indices_tensor
        self.name = name
        self.doc_string = doc_string
        self.data = data
        self.data.resize(self.dims)

    def get_data(self):
        return (self.values.get_data(), self.indices.get_data(), self.dims)

    @staticmethod
    def create_from_onnx_sparse_tensor(tensor):
        dims = tuple(tensor.dims)
        name = tensor.name
        doc_string = tensor.doc_string
        values = OnnxTensor.create_from_onnx_tensor(tensor.values)
        indices = OnnxTensor.create_from_onnx_tensor(tensor.indices)
        return OnnxSparseTensor(values, indices, dims, name, doc_string)

    
class OnnxSparseTensorAttribute(OnnxAttribute):
    def __init__(self, name: str, tensor: OnnxSparseTensor, doc_string: Optional[str] = None):
        super(OnnxSparseTensorAttribute, self).__init__(name, doc_string)
        self.value = tensor

    @classmethod
    def create_from_attribute(cls, attribute):
        tensor = OnnxSparseTensor.create_from_onnx_sparse_tensor(attribute.sparse_tensor)
        return cls(attribute.name, tensor, attribute.doc_string)

    def get_value(self):
        return self.value.get_data()

    
class OnnxFloatTensor(OnnxTensor):
    def __init__(self, dims: Tuple[int, ...], data: np.ndarray, segment: Optional[Tuple[int, int]],
                 name: Optional[str],
                 doc_string: Optional[str]):
        super(OnnxFloatTensor, self).__init__(dims, data, segment, name, doc_string)

    @classmethod
    def create_from_onnx_tensor_(cls, tensor, dims, segment, name, doc_string):
        data = tensor.float_data
        if data is None or len(data) == 0:
            # TODO(HBS): make the astype generic
            raw_data = tensor.raw_data
            raw_data = list(struct.unpack('f' * int(len(raw_data) / 4), raw_data))
            data = np.array(raw_data).astype(np.float32)
        else:
            data = np.array(list(data)).astype(np.float32)

        return cls(dims, data, segment, name, doc_string)

    
class OnnxDoubleTensor(OnnxTensor):
    def __init__(self, dims: Tuple[int, ...], data: np.ndarray, segment: Optional[Tuple[int, int]],
                 name: Optional[str],
                 doc_string: Optional[str]):
        super(OnnxDoubleTensor, self).__init__(dims, data, segment, name, doc_string)

    @classmethod
    def create_from_onnx_tensor_(cls, tensor, dims, segment, name, doc_string):
        data = tensor.float_data
        if data is None or len(data) == 0:
            # TODO(HBS): make the astype generic
            raw_data = tensor.raw_data
            raw_data = list(struct.unpack('d' * int(len(raw_data) / 8), raw_data))
            data = np.array(raw_data).astype(np.float64)
        else:
            data = np.array(list(data)).astype(np.float64)

        return cls(dims, data, segment, name, doc_string)

class OnnxInt64Tensor(OnnxTensor):
    def __init__(self, dims: Tuple[int, ...], data: np.ndarray, segment: Optional[Tuple[int, int]],
                 name: Optional[str],
                 doc_string: Optional[str]):
        super(OnnxInt64Tensor, self).__init__(dims, data, segment, name, doc_string)

    @classmethod
    def create_from_onnx_tensor_(cls, tensor, dims, segment, name, doc_string):
        data = tensor.int64_data
        if data is None or len(data) == 0:
            # TODO(HBS): make the astype generic
            raw_data = tensor.raw_data
            raw_data = list(struct.unpack('q' * int(len(raw_data) / 8), raw_data))
            data = np.array(raw_data).astype(np.int64)
        else:
            data = np.array(list(data)).astype(np.int64)

        return cls(dims, data, segment, name, doc_string)


class OnnxStringTensor(OnnxTensor):
    def __init__(self, dims: Tuple[int, ...], data: np.ndarray, segment: Optional[Tuple[int, int]],
                 name: Optional[str],
                 doc_string: Optional[str]):
        super(OnnxStringTensor, self).__init__(dims, data, segment, name, doc_string)

    @classmethod
    def create_from_onnx_tensor_(cls, tensor, dims, segment, name, doc_string):
        cls(dims, tensor.string_data, segment, name, doc_string)


class OnnxValueInfo:
    def __init__(self, name: str, type: OnnxType, doc_string: Optional[str]):
        self.name = name
        self.type = type
        self.doc_string = doc_string

    @staticmethod
    def create_info(info):
        return [OnnxValueInfo.create_from_onnx_info(i) for i in info]

    @classmethod
    def create_from_onnx_info(cls, info):
        name = info.name
        type = OnnxType.create_type(info.type)
        doc_string = info.doc_string
        return cls(name, type, doc_string)


class OnnxGraph(Element):
    def __init__(self, nodes: List[Operation], name: Optional[str],
                 initializers: List[OnnxTensor], doc_string: Optional[str], inputs: List[OnnxValueInfo],
                 outputs: List[OnnxValueInfo], value_info: List[OnnxValueInfo]):
        self.nodes = nodes
        self.name = name
        self.initializers = initializers
        self.doc_string = doc_string
        self.inputs = inputs
        self.outputs = outputs
        self.value_info = value_info

    @classmethod
    def create_from_onnx_graph(cls, graph):
        nodes = [Operation.create_node(node.op_type, node) for node in graph.node]
        name = graph.name
        initializers = [OnnxTensor.create_from_onnx_tensor(tensor) for tensor in
                                          graph.initializer]
        doc_string = graph.doc_string
        inputs = OnnxValueInfo.create_info(graph.input)
        outputs = OnnxValueInfo.create_info(graph.output)
        value_info = OnnxValueInfo.create_info(graph.value_info)
        return cls(nodes, name, initializers, doc_string, inputs, outputs, value_info)

    def accept(self, visitor, network):
        visitor.visit_graph(self, network)
        for each_input in self.inputs:
            visitor.visit_net_input(each_input, network)
        for each_initializer in self.initializers:
            visitor.visit_initializer(each_initializer, network)
        visitor.visit_initializer_end(network)
        for each_output in self.outputs:
            visitor.visit_net_output(each_output, network)
        for each_node in self.nodes:
            each_node.accept(visitor, network)
        visitor.visit_graph_end(network)

    def extract_partial_graph(self, input_name: str, new_name: Optional[str] = '', old_name: Optional[str] = ''):
        """
        This function extracts standalone graphs of disconnected graphs. It only supports extraction of disconnected
        graphs, since for arbitrary splicing, inference of intermediate ValueInfoProto would be required to specify
        graph.outputs / inputs.

        Only use on initialized Graphs, function ignores initialization graphs

        :param input_name: name of first input (in topological order) of disconnected graph to be extracted, needs to
                           have an object in self.inputs where input.name equals input_name
        :return: returns OnnxGraph containing standalone Graph with graph_input as input. self contains the remainder.
        """
        new_input = [input for input in self.inputs if input.name == input_name].pop()
        if new_input is None:
            raise Exception('specified input {} does not exist' .format(new_input))

        new_nodes = []
        current_outputs = set([input_name])
        pendent_nodes = self.nodes
        inputs = set([input_name])

        # iterate until all reachable nodes reached
        change = True
        while change:
            change = False
            for n in pendent_nodes:
                if any([out for out in n.input if out in current_outputs]):
                    new_nodes.append(n)
                    pendent_nodes.remove(n)
                    current_outputs.update(n.output)
                    inputs.update(n.input)
                    change = True

        # extract corresponding inputs, outputs and initializers
        new_inputs = [input for input in self.inputs if input.name in inputs]
        new_outputs = [output for output in self.outputs if output.name in current_outputs]
        new_initializers = [initializers for initializers in self.initializers if initializers.name in inputs]
        new_graph = OnnxGraph(new_nodes, new_name, new_initializers, None, new_inputs, new_outputs, None)
        new_model = OnnxModel(new_graph, doc_string=None, training_info=[])

        self.inputs = list(set(self.inputs) - set(new_inputs))
        self.outputs = list(set(self.outputs) - set(new_outputs))
        self.initializers = list(set(self.initializers) - set(new_initializers))
        self.name = old_name

        # todo: split update_bindings?
        return new_model



class OnnxStringStringEntry:
    def __init__(self, key: Optional[str], value: Optional[str]):
        self.key = key
        self.value = value

    @classmethod
    def create_from_onnx_entry(cls, entry):
        key = entry.key
        value = entry.value
        return cls(key, value)


class OnnxTrainingInfo(Element):
    def __init__(self, initialization: OnnxGraph, algorithm: OnnxGraph,
                 initialization_binding: List[OnnxStringStringEntry], update_binding: List[OnnxStringStringEntry]):
        self.initialization = initialization
        self.algorithm = algorithm
        self.initialization_binding = initialization_binding
        self.update_binding = update_binding

    @classmethod
    def create_from_onnx_training(cls, training):
        initialization = OnnxGraph.create_from_onnx_graph(training.initialization)
        algorithm = OnnxGraph.create_from_onnx_graph(training.algorithm)
        initialization_binding = [OnnxStringStringEntry.create_from_onnx_entry(entry)
                                  for entry in training.initialization_binding]
        update_binding = [OnnxStringStringEntry.create_from_onnx_entry(entry)
                             for entry in training.update_binding]
        return cls(initialization, algorithm, initialization_binding, update_binding)

    def accept(self, visitor, network):
        pass



class OnnxModel(Element):
    def __init__(self, graph: OnnxGraph, doc_string: Optional[str], training_info: List[OnnxTrainingInfo]):
        self.graph = graph
        self.doc_string = doc_string
        self.training_info = training_info

    @classmethod
    def create_from_onnx_model(cls, model):
        doc_string = model.doc_string
        graph = OnnxGraph.create_from_onnx_graph(model.graph)
        training_info = [OnnxTrainingInfo.create_from_onnx_training(training) for training
                         in model.training_info]
        if len(model.metadata_props) > 0:
            raise Exception("TODO: Implement by HBS")  # TODO: Implement by HBS
        return cls(graph, doc_string, training_info)

    def accept(self, visitor, network):
        visitor.visit_model(self, network)
        self.graph.accept(visitor, network)

    def get_input_nodes(self) -> List[OnnxValueInfo]:
        return [node for node in self.graph.inputs]
    def get_output_nodes(self) -> List[OnnxValueInfo]:
        return [node for node in self.graph.outputs]

    def get_inputs(self) -> List[str]:
        return [node.name for node in self.graph.inputs]
    def get_outputs(self) -> List[str]:
        return [node.name for node in self.graph.outputs]


    def add_operation(self, op: Operation) -> int:
        """
        Add an operation labeling nodes in a topological ordering.
        @return the place where the element was inserted
        """
        place_to_insert = 0
        need_inputs = op.input if op.input is not None else []
        need_inputs = list(set(need_inputs) - set([x.name for x in self.graph.inputs]))
        if len(need_inputs) > 0:
            for i, node in enumerate(self.graph.nodes):
                curr_len = len(need_inputs)
                output = node.output if type(node.output) == list else [node.output]
                need_inputs = list(set(need_inputs) - set(output))
                if curr_len > len(need_inputs):
                    place_to_insert = i + 1

        self.graph.nodes.insert(place_to_insert, op)

        return place_to_insert

    def initialize(self):
        from deep500.frameworks.pytorch import PyTorchVisitor, PyTorchNetwork, PyTorchNativeNetwork
        import deep500 as d5
        import torch

        initializers = []
        # calculate all initializers as torch.tensors directly
        for training in self.training_info:

            if len(training.initialization.nodes) == 0:
                continue

            # create GraphExecutor graph
            visitor = PyTorchVisitor()
            network = PyTorchNetwork(d5.utils.device.CPUDevice())
            training.initialization.accept(visitor, network)
            graph = visitor.model.to('cpu')
            graph.eval()
            new_network = PyTorchNativeNetwork(graph)
            new_network.outputs = network.outputs
            network = new_network

            # execute model without inputs
            with torch.no_grad():
                graph()

            algo_initializers = []
            bindings = training.initialization_binding
            for i, out in enumerate(list(network.outputs)):
                # save initializers according to initialization_binding
                key = [a.key for a in bindings if a.value == out].pop()

                tensor = graph._params[out].detach().cpu().numpy()
                tensor_type = tensor.dtype

                if np.issubdtype(tensor_type, np.single)  \
                        or np.issubdtype(tensor_type, np.float16):
                    initializer = OnnxFloatTensor(tensor.shape, tensor, segment=None, name=key, doc_string=None)
                elif np.issubdtype(tensor_type, np.int64):
                    initializer = OnnxInt64Tensor(tensor.shape, tensor, segment=None, name=key, doc_string=None)
                elif np.issubdtype(tensor_type, str):
                    initializer = OnnxStringTensor(tensor.shape, tensor, segment=None, name=key, doc_string=None)
                elif np.issubdtype(tensor_type, np.double):
                    initializer = OnnxDoubleTensor(tensor.shape, tensor, segment=None, name=key, doc_string=None)
                else:
                    raise Exception('Tensor type: {} is not supported at the moment'
                                    .format(tensor_type))

                if key in [o.name for o in training.algorithm.inputs]:
                    algo_initializers.append(initializer)
                else:
                    initializers.append(initializer)

            training.algorithm.initializers.extend(algo_initializers)
        self.graph.initializers.extend(initializers)
        return
