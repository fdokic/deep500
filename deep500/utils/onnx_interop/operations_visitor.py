import abc
from deep500.lv1.network import Network
from deep500.utils.onnx_interop.generated_operators import *


class OperationsVisitor(abc.ABC):
    def visit_lstm(self, op: LSTM, network: Network):
        raise Exception('implement this method')

    def visit_identity(self, op: Identity, network: Network):
        raise Exception('implement this method')

    def visit_abs(self, op: Abs, network: Network):
        raise Exception('implement this method')

    def visit_batchnormalization(self, op: BatchNormalization, network: Network):
        raise Exception('implement this method')

    def visit_mean(self, op: Mean, network: Network):
        raise Exception('implement this method')

    def visit_add(self, op: Add, network: Network):
        raise Exception('implement this method')

    def visit_globalmaxpool(self, op: GlobalMaxPool, network: Network):
        raise Exception('implement this method')

    def visit_cast(self, op: Cast, network: Network):
        raise Exception('implement this method')

    def visit_averagepool(self, op: AveragePool, network: Network):
        raise Exception('implement this method')

    def visit_and(self, op: And, network: Network):
        raise Exception('implement this method')

    def visit_lrn(self, op: LRN, network: Network):
        raise Exception('implement this method')

    def visit_argmax(self, op: ArgMax, network: Network):
        raise Exception('implement this method')

    def visit_resize(self, op: Resize, network: Network):
        raise Exception('implement this method')

    def visit_expand(self, op: Expand, network: Network):
        raise Exception('implement this method')

    def visit_neg(self, op: Neg, network: Network):
        raise Exception('implement this method')

    def visit_mul(self, op: Mul, network: Network):
        raise Exception('implement this method')

    def visit_argmin(self, op: ArgMin, network: Network):
        raise Exception('implement this method')

    def visit_castmap(self, op: CastMap, network: Network):
        raise Exception('implement this method')

    def visit_exp(self, op: Exp, network: Network):
        raise Exception('implement this method')

    def visit_div(self, op: Div, network: Network):
        raise Exception('implement this method')

    def visit_reversesequence(self, op: ReverseSequence, network: Network):
        raise Exception('implement this method')

    def visit_ceil(self, op: Ceil, network: Network):
        raise Exception('implement this method')

    def visit_depthtospace(self, op: DepthToSpace, network: Network):
        raise Exception('implement this method')

    def visit_clip(self, op: Clip, network: Network):
        raise Exception('implement this method')

    def visit_rnn(self, op: RNN, network: Network):
        raise Exception('implement this method')

    def visit_concat(self, op: Concat, network: Network):
        raise Exception('implement this method')

    def visit_constant(self, op: Constant, network: Network):
        raise Exception('implement this method')

    def visit_lppool(self, op: LpPool, network: Network):
        raise Exception('implement this method')

    def visit_conv(self, op: Conv, network: Network):
        raise Exception('implement this method')

    def visit_not(self, op: Not, network: Network):
        raise Exception('implement this method')

    def visit_gather(self, op: Gather, network: Network):
        raise Exception('implement this method')

    def visit_convtranspose(self, op: ConvTranspose, network: Network):
        raise Exception('implement this method')

    def visit_dropout(self, op: Dropout, network: Network):
        raise Exception('implement this method')

    def visit_leakyrelu(self, op: LeakyRelu, network: Network):
        raise Exception('implement this method')

    def visit_elu(self, op: Elu, network: Network):
        raise Exception('implement this method')

    def visit_globalaveragepool(self, op: GlobalAveragePool, network: Network):
        raise Exception('implement this method')

    def visit_gatherelements(self, op: GatherElements, network: Network):
        raise Exception('implement this method')

    def visit_gemm(self, op: Gemm, network: Network):
        raise Exception('implement this method')

    def visit_maxpool(self, op: MaxPool, network: Network):
        raise Exception('implement this method')

    def visit_equal(self, op: Equal, network: Network):
        raise Exception('implement this method')

    def visit_tile(self, op: Tile, network: Network):
        raise Exception('implement this method')

    def visit_flatten(self, op: Flatten, network: Network):
        raise Exception('implement this method')

    def visit_floor(self, op: Floor, network: Network):
        raise Exception('implement this method')

    def visit_gru(self, op: GRU, network: Network):
        raise Exception('implement this method')

    def visit_scatterelements(self, op: ScatterElements, network: Network):
        raise Exception('implement this method')

    def visit_globallppool(self, op: GlobalLpPool, network: Network):
        raise Exception('implement this method')

    def visit_greater(self, op: Greater, network: Network):
        raise Exception('implement this method')

    def visit_hardsigmoid(self, op: HardSigmoid, network: Network):
        raise Exception('implement this method')

    def visit_selu(self, op: Selu, network: Network):
        raise Exception('implement this method')

    def visit_hardmax(self, op: Hardmax, network: Network):
        raise Exception('implement this method')

    def visit_if(self, op: If, network: Network):
        raise Exception('implement this method')

    def visit_min(self, op: Min, network: Network):
        raise Exception('implement this method')

    def visit_instancenormalization(self, op: InstanceNormalization, network: Network):
        raise Exception('implement this method')

    def visit_less(self, op: Less, network: Network):
        raise Exception('implement this method')

    def visit_eyelike(self, op: EyeLike, network: Network):
        raise Exception('implement this method')

    def visit_randomnormal(self, op: RandomNormal, network: Network):
        raise Exception('implement this method')

    def visit_slice(self, op: Slice, network: Network):
        raise Exception('implement this method')

    def visit_prelu(self, op: PRelu, network: Network):
        raise Exception('implement this method')

    def visit_log(self, op: Log, network: Network):
        raise Exception('implement this method')

    def visit_logsoftmax(self, op: LogSoftmax, network: Network):
        raise Exception('implement this method')

    def visit_loop(self, op: Loop, network: Network):
        raise Exception('implement this method')

    def visit_lpnormalization(self, op: LpNormalization, network: Network):
        raise Exception('implement this method')

    def visit_matmul(self, op: MatMul, network: Network):
        raise Exception('implement this method')

    def visit_reducel2(self, op: ReduceL2, network: Network):
        raise Exception('implement this method')

    def visit_max(self, op: Max, network: Network):
        raise Exception('implement this method')

    def visit_maxroipool(self, op: MaxRoiPool, network: Network):
        raise Exception('implement this method')

    def visit_or(self, op: Or, network: Network):
        raise Exception('implement this method')

    def visit_pad(self, op: Pad, network: Network):
        raise Exception('implement this method')

    def visit_randomuniformlike(self, op: RandomUniformLike, network: Network):
        raise Exception('implement this method')

    def visit_reciprocal(self, op: Reciprocal, network: Network):
        raise Exception('implement this method')

    def visit_pow(self, op: Pow, network: Network):
        raise Exception('implement this method')

    def visit_randomnormallike(self, op: RandomNormalLike, network: Network):
        raise Exception('implement this method')

    def visit_onehot(self, op: OneHot, network: Network):
        raise Exception('implement this method')

    def visit_randomuniform(self, op: RandomUniform, network: Network):
        raise Exception('implement this method')

    def visit_concatfromsequence(self, op: ConcatFromSequence, network: Network):
        raise Exception('implement this method')

    def visit_reducel1(self, op: ReduceL1, network: Network):
        raise Exception('implement this method')

    def visit_reducelogsum(self, op: ReduceLogSum, network: Network):
        raise Exception('implement this method')

    def visit_reducelogsumexp(self, op: ReduceLogSumExp, network: Network):
        raise Exception('implement this method')

    def visit_reducemax(self, op: ReduceMax, network: Network):
        raise Exception('implement this method')

    def visit_onehotencoder(self, op: OneHotEncoder, network: Network):
        raise Exception('implement this method')

    def visit_isnan(self, op: IsNaN, network: Network):
        raise Exception('implement this method')

    def visit_reducemean(self, op: ReduceMean, network: Network):
        raise Exception('implement this method')

    def visit_reducemin(self, op: ReduceMin, network: Network):
        raise Exception('implement this method')

    def visit_treeensembleregressor(self, op: TreeEnsembleRegressor, network: Network):
        raise Exception('implement this method')

    def visit_reduceprod(self, op: ReduceProd, network: Network):
        raise Exception('implement this method')

    def visit_reducesum(self, op: ReduceSum, network: Network):
        raise Exception('implement this method')

    def visit_reducesumsquare(self, op: ReduceSumSquare, network: Network):
        raise Exception('implement this method')

    def visit_relu(self, op: Relu, network: Network):
        raise Exception('implement this method')

    def visit_reshape(self, op: Reshape, network: Network):
        raise Exception('implement this method')

    def visit_shape(self, op: Shape, network: Network):
        raise Exception('implement this method')

    def visit_sigmoid(self, op: Sigmoid, network: Network):
        raise Exception('implement this method')

    def visit_size(self, op: Size, network: Network):
        raise Exception('implement this method')

    def visit_softmax(self, op: Softmax, network: Network):
        raise Exception('implement this method')

    def visit_softplus(self, op: Softplus, network: Network):
        raise Exception('implement this method')

    def visit_softsign(self, op: Softsign, network: Network):
        raise Exception('implement this method')

    def visit_spacetodepth(self, op: SpaceToDepth, network: Network):
        raise Exception('implement this method')

    def visit_tfidfvectorizer(self, op: TfIdfVectorizer, network: Network):
        raise Exception('implement this method')

    def visit_split(self, op: Split, network: Network):
        raise Exception('implement this method')

    def visit_imputer(self, op: Imputer, network: Network):
        raise Exception('implement this method')

    def visit_sqrt(self, op: Sqrt, network: Network):
        raise Exception('implement this method')

    def visit_squeeze(self, op: Squeeze, network: Network):
        raise Exception('implement this method')

    def visit_topk(self, op: TopK, network: Network):
        raise Exception('implement this method')

    def visit_sub(self, op: Sub, network: Network):
        raise Exception('implement this method')

    def visit_sum(self, op: Sum, network: Network):
        raise Exception('implement this method')

    def visit_shrink(self, op: Shrink, network: Network):
        raise Exception('implement this method')

    def visit_tanh(self, op: Tanh, network: Network):
        raise Exception('implement this method')

    def visit_transpose(self, op: Transpose, network: Network):
        raise Exception('implement this method')

    def visit_unsqueeze(self, op: Unsqueeze, network: Network):
        raise Exception('implement this method')

    def visit_upsample(self, op: Upsample, network: Network):
        raise Exception('implement this method')

    def visit_svmclassifier(self, op: SVMClassifier, network: Network):
        raise Exception('implement this method')

    def visit_xor(self, op: Xor, network: Network):
        raise Exception('implement this method')

    def visit_acos(self, op: Acos, network: Network):
        raise Exception('implement this method')

    def visit_asin(self, op: Asin, network: Network):
        raise Exception('implement this method')

    def visit_atan(self, op: Atan, network: Network):
        raise Exception('implement this method')

    def visit_cos(self, op: Cos, network: Network):
        raise Exception('implement this method')

    def visit_sin(self, op: Sin, network: Network):
        raise Exception('implement this method')

    def visit_tan(self, op: Tan, network: Network):
        raise Exception('implement this method')

    def visit_multinomial(self, op: Multinomial, network: Network):
        raise Exception('implement this method')

    def visit_scan(self, op: Scan, network: Network):
        raise Exception('implement this method')

    def visit_compress(self, op: Compress, network: Network):
        raise Exception('implement this method')

    def visit_constantofshape(self, op: ConstantOfShape, network: Network):
        raise Exception('implement this method')

    def visit_maxunpool(self, op: MaxUnpool, network: Network):
        raise Exception('implement this method')

    def visit_scatter(self, op: Scatter, network: Network):
        raise Exception('implement this method')

    def visit_sinh(self, op: Sinh, network: Network):
        raise Exception('implement this method')

    def visit_cosh(self, op: Cosh, network: Network):
        raise Exception('implement this method')

    def visit_asinh(self, op: Asinh, network: Network):
        raise Exception('implement this method')

    def visit_acosh(self, op: Acosh, network: Network):
        raise Exception('implement this method')

    def visit_nonmaxsuppression(self, op: NonMaxSuppression, network: Network):
        raise Exception('implement this method')

    def visit_atanh(self, op: Atanh, network: Network):
        raise Exception('implement this method')

    def visit_sign(self, op: Sign, network: Network):
        raise Exception('implement this method')

    def visit_erf(self, op: Erf, network: Network):
        raise Exception('implement this method')

    def visit_where(self, op: Where, network: Network):
        raise Exception('implement this method')

    def visit_nonzero(self, op: NonZero, network: Network):
        raise Exception('implement this method')

    def visit_meanvariancenormalization(self, op: MeanVarianceNormalization, network: Network):
        raise Exception('implement this method')

    def visit_stringnormalizer(self, op: StringNormalizer, network: Network):
        raise Exception('implement this method')

    def visit_mod(self, op: Mod, network: Network):
        raise Exception('implement this method')

    def visit_thresholdedrelu(self, op: ThresholdedRelu, network: Network):
        raise Exception('implement this method')

    def visit_matmulinteger(self, op: MatMulInteger, network: Network):
        raise Exception('implement this method')

    def visit_qlinearmatmul(self, op: QLinearMatMul, network: Network):
        raise Exception('implement this method')

    def visit_convinteger(self, op: ConvInteger, network: Network):
        raise Exception('implement this method')

    def visit_qlinearconv(self, op: QLinearConv, network: Network):
        raise Exception('implement this method')

    def visit_quantizelinear(self, op: QuantizeLinear, network: Network):
        raise Exception('implement this method')

    def visit_gathernd(self, op: GatherND, network: Network):
        raise Exception('implement this method')

    def visit_dequantizelinear(self, op: DequantizeLinear, network: Network):
        raise Exception('implement this method')

    def visit_isinf(self, op: IsInf, network: Network):
        raise Exception('implement this method')

    def visit_roialign(self, op: RoiAlign, network: Network):
        raise Exception('implement this method')

    def visit_sequencelength(self, op: SequenceLength, network: Network):
        raise Exception('implement this method')

    def visit_bitshift(self, op: BitShift, network: Network):
        raise Exception('implement this method')

    def visit_unique(self, op: Unique, network: Network):
        raise Exception('implement this method')

    def visit_cumsum(self, op: CumSum, network: Network):
        raise Exception('implement this method')

    def visit_round(self, op: Round, network: Network):
        raise Exception('implement this method')

    def visit_dynamicquantizelinear(self, op: DynamicQuantizeLinear, network: Network):
        raise Exception('implement this method')

    def visit_range(self, op: Range, network: Network):
        raise Exception('implement this method')

    def visit_det(self, op: Det, network: Network):
        raise Exception('implement this method')

    def visit_scatternd(self, op: ScatterND, network: Network):
        raise Exception('implement this method')

    def visit_sequenceempty(self, op: SequenceEmpty, network: Network):
        raise Exception('implement this method')

    def visit_sequenceconstruct(self, op: SequenceConstruct, network: Network):
        raise Exception('implement this method')

    def visit_sequenceinsert(self, op: SequenceInsert, network: Network):
        raise Exception('implement this method')

    def visit_sequenceat(self, op: SequenceAt, network: Network):
        raise Exception('implement this method')

    def visit_sequenceerase(self, op: SequenceErase, network: Network):
        raise Exception('implement this method')

    def visit_splittosequence(self, op: SplitToSequence, network: Network):
        raise Exception('implement this method')

    def visit_arrayfeatureextractor(self, op: ArrayFeatureExtractor, network: Network):
        raise Exception('implement this method')

    def visit_binarizer(self, op: Binarizer, network: Network):
        raise Exception('implement this method')

    def visit_categorymapper(self, op: CategoryMapper, network: Network):
        raise Exception('implement this method')

    def visit_dictvectorizer(self, op: DictVectorizer, network: Network):
        raise Exception('implement this method')

    def visit_featurevectorizer(self, op: FeatureVectorizer, network: Network):
        raise Exception('implement this method')

    def visit_labelencoder(self, op: LabelEncoder, network: Network):
        raise Exception('implement this method')

    def visit_linearclassifier(self, op: LinearClassifier, network: Network):
        raise Exception('implement this method')

    def visit_linearregressor(self, op: LinearRegressor, network: Network):
        raise Exception('implement this method')

    def visit_normalizer(self, op: Normalizer, network: Network):
        raise Exception('implement this method')

    def visit_svmregressor(self, op: SVMRegressor, network: Network):
        raise Exception('implement this method')

    def visit_scaler(self, op: Scaler, network: Network):
        raise Exception('implement this method')

    def visit_treeensembleclassifier(self, op: TreeEnsembleClassifier, network: Network):
        raise Exception('implement this method')

    def visit_zipmap(self, op: ZipMap, network: Network):
        raise Exception('implement this method')

