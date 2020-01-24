import abc
from deep500.lv1.network import Network
from deep500.utils.onnx_interop.generated_operators import *


class EmptyOperationsVisitor(abc.ABC):
    def visit_lstm(self, op: LSTM, network: Network):
        pass

    def visit_identity(self, op: Identity, network: Network):
        pass

    def visit_abs(self, op: Abs, network: Network):
        pass

    def visit_batchnormalization(self, op: BatchNormalization, network: Network):
        pass

    def visit_mean(self, op: Mean, network: Network):
        pass

    def visit_add(self, op: Add, network: Network):
        pass

    def visit_globalmaxpool(self, op: GlobalMaxPool, network: Network):
        pass

    def visit_cast(self, op: Cast, network: Network):
        pass

    def visit_averagepool(self, op: AveragePool, network: Network):
        pass

    def visit_and(self, op: And, network: Network):
        pass

    def visit_lrn(self, op: LRN, network: Network):
        pass

    def visit_argmax(self, op: ArgMax, network: Network):
        pass

    def visit_resize(self, op: Resize, network: Network):
        pass

    def visit_expand(self, op: Expand, network: Network):
        pass

    def visit_neg(self, op: Neg, network: Network):
        pass

    def visit_mul(self, op: Mul, network: Network):
        pass

    def visit_argmin(self, op: ArgMin, network: Network):
        pass

    def visit_castmap(self, op: CastMap, network: Network):
        pass

    def visit_exp(self, op: Exp, network: Network):
        pass

    def visit_div(self, op: Div, network: Network):
        pass

    def visit_reversesequence(self, op: ReverseSequence, network: Network):
        pass

    def visit_ceil(self, op: Ceil, network: Network):
        pass

    def visit_depthtospace(self, op: DepthToSpace, network: Network):
        pass

    def visit_clip(self, op: Clip, network: Network):
        pass

    def visit_rnn(self, op: RNN, network: Network):
        pass

    def visit_concat(self, op: Concat, network: Network):
        pass

    def visit_constant(self, op: Constant, network: Network):
        pass

    def visit_lppool(self, op: LpPool, network: Network):
        pass

    def visit_conv(self, op: Conv, network: Network):
        pass

    def visit_not(self, op: Not, network: Network):
        pass

    def visit_gather(self, op: Gather, network: Network):
        pass

    def visit_convtranspose(self, op: ConvTranspose, network: Network):
        pass

    def visit_dropout(self, op: Dropout, network: Network):
        pass

    def visit_leakyrelu(self, op: LeakyRelu, network: Network):
        pass

    def visit_elu(self, op: Elu, network: Network):
        pass

    def visit_globalaveragepool(self, op: GlobalAveragePool, network: Network):
        pass

    def visit_gatherelements(self, op: GatherElements, network: Network):
        pass

    def visit_gemm(self, op: Gemm, network: Network):
        pass

    def visit_maxpool(self, op: MaxPool, network: Network):
        pass

    def visit_equal(self, op: Equal, network: Network):
        pass

    def visit_tile(self, op: Tile, network: Network):
        pass

    def visit_flatten(self, op: Flatten, network: Network):
        pass

    def visit_floor(self, op: Floor, network: Network):
        pass

    def visit_gru(self, op: GRU, network: Network):
        pass

    def visit_scatterelements(self, op: ScatterElements, network: Network):
        pass

    def visit_globallppool(self, op: GlobalLpPool, network: Network):
        pass

    def visit_greater(self, op: Greater, network: Network):
        pass

    def visit_hardsigmoid(self, op: HardSigmoid, network: Network):
        pass

    def visit_selu(self, op: Selu, network: Network):
        pass

    def visit_hardmax(self, op: Hardmax, network: Network):
        pass

    def visit_if(self, op: If, network: Network):
        pass

    def visit_min(self, op: Min, network: Network):
        pass

    def visit_instancenormalization(self, op: InstanceNormalization, network: Network):
        pass

    def visit_less(self, op: Less, network: Network):
        pass

    def visit_eyelike(self, op: EyeLike, network: Network):
        pass

    def visit_randomnormal(self, op: RandomNormal, network: Network):
        pass

    def visit_slice(self, op: Slice, network: Network):
        pass

    def visit_prelu(self, op: PRelu, network: Network):
        pass

    def visit_log(self, op: Log, network: Network):
        pass

    def visit_logsoftmax(self, op: LogSoftmax, network: Network):
        pass

    def visit_loop(self, op: Loop, network: Network):
        pass

    def visit_lpnormalization(self, op: LpNormalization, network: Network):
        pass

    def visit_matmul(self, op: MatMul, network: Network):
        pass

    def visit_reducel2(self, op: ReduceL2, network: Network):
        pass

    def visit_max(self, op: Max, network: Network):
        pass

    def visit_maxroipool(self, op: MaxRoiPool, network: Network):
        pass

    def visit_or(self, op: Or, network: Network):
        pass

    def visit_pad(self, op: Pad, network: Network):
        pass

    def visit_randomuniformlike(self, op: RandomUniformLike, network: Network):
        pass

    def visit_reciprocal(self, op: Reciprocal, network: Network):
        pass

    def visit_pow(self, op: Pow, network: Network):
        pass

    def visit_randomnormallike(self, op: RandomNormalLike, network: Network):
        pass

    def visit_onehot(self, op: OneHot, network: Network):
        pass

    def visit_randomuniform(self, op: RandomUniform, network: Network):
        pass

    def visit_concatfromsequence(self, op: ConcatFromSequence, network: Network):
        pass

    def visit_reducel1(self, op: ReduceL1, network: Network):
        pass

    def visit_reducelogsum(self, op: ReduceLogSum, network: Network):
        pass

    def visit_reducelogsumexp(self, op: ReduceLogSumExp, network: Network):
        pass

    def visit_reducemax(self, op: ReduceMax, network: Network):
        pass

    def visit_onehotencoder(self, op: OneHotEncoder, network: Network):
        pass

    def visit_isnan(self, op: IsNaN, network: Network):
        pass

    def visit_reducemean(self, op: ReduceMean, network: Network):
        pass

    def visit_reducemin(self, op: ReduceMin, network: Network):
        pass

    def visit_treeensembleregressor(self, op: TreeEnsembleRegressor, network: Network):
        pass

    def visit_reduceprod(self, op: ReduceProd, network: Network):
        pass

    def visit_reducesum(self, op: ReduceSum, network: Network):
        pass

    def visit_reducesumsquare(self, op: ReduceSumSquare, network: Network):
        pass

    def visit_relu(self, op: Relu, network: Network):
        pass

    def visit_reshape(self, op: Reshape, network: Network):
        pass

    def visit_shape(self, op: Shape, network: Network):
        pass

    def visit_sigmoid(self, op: Sigmoid, network: Network):
        pass

    def visit_size(self, op: Size, network: Network):
        pass

    def visit_softmax(self, op: Softmax, network: Network):
        pass

    def visit_softplus(self, op: Softplus, network: Network):
        pass

    def visit_softsign(self, op: Softsign, network: Network):
        pass

    def visit_spacetodepth(self, op: SpaceToDepth, network: Network):
        pass

    def visit_tfidfvectorizer(self, op: TfIdfVectorizer, network: Network):
        pass

    def visit_split(self, op: Split, network: Network):
        pass

    def visit_imputer(self, op: Imputer, network: Network):
        pass

    def visit_sqrt(self, op: Sqrt, network: Network):
        pass

    def visit_squeeze(self, op: Squeeze, network: Network):
        pass

    def visit_topk(self, op: TopK, network: Network):
        pass

    def visit_sub(self, op: Sub, network: Network):
        pass

    def visit_sum(self, op: Sum, network: Network):
        pass

    def visit_shrink(self, op: Shrink, network: Network):
        pass

    def visit_tanh(self, op: Tanh, network: Network):
        pass

    def visit_transpose(self, op: Transpose, network: Network):
        pass

    def visit_unsqueeze(self, op: Unsqueeze, network: Network):
        pass

    def visit_upsample(self, op: Upsample, network: Network):
        pass

    def visit_svmclassifier(self, op: SVMClassifier, network: Network):
        pass

    def visit_xor(self, op: Xor, network: Network):
        pass

    def visit_acos(self, op: Acos, network: Network):
        pass

    def visit_asin(self, op: Asin, network: Network):
        pass

    def visit_atan(self, op: Atan, network: Network):
        pass

    def visit_cos(self, op: Cos, network: Network):
        pass

    def visit_sin(self, op: Sin, network: Network):
        pass

    def visit_tan(self, op: Tan, network: Network):
        pass

    def visit_multinomial(self, op: Multinomial, network: Network):
        pass

    def visit_scan(self, op: Scan, network: Network):
        pass

    def visit_compress(self, op: Compress, network: Network):
        pass

    def visit_constantofshape(self, op: ConstantOfShape, network: Network):
        pass

    def visit_maxunpool(self, op: MaxUnpool, network: Network):
        pass

    def visit_scatter(self, op: Scatter, network: Network):
        pass

    def visit_sinh(self, op: Sinh, network: Network):
        pass

    def visit_cosh(self, op: Cosh, network: Network):
        pass

    def visit_asinh(self, op: Asinh, network: Network):
        pass

    def visit_acosh(self, op: Acosh, network: Network):
        pass

    def visit_nonmaxsuppression(self, op: NonMaxSuppression, network: Network):
        pass

    def visit_atanh(self, op: Atanh, network: Network):
        pass

    def visit_sign(self, op: Sign, network: Network):
        pass

    def visit_erf(self, op: Erf, network: Network):
        pass

    def visit_where(self, op: Where, network: Network):
        pass

    def visit_nonzero(self, op: NonZero, network: Network):
        pass

    def visit_meanvariancenormalization(self, op: MeanVarianceNormalization, network: Network):
        pass

    def visit_stringnormalizer(self, op: StringNormalizer, network: Network):
        pass

    def visit_mod(self, op: Mod, network: Network):
        pass

    def visit_thresholdedrelu(self, op: ThresholdedRelu, network: Network):
        pass

    def visit_matmulinteger(self, op: MatMulInteger, network: Network):
        pass

    def visit_qlinearmatmul(self, op: QLinearMatMul, network: Network):
        pass

    def visit_convinteger(self, op: ConvInteger, network: Network):
        pass

    def visit_qlinearconv(self, op: QLinearConv, network: Network):
        pass

    def visit_quantizelinear(self, op: QuantizeLinear, network: Network):
        pass

    def visit_gathernd(self, op: GatherND, network: Network):
        pass

    def visit_dequantizelinear(self, op: DequantizeLinear, network: Network):
        pass

    def visit_isinf(self, op: IsInf, network: Network):
        pass

    def visit_roialign(self, op: RoiAlign, network: Network):
        pass

    def visit_sequencelength(self, op: SequenceLength, network: Network):
        pass

    def visit_bitshift(self, op: BitShift, network: Network):
        pass

    def visit_unique(self, op: Unique, network: Network):
        pass

    def visit_cumsum(self, op: CumSum, network: Network):
        pass

    def visit_round(self, op: Round, network: Network):
        pass

    def visit_dynamicquantizelinear(self, op: DynamicQuantizeLinear, network: Network):
        pass

    def visit_range(self, op: Range, network: Network):
        pass

    def visit_det(self, op: Det, network: Network):
        pass

    def visit_scatternd(self, op: ScatterND, network: Network):
        pass

    def visit_sequenceempty(self, op: SequenceEmpty, network: Network):
        pass

    def visit_sequenceconstruct(self, op: SequenceConstruct, network: Network):
        pass

    def visit_sequenceinsert(self, op: SequenceInsert, network: Network):
        pass

    def visit_sequenceat(self, op: SequenceAt, network: Network):
        pass

    def visit_sequenceerase(self, op: SequenceErase, network: Network):
        pass

    def visit_splittosequence(self, op: SplitToSequence, network: Network):
        pass

    def visit_arrayfeatureextractor(self, op: ArrayFeatureExtractor, network: Network):
        pass

    def visit_binarizer(self, op: Binarizer, network: Network):
        pass

    def visit_categorymapper(self, op: CategoryMapper, network: Network):
        pass

    def visit_dictvectorizer(self, op: DictVectorizer, network: Network):
        pass

    def visit_featurevectorizer(self, op: FeatureVectorizer, network: Network):
        pass

    def visit_labelencoder(self, op: LabelEncoder, network: Network):
        pass

    def visit_linearclassifier(self, op: LinearClassifier, network: Network):
        pass

    def visit_linearregressor(self, op: LinearRegressor, network: Network):
        pass

    def visit_normalizer(self, op: Normalizer, network: Network):
        pass

    def visit_svmregressor(self, op: SVMRegressor, network: Network):
        pass

    def visit_scaler(self, op: Scaler, network: Network):
        pass

    def visit_treeensembleclassifier(self, op: TreeEnsembleClassifier, network: Network):
        pass

    def visit_zipmap(self, op: ZipMap, network: Network):
        pass

