�	L���A@L���A@!L���A@	nW�8z��?nW�8z��?!nW�8z��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6L���A@ݳ�ђ@1���Hh7<@A_`V(���?I���4�?Y���T��?*	��S㥻�@2F
Iterator::Model!V�a �?!_�A�[�W@)y�ѩk�?1���_W@:Preprocessing2U
Iterator::Model::ParallelMapV2��*Q���?!2Ԉ�@)��*Q���?12Ԉ�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���a���?!��ͳ7@)܄{eު�?1��:�v�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice=�q�?!�{�pU3�?)=�q�?1�{�pU3�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��A_z��?!�!�F�@)cG�P��?1�7��o��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��mRѴ?!��Aj@)���B{?1�Z�����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�xy:w?!������?)�xy:w?1������?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapEb����?!�g���#@)�4�;�h?1Z�mzBY�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9oW�8z��?IlG��I4@Qv�i���S@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ݳ�ђ@ݳ�ђ@!ݳ�ђ@      ��!       "	���Hh7<@���Hh7<@!���Hh7<@*      ��!       2	_`V(���?_`V(���?!_`V(���?:	���4�?���4�?!���4�?B      ��!       J	���T��?���T��?!���T��?R      ��!       Z	���T��?���T��?!���T��?b      ��!       JGPUYoW�8z��?b qlG��I4@yv�i���S@�"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput0�N�?!0�N�?0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�h� �?!�$P��?0":
sequential/conv2d_1/Relu_FusedConv2DM��>�g�?!ۢ�a���?"5
sequential/dense/MatMulMatMulz�r}�!�?!
������?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul�-��3��?!辤�b��?"C
%gradient_tape/sequential/dense/MatMulMatMul���^��?!��3���?0"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdama[&�?!z��!K1�?"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�%p��?!��P�3��?0"6
sequential/conv2d/Conv2DConv2D�����6�?!-����c�?0"_
>gradient_tape/sequential/average_pooling2d/AvgPool/AvgPoolGradAvgPoolGrad�����,�?!"#�R��?Q      Y@Y�Vg�bn1@aG*�Vg�T@q�����@y]�D�Q�?"�	
both�Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 