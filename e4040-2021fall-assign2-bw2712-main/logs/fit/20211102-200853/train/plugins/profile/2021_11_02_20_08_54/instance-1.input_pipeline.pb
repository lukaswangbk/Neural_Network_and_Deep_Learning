	r�@HdB@r�@HdB@!r�@HdB@	���E�?���E�?!���E�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6r�@HdB@��-��?@1e���
@A9�Վ��?I4K��r�?Y��f*�#�?*	/�$qe@2U
Iterator::Model::ParallelMapV2��~�Ϛ�?!��!��RF@)��~�Ϛ�?1��!��RF@:Preprocessing2F
Iterator::Model��ne�κ?!��QQ�N@)�����Μ?1�D`V�f0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV�@I�?!'����3@)PVW@�?18���R0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	À%W��?!����,@):vP��?1�.�*�_@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�K��$w�?!�N6�p�@)�K��$w�?1�N6�p�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorq9^��Iy?!x߰l�@)q9^��Iy?1x߰l�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!4)��zC@)��-�x?1p$���>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥛?!fqD=#{/@)�^�sa�g?1���Iu��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���E�?I{�p�3�V@Q-�k���!@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��-��?@��-��?@!��-��?@      ��!       "	e���
@e���
@!e���
@*      ��!       2	9�Վ��?9�Վ��?!9�Վ��?:	4K��r�?4K��r�?!4K��r�?B      ��!       J	��f*�#�?��f*�#�?!��f*�#�?R      ��!       Z	��f*�#�?��f*�#�?!��f*�#�?b      ��!       JGPUY���E�?b q{�p�3�V@y-�k���!@