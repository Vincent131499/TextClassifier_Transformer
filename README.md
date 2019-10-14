# TextClassifier_BERT
个人基于谷歌开源的BERT编写的文本分类器（基于微调方式），可自由加载NLP领域知名的预训练语言模型BERT、
Bert-wwm、Roberta、ALBert以及ERNIE1.0.<br>
该项目支持两种预测方式：<br>
（1）线下实时预测<br>
（2）服务端实时预测<br>
## 运行环境
* Python3.6+<br>
* Tensorflow1.10+/Tensorflow-gpu1.10+<br>
## 项目说明
主要分为两种运行模式：<br>
模式1：线下实时预测<br>
step1:数据准备<br>
step2:模型训练<br>
step3:模型导出<br>
step4:线下实时预测<br>
模式2：服务端实时预测
step1:数据准备<br>
step2:模型训练<br>
step3:模型转换<br>
step4:服务部署<br>
step5:api请求-预测<br>
### 注意事项
1.如果你只是想体验从模型训练到本地线下预测这一套流程，只需要按照模式1依次执行即可<br>
2.若你想想体验从模型训练到模型部署整个流程，则需要按照模式2依次执行<br>
<br> 下面将针对以上两个模式的运行方式进行详细说明。<br>
## 模式1：线下实时预测
### Step1：数据准备
为了快速实验项目效果，这里使用了样本规模较小的手机评论数据，数据比较简单，有三个分类：-1（差评）、0（中评）、1（好评），数据样例如下所示：<br>
![数据描述](https://github.com/Vincent131499/TextClassifier_BERT/raw/master/imgs/dataset_desc.jpg)
ps:本项目中已将其拆分成了train.tsv、dev.txv、test.tsv三个文件<br>
### Step2:模型训练
训练命令：<br>
```Bash
bash train.sh
```
详细说明：训练模型直接使用bert微调的方式进行训练，对应的程序文件为run_classifier_serving.py。关于微调bert进行训练的代码网上介绍的
很多，这里就不一一介绍。主要是创建针对该任务的Processor即：SentimentProcessor，在这个processor的_create_examples()和get_labels()函数自定义，如下所示：
```Python
class SetimentProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""

    """
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label_list.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as fd:
            pickle.dump(self.labels, fd)
    """
    return ["-1", "0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0: 
        continue
      guid = "%s-%s" % (set_type, i)

      #debug (by xmxoxo)
      #print("read line: No.%d" % i)

      text_a = tokenization.convert_to_unicode(line[1])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples
```
<br>注意，此处作出的一个特别变动之处是在conver_single_example()函数中增加了一段保存label的代码，在训练过程中在保存的模型路径下生成label2id.pkl文件，代码如下所示：<br>
```Python
#--- save label2id.pkl ---
#在这里输出label2id.pkl , add by stephen 2019-10-12
output_label2id_file = os.path.join(FLAGS.output_dir, "label2id.pkl")
if not os.path.exists(output_label2id_file):
   with open(output_label2id_file,'wb') as w:
      pickle.dump(label_map,w)
#--- Add end ---
```
### Step3：模型导出
运行如下命令：
```Bash
bash export.sh
```
会在指定的exported目录下生成以一个时间戳命名的模型目录。
详细说明：在run_classifier_serving中定义serving_input_fn()函数，如下：
```Python
def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })
    return input_fn
```
继而在run_classifier_serving中定义do_export选项：
```Python
if do_export:
   estimator._export_to_tpu = False
   estimator.export_savedmodel(Flags.export_dir, serving_input_fn)
```
### Step4：线下实时预测
运行test_serving.py文件，即可进行线下实时预测。<br>
运行效果如下所示：<br>
![运行效果图]<br>
## 模式2：服务端实时预测
===TODO===
