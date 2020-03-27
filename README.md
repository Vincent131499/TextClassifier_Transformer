# TextClassifier_Transformer
个人基于谷歌开源的BERT编写的文本分类器（基于微调方式），可自由加载NLP领域知名的预训练语言模型BERT、
Roberta、ALBert及其wwm版本，同时适配ERNIE1.0.<br>
该项目支持两种预测方式：<br>
（1）线下实时预测<br>
（2）服务端实时预测<br>

## 新增改动
2020-03-25:<br>
(1)项目名由'TextClassifier_BERT'更改为'TextClassifier_Transformer';<br>
(2)新增ELECTRA、AlBert两个预训练模型。<br>
**注意：在使用AlBert时，请将该项目下的modeling.py文件更新为ALBert项目中下的modeling.py，而后在运行**<br>
2020-03-04:<br>
模型部署增加tf-serving机制，具体实施方式见[This Blog](https://Vincent131499.github.io/2020/02/28/以BERT分类为例阐述模型部署关键技术)

## 运行环境
* Python3.6+<br>
* Tensorflow1.10+/Tensorflow-gpu1.10+<br>
<br>
提供知名的预训练语言模型下载地址（其中百度开源的Ernie模型已转换成tf格式）：<br>
Bert-base:链接：https://pan.baidu.com/s/18h9zgvnlU5ztwaQNnzBXTg 提取码：9r1z <br>
Roberta:链接：https://pan.baidu.com/s/1FBpok7U9ekYJRu1a8NSM-Q 提取码：i50r <br>
Bert-wwm:链接：链接：https://pan.baidu.com/s/1lhoJCT_LkboC1_1YXk1ItQ 提取码：ejt7 <br>
ERNIE1.0:链接：链接：https://pan.baidu.com/s/1S6MI8rQyQ4U7dLszyb73Yw 提取码：gc6f <br>
ELECTRA-Tiny:链接：https://pan.baidu.com/s/11QaL7A4YSCYq4YlGyU1_vA 提取码：27jb  <br>
AlBert-base:链接：https://pan.baidu.com/s/1U7Zx73ngci2Oqp3SLaVOaw 提取码：uijw <br>
<br>

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
step5:应用端<br>
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
train.sh参数说明：
```Bash
export BERT_BASE_DIR=./chinese_roberta_zh_l12 #指定预训练的语言模型所在路径
export DATA_DIR=./dat #指定数据集所在路径
export TRAINED_CLASSIFIER=./output #训练的模型输出路径
export MODEL_NAME=mobile_0_roberta_base #训练的模型命名
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
export.sh参数说明：
```Bash
#以下四个参数应与train.sh中设置的值保持一致
export BERT_BASE_DIR=./chinese_roberta_zh_l12
export DATA_DIR=./dat
export TRAINED_CLASSIFIER=./output
export MODEL_NAME=mobile_0_roberta_base
```
会在指定的exported目录下生成以一个时间戳命名的模型目录。<br>
详细说明：run_classifier.py 主要设计为单次运行的目的，如果把 do_predict 参数设置成 True，倒也确实可以预测，但输入样本是基于文件的，并且不支持将模型持久化在内存里进行 serving，因此需要自己改一些代码，达到两个目的：<br>
（1）允许将模型加载到内存里，即：允许一次加载，多次调用。<br>
（2）允许读取非文件中的样本进行预测。譬如从标准输入流读取样本输入。<br>
* 将模型加载到内存里<br>
run_classifier.py 的 859 行加载了模型为 estimator 变量，但是遗憾的是 estimator 原生并不支持一次加载，多次预测。参见：https://guillaumegenthial.github.io/serving-tensorflow-estimator.html。
因此需要使用 estimator.export_saved_model() 方法把 estimator 重新导出成 tf.saved_model。
代码参考了 https://github.com/bigboNed3/bert_serving）,在run_classifier_serving中定义serving_input_fn()函数，如下：<br>
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
![运行效果图](https://github.com/Vincent131499/TextClassifier_BERT/raw/master/imgs/serving_offline.jpg)<br>
详细说明：导出模型后，就不需要第 859 行那个 estimator 对象了，可以自行从刚刚的导出模型目录加载模型，代码如下：<br>
```Python
predict_fn = tf.contrib.predictor.from_saved_model('/exported/1571054350')
```
基于上面的 predict_fn 变量，就可以直接进行预测了。下面是一个从标准输入流读取问题样本，并预测分类的样例代码：<br>
```Python
while True:
    question = input("> ")
    predict_example = InputExample("id", question, None, '某固定伪标记')
    feature = convert_single_example(100, predict_example, label_list,
                                        FLAGS.max_seq_length, tokenizer)
 
    prediction = predict_fn({
        "input_ids":[feature.input_ids],
        "input_mask":[feature.input_mask],
        "segment_ids":[feature.segment_ids],
        "label_ids":[feature.label_id],
    })
    probabilities = prediction["probabilities"]
    label = label_list[probabilities.argmax()]
    print(label)
```
## 模式2：服务端实时预测
首先针对该模式的基本架构进行说明：<br>
![服务端部署架构](https://github.com/Vincent131499/TextClassifier_BERT/raw/master/imgs/serving_deploy_arcticture.jpg)
<br>架构说明： <br>
BERT模型服务端：加载模型，进行实时预测的服务； 使用的是 BERT-BiLSTM-CRF-NER提供的bert-base；<br>
API服务端：调用实时预测服务，为应用提供API接口的服务，用flask编写；<br>
应用端：最终的应用端； 我这里为了简便，并没有编写网页，直接调用了api接口。<br>
### Step1:数据准备
同模式1中的Step1介绍。
### Step2:模型训练
同模式1中的Step2介绍。
### Step3:模型转换
运行如下命令：
```Bash
bash model_convert.sh
```
会在$TRAINED_CLASSIFIER/$EXP_NAME生成pb格式的模型文件<br>
model_convert.sh参数说明：
```Bash
export BERT_BASE_DIR=./chinese_roberta_zh_l12 #训练模型时使用的预训练语言模型所在路径
export TRAINED_CLASSIFIER=./output #训练好的模型输出的路径
export EXP_NAME=mobile_0_roberta_base #训练后保存的模型命名

python freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -max_seq_len 128 #注意，这里的max_seq_len应与训练的脚本train.sh设置的max_seq_length参数值保持一致
```
### Step4:模型部署
运行如下命令：
```Bash
bash bert_classify_server.sh 
```
提示：在运行这个命令前需要保证安装了bert-base这个库，使用如下命令进行安装：
```Bash
pip install bert-base
```
**注意**：<br>
port 和 port_out 这两个参数是API调用的端口号，默认是5555和5556,如果你准备部署多个模型服务实例，那一定要指定自己的端口号，避免冲突。
我这里是改为： 5575 和 5576<br>
如果报错没运行起来，可能是有些模块没装上,都是 bert_base/server/http.py里引用的，装上就好了：
```
sudo pip install flask 
sudo pip install flask_compress
sudo pip install flask_cors
sudo pip install flask_json
```
我这里的配置是2个GTX 1080 Ti，这个时候双卡的优势终于发挥出来了，GPU 1用于预测，GPU 0还可以继续训练模型。<br>
部署成功示例图如下：<br>
![部署成功示例图](https://github.com/Vincent131499/TextClassifier_BERT/raw/master/imgs/deploy-success.jpg)
### Step5:应用端
运行如下命令：
```Bash
python api/api_service_flask.py
```
即可通过指定api接口(本项目中是http://192.168.9.23:8910/predict_online?text=我好开心)访问部署的服务器。<br>
通过浏览器进行请求：<br>
![浏览器请求](https://github.com/Vincent131499/TextClassifier_BERT/raw/master/imgs/api_example.jpg)
