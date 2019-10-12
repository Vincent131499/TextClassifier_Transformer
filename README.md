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
<br> `下面将针对以上两个模式的运行方式进行详细说明。`<br>
## 模式1：线下实时预测
### Step1：数据准备
