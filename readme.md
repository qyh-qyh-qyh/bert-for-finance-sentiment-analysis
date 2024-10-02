### Title

---

finance-sentiment-analysis(金融情感分析)

### Introduce

---

background:首先，这是一份礼物，其次是南开大学金融科技课程的HW2  
       项目作用：本项目主要针对金融文章的标题或者新闻进行情感分析，初步判断文章内容是积极的或是消极的  
       项目技术简介：本项目第一步训练**微调了一个预训练好的bert模型**：使用预训练大模型（很小的大模型，但对laptop很大）本身具备的语义理解能力，搜集带有标签的数据集进行微调，让大模型适应金融情感分析这一下游任务，调节batch-size（批次大小）,fp16=True（是否将模型从float32转化为float16）在本地电脑提高训练速度的同时又不损失模型能力，调节batch-size,num_train_epochs（训练轮次）以提高模型的泛化能力为微调目的，训练效果图如下：![训练过程图](D:\life\grade_three_1\nku_sentiment_analysis\readme.assets\训练效果图-17278711652152.png)

本项目第二步**使用模型进行推理，验证模型在验证集上的效果**，完成了模型推理以及计算模型在验证上的F1分数,效果如下：![模型在验证集上的效果](D:\life\grade_three_1\nku_sentiment_analysis\readme.assets\验证集效果.png)

本项目第三步使用**pdfplumber库读取pdf文件**，因为文本文件一般信息量较大并且其中表示金融相关的情感信息含量较少，经过对一定数量的相关金融文章研究，发现体现金融相关的情感信息几乎全部出现在包含具体数据的句子中，**故我们通过判断文章中含有数字的句子的积极消极情感来判断文章的积极和消极情感**,因为消极情感的句子在金融文章中往往更能说明问题，同时中性的含有数字的句子我们往往更加容易判断成积极的，故我们制定一个规则:**如果积极情感句子数目不大于五倍的消极情感数目，则文章情感为消极，否则文章情感为积极**，经过以上分析，我们**采用re库提取出文章中含有数字的句子**,然后用模型进行推理，得到最终结果，存储在result.xlsx中  
       **PS：result.xlsx文件中每一行按照顺序对应了pdf_files文件中的每一行，每一列从左到右分别代表文章中含有数字的句子总数，积极的句子数，消极的句子数，文章情感分析结果（0代表消极，1代表积极）
       
### 目录结构

---

```python
.
├── config.ini		#储存一些路径信息和配置信息，用于软编码需要
├── data			#储存数据文件，news_seed为训练集，test_data为数据集
│   ├── news_seed.xlsx
│   └── test_data.csv
├── model_weights.pth		#训练好的模型权重
├── pdf_files		#要处理的目标文件
├── requirements.txt		#项目的相关依赖文件
├── result.xlsx		#结果文件
├── src			#源码
│   ├── evaluate.py
│   ├── extract_pdf_and_inference.py
│   └── train.py
└── weibo_senti_100k.csv
```

### Quick start

---

可以fork本仓库，使用git clone 复制该项目文件到本地  
       在本地项目目录下，pip install -r requirements.txt即可完成环境配置  
       然后自行替换pdf_files中的文件作为你要分析的目标文件，在该项目目录下命令行输入 python src/extract_pdf_and_inference.py即可对自己pdf_files下的所有文件进行情感分析，结果会覆盖掉result.xlsx中的原有文件
       **PS:相关配置请看config.ini文件，如果要提升训练效果请自行阅读训练和验证源码并查阅资料，在config,ini文件中修改训练配置**
       
### TO-DO

---

1. 有时间或者学习到了什么更好的方法进一步优化模型提高模型推理效果
2. 实现ipynb代码（据说是要提交ipynb文件）