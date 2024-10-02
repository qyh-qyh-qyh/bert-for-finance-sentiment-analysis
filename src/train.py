import torch
import datasets
from transformers import AutoModelForSequenceClassification,BertTokenizerFast
from transformers import TrainingArguments,Trainer

from accelerate import Accelerator

import pandas as pd
import numpy as np
import os

import configparser

config=configparser.ConfigParser()

config.read("config.ini")

pretrained_model_name_or_path=config.get('train','pretrained_model_name_or_path')
pretrained_tokenizer_name_or_path=config.get("train","pretrained_tokenizer_name_or_path")
train_dataset_path=config.get('train','train_dataset_path')
output_dir=config.get("train","output_dir")
per_device_train_batch_size=int(config.get("train","per_device_train_batch_size"))
num_train_epochs=float(config.get("train","num_train_epochs"))
learning_rate=float(config.get("train","learning_rate"))
warmup_radio=float(config.get("train","warmup_radio"))
weight_decay=float(config.get("train","weight_decay"))
model_weights_path=config.get("train","model_weights_path")

model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer_name_or_path)

def transform_dataset(dataset):
    #填充是为了确保向量输入长度都一致，截断能有效限制输入的文本,第一个参数表示具体的列要截断的列，一般是包含文字的列
    return tokenizer(dataset['title'],padding="max_length",truncation=True)

def train(train_dataset_path):
    #model=AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2)
    #tokenizer=BertTokenizerFast.from_pretrained("bert-base-chinese")

    df_total_data=pd.read_excel(train_dataset_path)
    df_valid_data=pd.DataFrame()
    df_valid_data["labels"]=df_total_data["正/负面"]
    df_valid_data["title"]=df_total_data["标题"]


    train_dataset=datasets.Dataset.from_pandas(df_valid_data)

    #必须要这样，不然其实直接调用函数处理数据集得到的是一个字典，每个键对应多个输入的值，而用map之后返回的是一个列表，里面是多个字典
    train_dataset=train_dataset.map(transform_dataset)
    #print(train_dataset[0].ids)
    #sample=next(iter(train_dataset))
    #print(sample)

    #train_dataset=train_dataset.with_format('torch')

    training_args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        #per_device_val_batch_size=64,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_radio,
        weight_decay=weight_decay,
        fp16=True
    )

    trainer=Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args
    )

    trainer.train()

    torch.save(model.state_dict(),model_weights_path)

if __name__=="__main__":
    train(train_dataset_path)
