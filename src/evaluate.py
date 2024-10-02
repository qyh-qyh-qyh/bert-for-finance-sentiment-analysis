import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

import pandas as pd

import configparser

config=configparser.ConfigParser()
config.read('config.ini')

pretrained_model_name_or_path=config.get("eval","pretrained_model_name_or_path")
pretrained_tokenizer_name_or_path=config.get("eval","pretrained_tokenizer_name_or_path")
eval_dataset_path=config.get("eval","eval_dataset_path")
trained_model_weights_path=config.get("eval","trained_model_weights_path")

model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
tokenizer=AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
model.load_state_dict(torch.load(trained_model_weights_path))
model.eval()

def transform_dataset(dataset):
    return tokenizer(dataset['title'],padding="max_length",truncation=True)

def evaluate(eval_dataset_path):
    #model=AutoModelForSequenceClassification.from_pretrained("bert-base-chinese")
    #tokenizer=AutoTokenizer.from_pretrained("bert-base-chinese")
    #model.load_state_dict(torch.load("model_weights.pth"))
    #model.eval()

    df=pd.read_csv(eval_dataset_path)
    df_eval_data=pd.DataFrame()
    df_eval_data['title']=df["标题"]
    df_eval_data['labels']=df["正负面"]

    eval_dataset=datasets.Dataset.from_pandas(df_eval_data)

    eval_dataset=eval_dataset.map(transform_dataset)

    #eval_dataser=eval_dataset.with_format('torch')

    predictions=[]
    labels=df_eval_data['labels'].tolist()
    for batch in eval_dataset:
        inputs=tokenizer(batch['title'],padding="max_length",truncation=True,return_tensors='pt')
        #print(type(inputs))
        with torch.no_grad():
            outputs=model(**inputs)
        logits=outputs.logits
        predicted_idx=torch.argmax(logits,dim=1)
        predictions.extend(predicted_idx.tolist())

    f1=f1_score(labels,predictions,average='binary')
    print(f'F1-score:{f1}')

if __name__=="__main__":
    evaluate(eval_dataset_path)