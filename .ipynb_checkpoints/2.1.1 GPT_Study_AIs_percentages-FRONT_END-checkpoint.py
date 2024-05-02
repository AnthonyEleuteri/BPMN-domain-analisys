#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import ast
import re
import torch
import chardet
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from transformers import (GPT2Tokenizer, GPT2Model, 
                          set_seed,
                          training_args,
                          trainer,
                          GPT2Config,
                          get_cosine_schedule_with_warmup,
                          GPT2ForSequenceClassification)


# Funzione che dato un 'path' restituisce il suo encoding.  
def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        content=f.read()
        result = chardet.detect(content)
        return result['encoding']
        

set_seed(123)
epochs=4
batch_size = 32


# Numero massimo della sequenza
# La sequenza <80 avrà del padding, la sequenza >80 sarà troncata
max_length = 510

# Usiamo la cpu se la gpu non viene trova
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nome del trasformers model pre-allenato
model_name_or_path = 'gpt2'

# Dizionario delle etichette e il loro ID
labels_ids = {'Manufacturing': 0, 'Logistics':1, 'Public Administration': 2, 'Healthcare': 3, 'Education': 4}

# Numero di etichette che stiamo utilizzando
n_labels = len(labels_ids)


# In[2]:


from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    GPT2ForSequenceClassification
)

class BPMNDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.n_examples = len(texts)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {'text': self.texts[item]}


class Gpt2ClassificationCollator(object):
    def __init__(self, 
                 use_tokenizer, 
                 max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs

def train(model, dataloader, optimizer, scheduler, device):
    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.train()

    
    # Utilizzo tqdm per visualizzare una barra di avanzamento mentre itero sui batch
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        #loss=discrepanza tra le previsioni del modello e i valori reali dell'obiettivo (ground truth)
        #logits=appresentano le "probabilità" che il modello assegna a ciascuna classe di output
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #Aggiorno i pesi dell'ottimizzatore e lo scheduler
        optimizer.step()
        scheduler.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        
    avg_epoch_loss = total_loss / len(dataloader)
    
    return true_labels, predictions_labels


def validation(dataloader, device_, model):
    predicted_probabilities = []

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device_) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_probabilities.extend(probabilities.tolist())

    return predicted_probabilities


# In[3]:


class BPMNDomainDataset(Dataset):
    def __init__(self, path):
        if not os.path.isfile(path):
            raise ValueError('Invalid `path` variable! Needs to be a file')
        
        self.df = pd.read_csv(path, sep=';', engine='python', encoding=get_file_encoding(path))
        self.descriptions = self.df['Labels'].to_list()
        self.domains = self.df['CollectionName'].to_list()  
        self.flattened_domains = [label for sublist in self.domains for label in sublist.split(',')]    
        self.n_examples = len(self.descriptions)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {"text": self.descriptions[item], "label": self.flattened_domains[item]}


class Gpt2ClassificationCollatorDomain(object):
    def __init__(self, 
                use_tokenizer, 
                labels_encoder, 
                max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

    def __call__(self, sequences):
        texts = [sequence.get('text', None) for sequence in sequences]  # Use .get() with default None
        labels = [sequence.get('label', None) for sequence in sequences]  # Use .get() with default None
        label_ids = [self.labels_encoder[label] for label in labels]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs['labels'] = torch.tensor(label_ids)  
        return inputs


# In[4]:


from flask import Flask, render_template, request
from collections import defaultdict
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer


app = Flask(__name__)

# Funzione per l'elaborazione del testo
def elabora_testo(testo):

    print('Loading configuration and model...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                          num_labels=n_labels)
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model_config.pad_token_id = tokenizer.pad_token_id
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                                      config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print('Model loaded to `%s`'%device)




    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          max_sequence_len=max_length)
    gpt2_classification_collator_domain = Gpt2ClassificationCollatorDomain(use_tokenizer=tokenizer,
                                                                       labels_encoder=labels_ids,
                                                                       max_sequence_len=max_length)
    print('Dealing with Train...')
    train_dataset = BPMNDomainDataset(path='./AI_Generated_Datas/CombinedGeneratedWords.csv')
    print('Created `train_dataset` with %d examples!'%len(train_dataset))
    train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=gpt2_classification_collator_domain)  
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    print('Dealing with Validation...')
    valid_dataset = BPMNDataset(testo)  
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              collate_fn=gpt2_classification_collator)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))



    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)
    print('Epoch loop ...')
    for epoch in tqdm(range(epochs)):
        print('Training on batches...')
        train_labels, train_predict = train(model, train_dataloader, optimizer, scheduler, device)
        print('Validation on batches...')
        valid_predict = validation(valid_dataloader, device, model)

    # Mappa dei nomi delle etichette predette
    label_names = {
        0: 'Manufacturing',
        1: 'Logistics',
        2: 'Public Administration',
        3: 'Healthcare',
        4: 'Education'
    }

    # Inizializziamo un dizionario per accumulare i conteggi per tutte le etichette
    counts_all = defaultdict(int)

    # Iteriamo su ogni insieme di probabilità predette
    for pred_probabilities in valid_predict:
        for pred_label_idx, prob in enumerate(pred_probabilities):
            pred_label = label_names[pred_label_idx]
            counts_all[pred_label] += prob

    # Calcoliamo il totale delle probabilità
    total_probabilities = sum(sum(pred_probabilities) for pred_probabilities in valid_predict)

    # Costruiamo una stringa HTML con i risultati
    html_results = "<h1>Risultati</h1>"
    html_results += "<ul>"
    for label, name in label_names.items():
        count = counts_all[name]
        percentage = (count / total_probabilities) * 100
        html_results += f"<li>{name}: {percentage:.2f}%</li>"
    html_results += "</ul>"
    return html_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        testo = request.form['testo']
        risultati = elabora_testo(testo)
        return render_template('index.html', risultati=risultati)
    return render_template('index.html', risultati=None)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:


from collections import defaultdict

# Mappa dei nomi delle etichette predette
label_names = {
    0: 'Manufacturing',
    1: 'Logistics',
    2: 'Public Administration',
    3: 'Healthcare',
    4: 'Education'
}


# Inizializziamo un dizionario per accumulare i conteggi per tutte le etichette
counts_all = defaultdict(int)

# Iteriamo su ogni insieme di probabilità predette
for pred_probabilities in valid_predict:
    for pred_label_idx, prob in enumerate(pred_probabilities):
        pred_label = label_names[pred_label_idx]
        counts_all[pred_label] += prob

# Calcoliamo il totale delle probabilità
total_probabilities = sum(sum(pred_probabilities) for pred_probabilities in valid_predict)


print(valid_predict)
print("Percentage of predictions:")
for label, name in label_names.items():
    count = counts_all[name]
    percentage = (count / total_probabilities) * 100
    print(f"{name}: {percentage:.2f}%")


# In[ ]:




