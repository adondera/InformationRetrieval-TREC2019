import spacy
import torch
import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from numba import jit


@jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    assert (u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta


nlp = spacy.load('en_core_web_sm')

eval_set = pd.read_csv('eval_set.tsv', sep='\t', header=None)

bert_model_name = './distilbert_final_model'
tokenizer_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

model.eval()
with open('intermediate_results/testing_desm_scores_distilbert_finetuned.tsv', 'w') as f:
    for _, (q_id, q_str, doc_id, doc_str) in tqdm.tqdm(eval_set.iterrows()):
        x = tokenizer(q_str, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
        query_cls = model(**x, output_hidden_states=True).hidden_states[-1][:, 0].view(-1).cpu().detach().numpy()
        sentences = [s.text for s in nlp(doc_str[7:-7]).sents][1:]
        doc_score = np.zeros(model.config.dim)
        for sent in sentences:
            x = tokenizer(sent, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            doc_score += model(**x, output_hidden_states=True).hidden_states[-1][:, 0].view(-1).cpu().detach().numpy()
        try:
            doc_score /= len(sentences)
            sim = cosine_similarity_numba(query_cls, doc_score)
        except:
            sim = 0
        f.write(f'{q_id}\t{doc_id}\t{sim}\n')
