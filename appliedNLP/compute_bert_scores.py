import torch
import tqdm
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')
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

data = pd.read_csv('test_set.tsv', sep='\t', header=None)

queries_to_bert_score = defaultdict(dict)
k = 3
with open('full_results_eval_set_bert.tsv', 'w') as f:
    for _, (query_id, query_str, doc_id, doc_text) in tqdm.tqdm(data.iterrows()):
        sentences = [s.text for s in nlp(doc_text[7:-7]).sents][1:]
        res = []
        for s in sentences:
            x = tokenizer(query_str, s, truncation=True, padding=True, max_length=512, return_tensors='pt').to(
                device)
            res.append(model(**x).logits[:, 1].item())
        res = sorted(res)[-k:]
        output_str = '\t'.join(str(x) for x in res)
        f.write(f'{query_id}\t{doc_id}\t{output_str}\n')
        # print(f'{query_id}\t{doc_id}\t{score}\n')
