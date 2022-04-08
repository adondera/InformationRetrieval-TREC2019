import pandas as pd
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from sklearn.metrics import accuracy_score


class BertBinaryClassification:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device("cpu")
            print('No GPU available, using the CPU instead.')
        self.model.to(self.device)

    def forward(self, batch):
        output = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        return output

    def evaluate(self, dataset, batch_size):
        self.model.eval()
        y_predicted = []
        y_true = []
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _, batch in enumerate(tqdm.tqdm(eval_loader)):
            output = self.forward(batch)
            predictions = torch.argmax(output.logits, dim=-1).cpu().numpy()
            true_labels = batch['labels'].cpu().numpy()
            y_predicted.append(predictions)
            y_true.append(true_labels)
        return accuracy_score(np.concatenate(y_true), np.concatenate(y_predicted))


class MsMarcoDataset(Dataset):
    def __init__(self, train_data: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(self.train_data[0].to_list() * 2,
                                        self.train_data[1].to_list() + self.train_data[2].to_list(),
                                        truncation=True, padding=True, max_length=512,
                                        return_tensors='pt', return_token_type_ids=True)
        print("Finished encoding documents")

        total_samples = len(self.train_data[0])
        self.labels = torch.concat(
            [torch.ones(total_samples), torch.zeros(total_samples)]).type(torch.LongTensor)

    def __getitem__(self, index) -> T_co:
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

    def to_device(self, device):
        for item in self.encodings:
            self.encodings[item] = self.encodings[item].to(device)
        self.labels = self.labels.to(device)

model_name = './distilbert_final_model'
tokenizer_name = 'distilbert-base-uncased'
train_file = 'train/triples.train.small.tsv'
train_data = pd.read_csv(train_file, header=None, sep='\t', nrows=1000).dropna()
for i in range(3):
    train_data[i] = train_data[i].map(lambda s: s.encode('latin1').decode('utf8'))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
d = MsMarcoDataset(train_data, tokenizer)

# Training procedure
bert = BertBinaryClassification(model, {})
batch_size = 4
train_data_loader = DataLoader(d, batch_size=batch_size, shuffle=True)
d.to_device(bert.device)

losses = []
total_loss = 0
# optim = AdamW(bert.model.parameters(), lr=5e-5)

print(f"Accuracy before training: {bert.evaluate(d, batch_size)}")

# bert.model.train()
# for index, batch in enumerate(tqdm.tqdm(train_data_loader)):
#     optim.zero_grad()
#     loss = bert.forward(batch).loss
#     total_loss += loss.item()
#     losses.append(total_loss / (index + 1))
#     loss.backward()
#     optim.step()

# print(f"Accuracy after training: {bert.evaluate(d, batch_size)}")

bert.model.save_pretrained('distilbert_final_model')
plt.plot(losses)
plt.show()
