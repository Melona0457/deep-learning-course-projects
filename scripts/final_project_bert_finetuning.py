
# Google Drive mount removed for public repo. Configure local data paths before running.



import os

project_root = 'data'

os.chdir(project_root + '/part1')



print(os.listdir())  # ['.DS_Store', 'final-project_finetuning _w_BERT.ipynb', 'data'] 정도 나오면 OK



import os


!python3 -m pip install pandas

!python3 -m pip install transformers


import os

from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup

import torch

from torch import nn

from torch.utils.data import DataLoader, Dataset

import pandas as pd

from tqdm import tqdm

import random

from torch.optim import AdamW

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import torch.nn.functional as F





%env CUDA_VISIBLE_DEVICES = 0



if torch.cuda.is_available() is True:

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


def load_imdb_data(data_file_path):

    if os.path.exists(data_file_path):

        df = pd.read_csv(data_file_path)

        texts = df['review'].tolist()

        labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]

        return texts, labels

    else:

        raise FileNotFoundError(f"The file '{data_file_path}' does not exist.")



data_file_path = './data/IMDB Dataset Train.csv'

texts, labels = load_imdb_data(data_file_path)


class CustomTextClassificationDataset(Dataset):

    def __init__(

        self,

        texts,

        labels,

        tokenizer,

        max_seq_length,

        use_augmentation: bool = False,   # train에서만 True로 사용

        aug_prob: float = 0.7,            # 한 샘플에 증강을 적용할 확률

        aug_mode: str = "mild",           # "mild" 또는 "strong"

    ):

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length



        self.use_augmentation = use_augmentation

        self.aug_prob = aug_prob

        self.aug_mode = aug_mode



    def __len__(self):

        return len(self.texts)



    # -----------------------------

    # 내부에서 사용할 텍스트 증강 함수

    # -----------------------------

    def _augment(self, text: str) -> str:

        """

        간단 텍스트 증강:

          - word dropout

          - random swap

          - (strong 모드에서만) 문장 일부만 사용하는 truncate

        외부 데이터/사전은 전혀 사용하지 않음.

        """

        words = text.split()

        if len(words) <= 3:

            # 너무 짧으면 증강하지 않고 원문 사용

            return text



        # 1) 모드별 파라미터 설정

        if self.aug_mode == "strong":

            word_drop_prob = 0.2   # 20% 단어 제거

            swap_prob = 0.2        # 20% 인접 단어 스왑

            truncate_prob = 0.3    # 30% 확률로 문장 일부만 사용

        else:

            # 기본: mild

            word_drop_prob = 0.1   # 10% 단어 제거

            swap_prob = 0.1        # 10% 인접 단어 스왑

            truncate_prob = 0.0    # 잘라내기는 하지 않음



        # 2) word dropout

        kept = []

        for w in words:

            # random.random() ∈ [0,1)

            if random.random() > word_drop_prob:

                kept.append(w)



        if len(kept) < 3:

            # 너무 많이 날아가면 원래 문장 사용

            kept = words



        words = kept



        # 3) random swap (인접 단어 위치 바꾸기)

        i = 0

        while i < len(words) - 1:

            if random.random() < swap_prob:

                words[i], words[i + 1] = words[i + 1], words[i]

                i += 2

            else:

                i += 1



        # 4) truncate (strong 모드에서만 일부만 사용)

        if truncate_prob > 0 and len(words) > 10 and random.random() < truncate_prob:

            # 문장의 60%만 사용

            cut_len = int(len(words) * 0.6)

            # 앞/중간/뒤 어느 쪽이든 랜덤으로 시작 위치 결정

            start = random.randint(0, len(words) - cut_len)

            words = words[start:start + cut_len]



        return " ".join(words)



    def __getitem__(self, idx):

        text = self.texts[idx]

        label = self.labels[idx]



        # train에서만, 일정 확률로 증강 적용

        if self.use_augmentation and random.random() < self.aug_prob:

            text = self._augment(text)



        encoding = self.tokenizer(

            text,

            add_special_tokens=True,

            max_length=self.max_seq_length,

            padding='max_length',

            truncation=True,

            return_attention_mask=True,

            return_tensors='pt'

        )



        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'label': torch.tensor(label, dtype=torch.long)

        }



class CustomBERTClassifier(nn.Module):

    def __init__(self, bert_model_name, num_classes):

        super(CustomBERTClassifier, self).__init__()

        # 사전학습된 BERT 불러오기

        self.bert = BertModel.from_pretrained(bert_model_name)



        ######################## TO-DO ########################

        # BERT의 hidden size 가져오기 (예: 768)

        hidden_size = self.bert.config.hidden_size



        # 과적합 방지를 위한 dropout 레이어

        self.dropout = nn.Dropout(p=0.05)



        # 최종 분류기: [CLS] 토큰의 벡터 -> 클래스 개수만큼의 로짓

        self.classifier = nn.Linear(hidden_size, num_classes)

        ######################## TO-DO ########################



    def forward(self, input_ids, attention_mask):

        # BERT의 출력: last_hidden_state, pooler_output 등 포함

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)



        # pooler_output: [CLS] 토큰에 해당하는 문장 표현 (배치 x hidden_size)

        pooled_output = outputs.pooler_output



        ######################## TO-DO ########################

        # dropout 적용

        pooled_output = self.dropout(pooled_output)



        # 최종 로짓 계산

        logits = self.classifier(pooled_output)

        ######################## TO-DO ########################



        return logits



def train_model(model, data_loader, optimizer, scheduler, device):

    model.train()

    for batch in tqdm(data_loader, desc="Train"):

        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)

        attention_mask = batch['attention_mask'].to(device)

        labels = batch['label'].to(device)



        # 모델 forward → logits

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)



        ######################## TO-DO ########################

        # 다중분류용 Cross Entropy Loss

        loss = F.cross_entropy(outputs, labels)

        ######################## TO-DO ########################



        loss.backward()

        optimizer.step()

        scheduler.step()





def evaluate_model(model, data_loader, device):

    model.eval()

    predictions = []

    actual_labels = []



    with torch.no_grad():

        for batch in tqdm(data_loader, desc="Validation"):

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            labels = batch['label'].to(device)



            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # outputs: logits → 가장 큰 값의 인덱스를 예측 라벨로 사용

            _, preds = torch.max(outputs, dim=1)



            predictions.extend(preds.cpu().tolist())

            actual_labels.extend(labels.cpu().tolist())



    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)



# Set up parameters

# Hint: generally, 5 ~ 10 epochs will be enough.

bert_model_name = 'bert-base-uncased'

num_classes = 2



######################## TO-DO ########################

max_seq_length = 512      # 리뷰가 좀 길어서 256 토큰 정도로 자르기

batch_size = 16           # BERT-base + seq_len 256 기준으로 코랩에서 안정적인 배치 크기

num_epochs = 3            # 기본 5 epoch; 성능 보고 6~8로 늘려도 됨

learning_rate = 2e-5      # BERT 파인튜닝에서 많이 쓰는 lr (2e-5 ~ 3e-5)



dropout = 0.05

weight_decay = 0.05

######################## TO-DO ########################



######################## DO NOT CHANGE ########################

train_texts, val_texts, train_labels, val_labels = \

train_test_split(texts, labels, test_size=0.2, random_state=42)

######################## DO NOT CHANGE ########################



# ---- 증강 설정 ----

USE_AUGMENTATION = True        # 증강을 사용하려면 True, 끄려면 False

AUG_MODE = "mild"            # "mild" 또는 "strong"

AUG_PROB = 0.7                 # 각 샘플에 증강을 적용할 확률



tokenizer = BertTokenizer.from_pretrained(bert_model_name)



# train: 증강 ON (옵션에 따라)

train_dataset = CustomTextClassificationDataset(

    train_texts,

    train_labels,

    tokenizer,

    max_seq_length,

    use_augmentation=USE_AUGMENTATION,

    aug_prob=AUG_PROB,

    aug_mode=AUG_MODE,

)



# validation: 절대 증강 X

val_dataset = CustomTextClassificationDataset(

    val_texts,

    val_labels,

    tokenizer,

    max_seq_length,

    use_augmentation=False,   # 항상 False

    aug_prob=0.0,

    aug_mode="mild",

)



train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size)



model = CustomBERTClassifier(bert_model_name, num_classes).to(device)



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=learning_rate,

    weight_decay=weight_decay

)



total_steps = len(train_dataloader) * num_epochs



# warmup 비율(보통 0~0.1). 너는 지금 warmup=0이었으니,

# 공정 비교를 위해 0으로 시작하거나, 성능 올리고 싶으면 0.1도 실험 가능.

warmup_ratio = 0.1

num_warmup_steps = int(total_steps * warmup_ratio)



scheduler = get_cosine_schedule_with_warmup(

    optimizer,

    num_warmup_steps=num_warmup_steps,

    num_training_steps=total_steps

)



# ==== 이번 실험(run)의 설정 기록 ====

experiment_config = {

    "bert_model_name": bert_model_name,

    "max_seq_length": max_seq_length,

    "batch_size": batch_size,

    "num_epochs": num_epochs,

    "learning_rate": learning_rate,

    "dropout": model.dropout.p,   # CustomBERTClassifier의 dropout

    "weight_decay": 0.01,         # optimizer에서 쓴 값이랑 맞추기

}



# ==== 실험 로그 저장 유틸 ====

import os, csv, json

from datetime import datetime



LOG_PATH = "experiment_log.csv"   # part1 폴더에 생성됨



def log_experiment(config, val_acc_per_epoch, best_val_acc, best_epoch, notes=""):

    """

    config: 하이퍼파라미터 dict

    val_acc_per_epoch: epoch마다의 validation accuracy 리스트

    best_val_acc: 최고 validation accuracy

    best_epoch: 최고 성능이 나온 epoch 번호(1부터 시작)

    notes: 자유롭게 적는 메모 (예: "baseline", "lr=3e-5" 등)

    """

    file_exists = os.path.isfile(LOG_PATH)



    with open(LOG_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        # 처음 만들 때만 헤더 쓰기

        if not file_exists:

            writer.writerow([

                "timestamp",

                "bert_model_name",

                "max_seq_length",

                "batch_size",

                "num_epochs",

                "learning_rate",

                "dropout",

                "weight_decay",

                "best_val_acc",

                "best_epoch",

                "val_acc_per_epoch",

                "notes",

            ])



        writer.writerow([

            datetime.now().isoformat(timespec="seconds"),

            config["bert_model_name"],

            config["max_seq_length"],

            config["batch_size"],

            config["num_epochs"],

            config["learning_rate"],

            config["dropout"],

            config["weight_decay"],

            best_val_acc,

            best_epoch,

            json.dumps(val_acc_per_epoch),  # 리스트를 문자열로 저장

            notes,

        ])



    print(f"[LOG] Appended to {LOG_PATH}")



# Train model and save best model + 로그 남기기



model_path = './finetuned_bert.pth'

eval_acc = 0.0

best_epoch = 0          # NEW: 최고 성능 epoch 기억

val_acc_history = []    # NEW: epoch별 validation accuracy 기록용 리스트



for epoch in range(num_epochs):

    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_model(model, train_dataloader, optimizer, scheduler, device)

    accuracy, report = evaluate_model(model, val_dataloader, device)



    print(f"Validation Accuracy: {accuracy:.4f}")

    print(report)



    # NEW: 히스토리에 현재 epoch의 val acc 추가

    val_acc_history.append(float(accuracy))



    if eval_acc < accuracy:

        torch.save(model.state_dict(), model_path)

        print('Saved Trained Model.')

        eval_acc = float(accuracy)

        best_epoch = epoch + 1



# NEW: 한 run이 끝난 뒤 로그 파일에 기록

log_experiment(

    config=experiment_config,

    val_acc_per_epoch=val_acc_history,

    best_val_acc=eval_acc,

    best_epoch=best_epoch,

    notes="Augment7, cosine, wd=dropout=0.05, mode=mild, prob=0.7"

)



