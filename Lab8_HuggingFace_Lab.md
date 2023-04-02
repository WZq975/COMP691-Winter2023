# 🤗 A Gentle Introduction to HuggingFace (HF)
---
HuggingFace provides you with a variety of pretrained models and
functionalities to train/fine-tune these models and make inferences.

Their [datasets](https://huggingface.co/docs/datasets/quickstart) library gives you access to many common NLP datasets. You can visualize these datasets on their [platform](https://huggingface.co/datasets) to get a sense of the data you would be working with.


```python
!pip install datasets transformers
```

    Requirement already satisfied: datasets in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (2.10.1)
    Requirement already satisfied: transformers in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (4.27.3)
    Requirement already satisfied: numpy>=1.17 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (1.23.5)
    Requirement already satisfied: aiohttp in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (3.8.4)
    Requirement already satisfied: packaging in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (22.0)
    Requirement already satisfied: tqdm>=4.62.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (4.64.1)
    Requirement already satisfied: responses<0.19 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (0.18.0)
    Requirement already satisfied: multiprocess in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (0.70.14)
    Requirement already satisfied: requests>=2.19.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (2.28.1)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (2023.3.0)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.2.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (0.12.1)
    Requirement already satisfied: pyyaml>=5.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (6.0)
    Requirement already satisfied: pyarrow>=6.0.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (11.0.0)
    Requirement already satisfied: pandas in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (1.5.3)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (0.3.6)
    Requirement already satisfied: xxhash in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from datasets) (3.2.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from transformers) (0.13.2)
    Requirement already satisfied: regex!=2019.12.17 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from transformers) (2022.10.31)
    Requirement already satisfied: filelock in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from transformers) (3.9.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (6.0.4)
    Requirement already satisfied: attrs>=17.3.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (1.8.2)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (2.0.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from aiohttp->datasets) (1.3.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (4.4.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (1.26.14)
    Requirement already satisfied: idna<4,>=2.5 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (2022.12.7)
    Requirement already satisfied: python-dateutil>=2.8.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from pandas->datasets) (2022.7)
    Requirement already satisfied: six>=1.5 in /home/wa_ziqia/miniconda3/envs/deepl2023/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)


## 🌠 Our Goal
Our goal for this tutorial is to get familiar with the [transformers](https://huggingface.co/docs/transformers/index) library from HuggingFace and use a pretrained model to fine-tune it on a sequece classification task. More specifically we will fine-tune a [BERT](https://arxiv.org/pdf/1810.04805.pdf) model on the [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity#data-instances) dataset.
> The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review.

> The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored. Each class has 1,800,000 training samples and 200,000 testing samples.

Since the dataset is quite large, we will be working with only a subset of this dataset throughout this tutorial.


## 🪜 Main Components
The main components we would need to develop to realize our goal are:

1. Load the data and make a dataset object for this task.
2. Write a collate function/class to tokenize/transform/truncate batches of inputs.
3. Make a custom model, which uses a pretrained model as its backbone and it is designed for our current task at hand.
4. Write the training loop and train the model.

> ⚠️ These steps constitues the basic building blocks to solve any other problem using HF.

## 🛒 Loading data
In this stage we will load the data from the `datasets` library. We will only load a small subset of the original dataset here in order to reduce the training time, but feel free to run this code on the full dataset on your own time and experiment with it.



```python
from datasets import load_dataset

dataset_train = load_dataset("amazon_polarity", split="train[:1000]")
dataset_test = load_dataset("amazon_polarity", split="test[:200]")
```

    Found cached dataset amazon_polarity (/home/wa_ziqia/.cache/huggingface/datasets/amazon_polarity/amazon_polarity/3.0.0/a27b32b7e7b88eb274a8fa8ba0f654f1fe998a87c22547557317793b5d2772dc)
    Found cached dataset amazon_polarity (/home/wa_ziqia/.cache/huggingface/datasets/amazon_polarity/amazon_polarity/3.0.0/a27b32b7e7b88eb274a8fa8ba0f654f1fe998a87c22547557317793b5d2772dc)



```python
#@title 🔍 Quick look at the data { run: "auto" }
#@markdown Lets have quick look at a few samples as well as the label distributions in our train and test set.
n_samples_to_see = 3 #@param {type: "integer"}
for i in range(n_samples_to_see):
  print("-"*30)
  print("title:", dataset_test[i]["title"])
  print("content:", dataset_test[i]["content"])
  print("label:", dataset_test[i]["label"])
```

    ------------------------------
    title: Great CD
    content: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I'm in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life's hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"
    label: 1
    ------------------------------
    title: One of the best game music soundtracks - for a game I didn't really play
    content: Despite the fact that I have only played a small portion of the game, the music I heard (plus the connection to Chrono Trigger which was great as well) led me to purchase the soundtrack, and it remains one of my favorite albums. There is an incredible mix of fun, epic, and emotional songs. Those sad and beautiful tracks I especially like, as there's not too many of those kinds of songs in my other video game soundtracks. I must admit that one of the songs (Life-A Distant Promise) has brought tears to my eyes on many occasions.My one complaint about this soundtrack is that they use guitar fretting effects in many of the songs, which I find distracting. But even if those weren't included I would still consider the collection worth it.
    label: 1
    ------------------------------
    title: Batteries died within a year ...
    content: I bought this charger in Jul 2003 and it worked OK for a while. The design is nice and convenient. However, after about a year, the batteries would not hold a charge. Might as well just get alkaline disposables, or look elsewhere for a charger that comes with batteries that have better staying power.
    label: 0



```python
def label_stats(ds):
    negative = 0
    positive = 0
    for i in range(ds.num_rows):
        if ds[i]["label"] == 1:
            positive += 1
        else:
            negative += 1
    return positive, negative
```


```python
for i, ds in enumerate([dataset_train, dataset_test]):
    positive, negative = label_stats(ds)
    if i == 0:
        str_indicator = "train"
    else:
        str_indicator = "test"
    print("+-" * 15)
    print("Set:", str_indicator)
    print(f"Positive samples: {positive}\nNegative samples: {negative}")
    print(f"Percentage of overall positive samples: {(positive*100.0)/(positive+negative)}%")
```

    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    Set: train
    Positive samples: 462
    Negative samples: 538
    Percentage of overall positive samples: 46.2%
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    Set: test
    Positive samples: 109
    Negative samples: 91
    Percentage of overall positive samples: 54.5%


## 🧲 Collate
Collate is a function that is called on every batch of data prepared by the [dataloader](https://https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Once we pass our dataset (e.g. `train_set`) to our dataloader, each batch will be a `list` of `dict` items. Therefore, this cannot be directed to the model. We need to perform the followings at this stage:


### 1️⃣ Tokenize the `text`
Tokenize the `text`portion of each sample (i.e. parsing the text to smaller chuncks). Tokenization can happen in many ways, traditionally this was done based the white spaces. With transformer-based models tokenization is performed based on the frequency of occurance of "chunk of text". This frequence can be learnt in many different ways, however the most common one is the [**wordpiece**](https://arxiv.org/pdf/1609.08144v2.pdf) model. 
> The wordpiece model is generated using a data-driven approach to maximize the language-model likelihood
of the training data, given an evolving word definition. Given a training corpus and a number of desired
tokens $D$, the optimization problem is to select $D$ wordpieces such that the resulting corpus is minimal in the
number of wordpieces when segmented according to the chosen wordpiece model.

Under this model:
1. Not all things can be converted to tokens depending on the model. For example, most models have been pretrained without any knowledge of emojis. So their token will be `[UNK]`, which stands for unknown.
2. Some words will be mapped to multiple tokens!
3. Depending on the kind of model, your tokens may or may not respect capitalization!


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```


```python
#@title 🔍 Quick look at tokenization { run: "auto", vertical-output: true }
input_sample = "We are very jubilant to demonstrate to you the 🤗 Transformers library." #@param {type: "string"}
tokenizer.tokenize(input_sample)
```




    ['we',
     'are',
     'very',
     'ju',
     '##bil',
     '##ant',
     'to',
     'demonstrate',
     'to',
     'you',
     'the',
     '[UNK]',
     'transformers',
     'library',
     '.']



### 2️⃣ Encoding
Once we have tokenized the text, we then need to convert these chuncks to numbers so we can feed them to our model. This conversion is basically a look-up in a dictionary **from `str` $\to$ `int`**. The tokenizer object can also perform this work. While it does so it will also add the *special* tokens needed by the model to the encodings. 


```python
#@title 🔍 Quick look at token encoding { run: "auto"}
input_sample = "We are very jubilant to demonstrate to you the 🤗 Transformers library." #@param {type: "string"}
print("--> Token Encodings:\n",tokenizer.encode(input_sample))
print("-."*15)
print("--> Token Encodings Decoded:\n",tokenizer.decode(tokenizer.encode(input_sample)))
```

    --> Token Encodings:
     [101, 2057, 2024, 2200, 18414, 14454, 4630, 2000, 10580, 2000, 2017, 1996, 100, 19081, 3075, 1012, 102]
    -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
    --> Token Encodings Decoded:
     [CLS] we are very jubilant to demonstrate to you the [UNK] transformers library. [SEP]


### 3️⃣ Truncate/Pad samples
Since all the sample in the batch will not have the same sequence length, we would need to truncate the longers (i.e. the ones that exeed a predefined maximum length) and pad the shorter ones so we that we can equal length for all the samples in the batch. Once this is achieved, we would need to convert the result to `torch.Tensor`s and return. These tensors will then be retrieved from the [dataloader](https://https//pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).


```python
from typing import List, Dict, Union
import torch


class Collate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["title"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})
```


```python
#@title 🧑‍🍳 Setting up the collate function { run: "auto" }
tokenizer_name = "distilbert-base-uncased" #@param {type: "string"}
sample_max_length = 64 #@param {type:"slider", min:32, max:512, step:1}
collate = Collate(tokenizer="distilbert-base-uncased", max_len=sample_max_length)
```

## 🤖 Model
Our model needs to classify an entire sequence of text. Once we feed an input sequence of length $k$ to a language model, it will output $k$ vectors. Now the question is which of these vectors or combition of these vectors should we use to classify the sequence?
We will use the first toke, special token `[cls]` for these purposes. *Refer to the [BERT paper](https://arxiv.org/abs/1810.04805) for more information.*

Since we have 2 classes (positive, and negative), this means we would need to make a classifier on top of the vector representations of the `[cls]` token. Our custom model will then look like:


```python
import torch
from transformers import AutoModel
from typing import Optional, Tuple


class ReviewClassifier(torch.nn.Module):
    def __init__(self, backbone: str, backbone_hidden_size: int, nb_classes: int):
        super(ReviewClassifier, self).__init__()
        self.backbone = backbone
        self.backbone_hidden_size = backbone_hidden_size
        self.nb_classes = nb_classes

        self.back_bone = AutoModel.from_pretrained(
            self.backbone,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = torch.nn.Linear(self.backbone_hidden_size, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        back_bone_output = self.back_bone(input_ids, attention_mask=attention_mask)
        hidden_states = back_bone_output[0]
        pooled_output = hidden_states[:, 0]  # getting the [CLS] token

        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits
```


```python
model = ReviewClassifier(backbone="distilbert-base-uncased", backbone_hidden_size=768, nb_classes=2)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertModel: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_projector.bias']
    - This IS expected if you are initializing DistilBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


## 🏓 Training Loop
In this section we will define the training loop to trian our model. Note that these model are sensative wrt the hyperparameters and it usually takes a while to find the right hyperparameters. The default hyperparameters should work fine for our test case.


```python
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

print(f"--> Device selected: {device}")
```

    --> Device selected: cuda



```python
def train_one_epoch(
    model: torch.nn.Module, training_data_loader: DataLoader, optimizer: torch.optim.Optimizer, logging_frequency: int
):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    logging_loss = 0
    for step, batch in enumerate(training_data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        logging_loss += loss.item()

        if (step + 1) % logging_frequency == 0:
            print(f"Training loss @ step {step+1}: {logging_loss/logging_frequency}")
            logging_loss = 0

    return epoch_loss / len(training_data_loader)


def evaluate(model: torch.nn.Module, test_data_loader: DataLoader, nb_classes: int):
    model.eval()
    model.to(device)
    eval_loss = 0
    correct_predictions = {i: 0 for i in range(nb_classes)}
    total_predictions = {i: 0 for i in range(nb_classes)}

    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            eval_loss += loss.item()

            predictions = np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
            for target, prediction in zip(batch["labels"].cpu().numpy(), predictions):
                if target == prediction:
                    correct_predictions[target] += 1
                total_predictions[target] += 1

    accuracy = (100.0 * sum(correct_predictions.values())) / sum(total_predictions.values())
    return accuracy, eval_loss / len(test_data_loader)
```


```python
#@title 🧑‍🍳 Setting hyperparameters for training { run: "auto" }
nb_epoch = 3 #@param {type: "slider", min:1, max:10, step:1}
batch_size = 64 #@param {type: "integer"}
logging_frequency = 5 #@param {type: "integer"}
learning_rate = 1e-5 #@param {type: "number"}

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

# setting up the optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

```


```python
model.to(device)

train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss  = evaluate(model, test_loader, 2)
    print(f"    Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Test: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})
```


    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]


    Training loss @ step 5: 0.6864136815071106
    Training loss @ step 10: 0.6735731720924377
    Training loss @ step 15: 0.6486520171165466
        Epoch: 1 Loss/Test: 0.6080581694841385, Loss/Test: 0.6677345559000969, Acc/Test: 73.5
    Training loss @ step 5: 0.5689218640327454
    Training loss @ step 10: 0.5336737990379333
    Training loss @ step 15: 0.4956661880016327
        Epoch: 2 Loss/Test: 0.3880041316151619, Loss/Test: 0.5235507879406214, Acc/Test: 84.0
    Training loss @ step 5: 0.37964634895324706
    Training loss @ step 10: 0.39118672013282774
    Training loss @ step 15: 0.31929888725280764
        Epoch: 3 Loss/Test: 0.561337485909462, Loss/Test: 0.37485682033002377, Acc/Test: 70.0


# 🗃️ Exercises
It is suggested that you have look over the `tokenizer` class and its functionalities before attempting the exercises.

## 1️⃣ Predict with more context
In the above training we only took advantage of the `title` of each review to predict its polarity.
1. Investigate whether it would be useful to instead use the `content` of each review?
2. Further investigate if it would be usefult to have both the `title` and `content` presented to model during training?


```python
# 1. Update the Collate class to use the "content" 

class Collate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["content"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})

tokenizer_name = "distilbert-base-uncased" #@param {type: "string"}
sample_max_length = 64 #@param {type:"slider", min:32, max:512, step:1}
collate = Collate(tokenizer="distilbert-base-uncased", max_len=sample_max_length)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

model.to(device)

train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss  = evaluate(model, test_loader, 2)
    print(f"    Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Test: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})
```


    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]


    Training loss @ step 5: 0.7047205448150635
    Training loss @ step 10: 0.6284037232398987
    Training loss @ step 15: 0.498206752538681
        Epoch: 1 Loss/Test: 0.5485259592533112, Loss/Test: 0.6075660232454538, Acc/Test: 77.0
    Training loss @ step 5: 0.5838162899017334
    Training loss @ step 10: 0.6462031006813049
    Training loss @ step 15: 0.6171675443649292
        Epoch: 2 Loss/Test: 0.4981459751725197, Loss/Test: 0.6082155182957649, Acc/Test: 75.0
    Training loss @ step 5: 0.42882089614868163
    Training loss @ step 10: 0.39044376015663146
    Training loss @ step 15: 0.4203352272510529
        Epoch: 3 Loss/Test: 0.6746558994054794, Loss/Test: 0.41266988404095173, Acc/Test: 74.5


### Using content presents better and more stable performance during training. Content is more informative than title.


```python
# 2. Update the Collate class to use both the title and content

class Collate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["title"] + " [SEP] " + batch_instance["content"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})

tokenizer_name = "distilbert-base-uncased" #@param {type: "string"}
sample_max_length = 64 #@param {type:"slider", min:32, max:512, step:1}
collate = Collate(tokenizer="distilbert-base-uncased", max_len=sample_max_length)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

model.to(device)

train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss  = evaluate(model, test_loader, 2)
    print(f"    Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Test: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})
```


    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]


    Training loss @ step 5: 0.423386812210083
    Training loss @ step 10: 0.35766669511795046
    Training loss @ step 15: 0.27586591243743896
        Epoch: 1 Loss/Test: 0.331856369972229, Loss/Test: 0.34316360112279654, Acc/Test: 84.0
    Training loss @ step 5: 0.30392126441001893
    Training loss @ step 10: 0.38584625720977783
    Training loss @ step 15: 0.35124048590660095
        Epoch: 2 Loss/Test: 0.30827657133340836, Loss/Test: 0.3360754083842039, Acc/Test: 85.0
    Training loss @ step 5: 0.19805218279361725
    Training loss @ step 10: 0.17479583621025085
    Training loss @ step 15: 0.20260158479213713
        Epoch: 3 Loss/Test: 0.29394229874014854, Loss/Test: 0.19540339522063732, Acc/Test: 87.5


### Training with both title and content shows the best performance compared with the first two cases.

## 2️⃣ Frozen representations
Modify the backbone so that we would only train the classifier layer, and the backbone stays frozen. How does the results compare to the unfrozen version?


```python
classifier_params = [param for name, param in model.named_parameters() if "classifier" in name]

optimizer_grouped_parameters = [
    {
        "params": classifier_params,
        "weight_decay": 0.0,
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

model.to(device)

train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss  = evaluate(model, test_loader, 2)
    print(f"    Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Test: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})

```


    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]


    Training loss @ step 5: 0.1596461296081543
    Training loss @ step 10: 0.09291488490998745
    Training loss @ step 15: 0.15172291845083236
        Epoch: 1 Loss/Test: 0.5576765118166804, Loss/Test: 0.13323528214823455, Acc/Test: 85.5
    Training loss @ step 5: 0.10412054732441903
    Training loss @ step 10: 0.10575225353240966
    Training loss @ step 15: 0.14210815355181694
        Epoch: 2 Loss/Test: 0.5515907891094685, Loss/Test: 0.11911454517394304, Acc/Test: 85.5
    Training loss @ step 5: 0.08342054337263108
    Training loss @ step 10: 0.11610592901706696
    Training loss @ step 15: 0.12816433608531952
        Epoch: 3 Loss/Test: 0.5461094807833433, Loss/Test: 0.11300338478758931, Acc/Test: 85.5


### Frozen backbone results in faster convergence. And the performance is good compared with the first three epochs of fine-tuning the entire model.

## 3️⃣ (Optional) Freeze then unfreeze
It has empirically been shown that freezing the backbone for the first few steps of training and then unfreezing it produces better performing models. Modify the training code to have this option for training. 


```python
def update_optimizer(optimizer, model, backbone_frozen):
    classifier_params = [param for name, param in model.named_parameters() if "classifier" in name]
    if backbone_frozen:
        optimizer_grouped_parameters = [
            {
                "params": classifier_params,
                "weight_decay": 0.0,
            }
        ]
    else:
        backbone_params = [param for name, param in model.named_parameters() if "classifier" not in name]
        optimizer_grouped_parameters = [
            {
                "params": classifier_params + backbone_params,
                "weight_decay": 0.0,
            }
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    return optimizer

initial_frozen_epochs = 1

train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    # Update the optimizer based on whether the backbone should be frozen or not
    backbone_frozen = e < initial_frozen_epochs
    optimizer = update_optimizer(optimizer, model, backbone_frozen)

    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss = evaluate(model, test_loader, 2)
    print(f"    Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Train: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})

```


    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]


    Training loss @ step 5: 0.06775848586112261
    Training loss @ step 10: 0.03402771819382906
    Training loss @ step 15: 0.05575862657278776
        Epoch: 1 Loss/Test: 0.6363000720739365, Loss/Train: 0.05318336054915562, Acc/Test: 85.0
    Training loss @ step 5: 0.043672281038016084
    Training loss @ step 10: 0.025968672893941402
    Training loss @ step 15: 0.016677454486489295
        Epoch: 2 Loss/Test: 0.5121978521347046, Loss/Train: 0.02863416718901135, Acc/Test: 88.5
    Training loss @ step 5: 0.020574381947517394
    Training loss @ step 10: 0.02387645998969674
    Training loss @ step 15: 0.0067906203679740425
        Epoch: 3 Loss/Test: 0.5763185098767281, Loss/Train: 0.0278868559980765, Acc/Test: 86.0


## 4️⃣ (Optional) Build an emotion aware AI
Lets now put everything we learned to the test by building an agent with some emotion detection abilities. Use the [emotion dataset](https://huggingface.co/datasets/emotion) to train an [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)-based model to detect the six basic emotions in our datasets. (anger, fear, joy, love, sadness, and surprise)


```python
from datasets import load_dataset

dataset = load_dataset("emotion")
dataset_train = dataset["train"]
dataset_test = dataset["test"]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

# Update the Collate function to handle the emotion dataset
class EmotionCollate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["text"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})

collate = EmotionCollate(tokenizer="albert-base-v2", max_len=64)
```


    Downloading builder script:   0%|          | 0.00/3.97k [00:00<?, ?B/s]



    Downloading metadata:   0%|          | 0.00/3.28k [00:00<?, ?B/s]



    Downloading readme:   0%|          | 0.00/8.78k [00:00<?, ?B/s]


    No config specified, defaulting to: emotion/split


    Downloading and preparing dataset emotion/split to /home/wa_ziqia/.cache/huggingface/datasets/emotion/split/1.0.0/cca5efe2dfeb58c1d098e0f9eeb200e9927d889b5a03c67097275dfb5fe463bd...



    Downloading data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Downloading data:   0%|          | 0.00/592k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/74.0k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/74.9k [00:00<?, ?B/s]



    Extracting data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Generating train split:   0%|          | 0/16000 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/2000 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/2000 [00:00<?, ? examples/s]


    Dataset emotion downloaded and prepared to /home/wa_ziqia/.cache/huggingface/datasets/emotion/split/1.0.0/cca5efe2dfeb58c1d098e0f9eeb200e9927d889b5a03c67097275dfb5fe463bd. Subsequent calls will reuse this data.



      0%|          | 0/3 [00:00<?, ?it/s]



    Downloading (…)lve/main/config.json:   0%|          | 0.00/684 [00:00<?, ?B/s]



    Downloading (…)ve/main/spiece.model:   0%|          | 0.00/760k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/1.31M [00:00<?, ?B/s]



```python
from transformers import AlbertModel


model = AlbertModel.from_pretrained("albert-base-v2")
class EmotionClassifier(torch.nn.Module):
    def __init__(self, backbone: str, backbone_hidden_size: int, nb_classes: int):
        super(EmotionClassifier, self).__init__()
        self.backbone = backbone
        self.backbone_hidden_size = backbone_hidden_size
        self.nb_classes = nb_classes

        self.back_bone = AlbertModel.from_pretrained(
            self.backbone,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = torch.nn.Linear(self.backbone_hidden_size, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        back_bone_output = self.back_bone(input_ids, attention_mask=attention_mask)
        hidden_states = back_bone_output[0]
        pooled_output = hidden_states[:, 0]  # getting the [CLS] token

        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits

model = EmotionClassifier(backbone="albert-base-v2", backbone_hidden_size=768, nb_classes=6)


train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

model.to(device)
nb_epoch = 5
classifier_params = [param for name, param in model.named_parameters() if "classifier" in name]

optimizer_grouped_parameters = [
    {
        "params": classifier_params,
        "weight_decay": 0.0,
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
train_bar = tqdm(range(nb_epoch), desc="Epoch")
for e in train_bar:
    # Update the optimizer based on whether the backbone should be frozen or not
#     backbone_frozen = e < initial_frozen_epochs
#     optimizer = update_optimizer(optimizer, model, backbone_frozen)
    
    train_loss = train_one_epoch(model, train_loader, optimizer, logging_frequency)
    eval_acc, eval_loss = evaluate(model, test_loader, 6) # There are 6 classes in the emotion dataset
    print(f" Epoch: {e+1} Loss/Test: {eval_loss}, Loss/Train: {train_loss}, Acc/Test: {eval_acc}")
    train_bar.set_postfix({"Loss/Train": train_loss, "Loss/Test": eval_loss, "Acc/Test": eval_acc})

```

    Some weights of the model checkpoint at albert-base-v2 were not used when initializing AlbertModel: ['predictions.bias', 'predictions.dense.weight', 'predictions.LayerNorm.bias', 'predictions.dense.bias', 'predictions.decoder.weight', 'predictions.decoder.bias', 'predictions.LayerNorm.weight']
    - This IS expected if you are initializing AlbertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing AlbertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of the model checkpoint at albert-base-v2 were not used when initializing AlbertModel: ['predictions.bias', 'predictions.dense.weight', 'predictions.LayerNorm.bias', 'predictions.dense.bias', 'predictions.decoder.weight', 'predictions.decoder.bias', 'predictions.LayerNorm.weight']
    - This IS expected if you are initializing AlbertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing AlbertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



    Epoch:   0%|          | 0/5 [00:00<?, ?it/s]


    Training loss @ step 5: 1.7215044498443604
    Training loss @ step 10: 1.6728541612625123
    Training loss @ step 15: 1.708013939857483
    Training loss @ step 20: 1.7007489204406738
    Training loss @ step 25: 1.691190814971924
    Training loss @ step 30: 1.68984375
    Training loss @ step 35: 1.7012216806411744
    Training loss @ step 40: 1.66433744430542
    Training loss @ step 45: 1.6535715818405152
    Training loss @ step 50: 1.6232777118682862
    Training loss @ step 55: 1.6377749919891358
    Training loss @ step 60: 1.6054049253463745
    Training loss @ step 65: 1.5997618675231933
    Training loss @ step 70: 1.672364354133606
    Training loss @ step 75: 1.6390195608139038
    Training loss @ step 80: 1.635919737815857
    Training loss @ step 85: 1.564455223083496
    Training loss @ step 90: 1.6188425064086913
    Training loss @ step 95: 1.5686997652053833
    Training loss @ step 100: 1.6247116804122925
    Training loss @ step 105: 1.6575784921646117
    Training loss @ step 110: 1.5630854606628417
    Training loss @ step 115: 1.5854912996292114
    Training loss @ step 120: 1.6166377544403077
    Training loss @ step 125: 1.568805742263794
    Training loss @ step 130: 1.605193567276001
    Training loss @ step 135: 1.5484853506088256
    Training loss @ step 140: 1.545970368385315
    Training loss @ step 145: 1.5776808738708497
    Training loss @ step 150: 1.5762733697891236
    Training loss @ step 155: 1.6047330379486084
    Training loss @ step 160: 1.5539275646209716
    Training loss @ step 165: 1.5842020988464356
    Training loss @ step 170: 1.581087017059326
    Training loss @ step 175: 1.5657620191574098
    Training loss @ step 180: 1.536041522026062
    Training loss @ step 185: 1.5870162487030028
    Training loss @ step 190: 1.5717156171798705
    Training loss @ step 195: 1.5976205587387085
    Training loss @ step 200: 1.6329406023025512
    Training loss @ step 205: 1.5853654146194458
    Training loss @ step 210: 1.5541209697723388
    Training loss @ step 215: 1.584219717979431
    Training loss @ step 220: 1.5551318883895875
    Training loss @ step 225: 1.6683475971221924
    Training loss @ step 230: 1.5444633960723877
    Training loss @ step 235: 1.5633499383926392
    Training loss @ step 240: 1.600359272956848
    Training loss @ step 245: 1.5286327600479126
    Training loss @ step 250: 1.4412809371948243
     Epoch: 1 Loss/Test: 1.5523383021354675, Loss/Train: 1.6055807905197144, Acc/Test: 40.5
    Training loss @ step 5: 1.6455963373184204
    Training loss @ step 10: 1.5976862907409668
    Training loss @ step 15: 1.5037878036499024
    Training loss @ step 20: 1.5463937520980835
    Training loss @ step 25: 1.6281051158905029
    Training loss @ step 30: 1.5289281606674194
    Training loss @ step 35: 1.5675542116165162
    Training loss @ step 40: 1.5937638759613038
    Training loss @ step 45: 1.5666862726211548
    Training loss @ step 50: 1.5548333883285523
    Training loss @ step 55: 1.550923800468445
    Training loss @ step 60: 1.6352694988250733
    Training loss @ step 65: 1.5746037006378173
    Training loss @ step 70: 1.5564049243927003
    Training loss @ step 75: 1.550331997871399
    Training loss @ step 80: 1.5547084093093873
    Training loss @ step 85: 1.5025143146514892
    Training loss @ step 90: 1.583439540863037
    Training loss @ step 95: 1.572393274307251
    Training loss @ step 100: 1.6065670728683472
    Training loss @ step 105: 1.5577210187911987
    Training loss @ step 110: 1.5621851205825805
    Training loss @ step 115: 1.5057318210601807
    Training loss @ step 120: 1.5937745571136475
    Training loss @ step 125: 1.5180290937423706
    Training loss @ step 130: 1.6300873517990113
    Training loss @ step 135: 1.5919705629348755
    Training loss @ step 140: 1.5809332132339478
    Training loss @ step 145: 1.565569519996643
    Training loss @ step 150: 1.6122603178024293
    Training loss @ step 155: 1.5370992422103882
    Training loss @ step 160: 1.4713426589965821
    Training loss @ step 165: 1.512699055671692
    Training loss @ step 170: 1.5612898349761963
    Training loss @ step 175: 1.523965549468994
    Training loss @ step 180: 1.5182973861694335
    Training loss @ step 185: 1.5658039808273316
    Training loss @ step 190: 1.5062865257263183
    Training loss @ step 195: 1.5207314491271973
    Training loss @ step 200: 1.5480863332748414
    Training loss @ step 205: 1.5448449373245239
    Training loss @ step 210: 1.5327778100967406
    Training loss @ step 215: 1.4936366558074952
    Training loss @ step 220: 1.5325276851654053
    Training loss @ step 225: 1.4675686359405518
    Training loss @ step 230: 1.470639443397522
    Training loss @ step 235: 1.5713453769683838
    Training loss @ step 240: 1.5744708061218262
    Training loss @ step 245: 1.4658097505569458
    Training loss @ step 250: 1.4848499059677125
     Epoch: 2 Loss/Test: 1.514723762869835, Loss/Train: 1.5508565468788147, Acc/Test: 38.95
    Training loss @ step 5: 1.5025535345077514
    Training loss @ step 10: 1.5384912252426148
    Training loss @ step 15: 1.5607666969299316
    Training loss @ step 20: 1.5494430541992188
    Training loss @ step 25: 1.5014452934265137
    Training loss @ step 30: 1.49620463848114
    Training loss @ step 35: 1.495279598236084
    Training loss @ step 40: 1.5526311159133912
    Training loss @ step 45: 1.5129241466522216
    Training loss @ step 50: 1.487498927116394
    Training loss @ step 55: 1.5020352602005005
    Training loss @ step 60: 1.5494057178497314
    Training loss @ step 65: 1.5474505424499512
    Training loss @ step 70: 1.4782512187957764
    Training loss @ step 75: 1.5287511825561524
    Training loss @ step 80: 1.5503345489501954
    Training loss @ step 85: 1.4978762865066528
    Training loss @ step 90: 1.5505949974060058
    Training loss @ step 95: 1.5740411281585693
    Training loss @ step 100: 1.5102305173873902
    Training loss @ step 105: 1.469995641708374
    Training loss @ step 110: 1.5425589084625244
    Training loss @ step 115: 1.4792234420776367
    Training loss @ step 120: 1.500689959526062
    Training loss @ step 125: 1.5141486406326294
    Training loss @ step 130: 1.432607316970825
    Training loss @ step 135: 1.6038545846939087
    Training loss @ step 140: 1.4770460367202758
    Training loss @ step 145: 1.4397047996520995
    Training loss @ step 150: 1.4758578538894653
    Training loss @ step 155: 1.5662209749221803
    Training loss @ step 160: 1.516641092300415
    Training loss @ step 165: 1.5590457916259766
    Training loss @ step 170: 1.5390382766723634
    Training loss @ step 175: 1.4739015579223633
    Training loss @ step 180: 1.5877967596054077
    Training loss @ step 185: 1.5727082014083862
    Training loss @ step 190: 1.5541956663131713
    Training loss @ step 195: 1.5434831619262694
    Training loss @ step 200: 1.5289115905761719
    Training loss @ step 205: 1.4740438222885133
    Training loss @ step 210: 1.4742789030075074
    Training loss @ step 215: 1.507567572593689
    Training loss @ step 220: 1.6009572505950929
    Training loss @ step 225: 1.4586663007736207
    Training loss @ step 230: 1.5786720991134644
    Training loss @ step 235: 1.5728407144546508
    Training loss @ step 240: 1.4661128759384154
    Training loss @ step 245: 1.4968255519866944
    Training loss @ step 250: 1.5157749652862549
     Epoch: 3 Loss/Test: 1.4881233870983124, Loss/Train: 1.520191598892212, Acc/Test: 39.65
    Training loss @ step 5: 1.5086355924606323
    Training loss @ step 10: 1.5153269529342652
    Training loss @ step 15: 1.4878267526626587
    Training loss @ step 20: 1.4745744466781616
    Training loss @ step 25: 1.501077127456665
    Training loss @ step 30: 1.5202488422393798
    Training loss @ step 35: 1.4545573234558105
    Training loss @ step 40: 1.4615532398223876
    Training loss @ step 45: 1.5138927698135376
    Training loss @ step 50: 1.5435882329940795
    Training loss @ step 55: 1.5030220746994019
    Training loss @ step 60: 1.560760259628296
    Training loss @ step 65: 1.5162882089614869
    Training loss @ step 70: 1.4862077474594115
    Training loss @ step 75: 1.4847218990325928
    Training loss @ step 80: 1.5030912160873413
    Training loss @ step 85: 1.4812950372695923
    Training loss @ step 90: 1.5721893310546875
    Training loss @ step 95: 1.4571950674057006
    Training loss @ step 100: 1.436559009552002
    Training loss @ step 105: 1.468324565887451
    Training loss @ step 110: 1.5171959400177002
    Training loss @ step 115: 1.442266583442688
    Training loss @ step 120: 1.4306885719299316
    Training loss @ step 125: 1.5396093368530273
    Training loss @ step 130: 1.526958131790161
    Training loss @ step 135: 1.5295637607574464
    Training loss @ step 140: 1.460082721710205
    Training loss @ step 145: 1.566672444343567
    Training loss @ step 150: 1.508916974067688
    Training loss @ step 155: 1.447930884361267
    Training loss @ step 160: 1.5458407402038574
    Training loss @ step 165: 1.5125418186187745
    Training loss @ step 170: 1.4524744272232055
    Training loss @ step 175: 1.532536220550537
    Training loss @ step 180: 1.4891494035720825
    Training loss @ step 185: 1.4802096605300903
    Training loss @ step 190: 1.4525571584701538
    Training loss @ step 195: 1.5307323455810546
    Training loss @ step 200: 1.5394468784332276
    Training loss @ step 205: 1.4495439529418945
    Training loss @ step 210: 1.4482901573181153
    Training loss @ step 215: 1.5393302679061889
    Training loss @ step 220: 1.4884023427963258
    Training loss @ step 225: 1.5147291660308837
    Training loss @ step 230: 1.4018651247024536
    Training loss @ step 235: 1.4867803812026978
    Training loss @ step 240: 1.4492143392562866
    Training loss @ step 245: 1.4761451482772827
    Training loss @ step 250: 1.4701338291168213
     Epoch: 4 Loss/Test: 1.4892820194363594, Loss/Train: 1.4936148881912232, Acc/Test: 37.75
    Training loss @ step 5: 1.4425822257995606
    Training loss @ step 10: 1.4574259281158448
    Training loss @ step 15: 1.4636321544647217
    Training loss @ step 20: 1.4392324924468993
    Training loss @ step 25: 1.4606940507888795
    Training loss @ step 30: 1.4845847606658935
    Training loss @ step 35: 1.4754969120025634
    Training loss @ step 40: 1.501175832748413
    Training loss @ step 45: 1.486658477783203
    Training loss @ step 50: 1.497943925857544
    Training loss @ step 55: 1.4815522193908692
    Training loss @ step 60: 1.5093823194503784
    Training loss @ step 65: 1.4148756265640259
    Training loss @ step 70: 1.4436028242111205
    Training loss @ step 75: 1.4206866025924683
    Training loss @ step 80: 1.4867148637771606
    Training loss @ step 85: 1.5396081924438476
    Training loss @ step 90: 1.5487081289291382
    Training loss @ step 95: 1.4921606063842774
    Training loss @ step 100: 1.456519079208374
    Training loss @ step 105: 1.5167119979858399
    Training loss @ step 110: 1.504753541946411
    Training loss @ step 115: 1.42199604511261
    Training loss @ step 120: 1.5402597427368163
    Training loss @ step 125: 1.5350120306015014
    Training loss @ step 130: 1.5210614919662475
    Training loss @ step 135: 1.521850299835205
    Training loss @ step 140: 1.4877012014389037
    Training loss @ step 145: 1.5242607116699218
    Training loss @ step 150: 1.5570303201675415
    Training loss @ step 155: 1.5494842052459716
    Training loss @ step 160: 1.52783043384552
    Training loss @ step 165: 1.5176272630691527
    Training loss @ step 170: 1.4935159921646117
    Training loss @ step 175: 1.5014246940612792
    Training loss @ step 180: 1.4504627227783202
    Training loss @ step 185: 1.4428431749343873
    Training loss @ step 190: 1.4891221284866334
    Training loss @ step 195: 1.5170034885406494
    Training loss @ step 200: 1.43388831615448
    Training loss @ step 205: 1.4969878196716309
    Training loss @ step 210: 1.4526869058609009
    Training loss @ step 215: 1.5029356002807617
    Training loss @ step 220: 1.4714970350265504
    Training loss @ step 225: 1.4887535095214843
    Training loss @ step 230: 1.4887449026107789
    Training loss @ step 235: 1.5000856637954711
    Training loss @ step 240: 1.4765613317489623
    Training loss @ step 245: 1.4859441280364991
    Training loss @ step 250: 1.5720912218093872
     Epoch: 5 Loss/Test: 1.495351005345583, Loss/Train: 1.4898673028945923, Acc/Test: 35.95



```python
def predict_emotion(model, tokenizer, text: str):
    emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]

    tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        predicted_emotion = emotions[np.argmax(probabilities)]

    return predicted_emotion, probabilities


text = "I am fear."
predicted_emotion, probabilities = predict_emotion(model, tokenizer, text)
print(f"Predicted emotion: {predicted_emotion}")
```

    Predicted emotion: fear



```python

```
