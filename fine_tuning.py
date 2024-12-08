import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)
print(torch.cuda.is_available())

data = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
data.columns = data.columns.str.replace('Column1.', '', regex=False)

data = data.iloc[:,:-5]
data = data.drop('ScoutingReportTemplateId', axis=1)
data = data.drop('ScoutingReportTemplate', axis=1)
data = data.drop('EventEndDate', axis=1)
data = data.drop('FilePartition', axis=1)
data = data.drop('Age', axis=1)
data = data.drop('ScoutingReportId', axis=1)
data = data.drop('ChangedAt',axis=1)

#filter players with less than 5 matches
minimum_match_amount = 5
id_counts = data['PlayerId'].value_counts()
data = data[data['PlayerId'].isin(id_counts[id_counts >= minimum_match_amount].index)]

max_seq_length = 5020
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
print(model.print_trainable_parameters())



