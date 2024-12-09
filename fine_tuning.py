from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

import pandas as pd
import numpy as np

# Warnings
import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.get_device_capability())

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




