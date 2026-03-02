from datetime import datetime
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current time: {current_time_str}")
from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import librosa
from datasets import load_dataset, Audio
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.utils.class_weight import compute_class_weight

print("Check if GPU available:")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

model_id = "facebook/mms-300m"

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True,
)

dataset = load_dataset("badrex/nnti-dataset-full")

train_ds = dataset["train"].shuffle(seed=42)
valid_ds = dataset["validation"].shuffle(seed=42)

# Resample to 16kHz
train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

def is_long_enough(example):
    return len(example["audio_filepath"]["array"]) / 16000 >= 2

train_ds = train_ds.filter(is_long_enough)
valid_ds = valid_ds.filter(is_long_enough)

input_features_key = "input_values"
max_duration = 10

LABELS = sorted(train_ds.unique("language"))
str_to_int = {s: i for i, s in enumerate(LABELS)}
int_to_str = {i: s for s, i in str_to_int.items()}
num_labels = len(LABELS)

print("Number of labels:", num_labels)
def augment_audio(audio, sr=16000):
    choice = np.random.rand()
    if choice < 0.4:
        rate = np.random.choice([0.9, 0.95, 1.05, 1.1])
        audio = librosa.effects.time_stretch(audio, rate=rate)
    elif choice < 0.7:
        steps = np.random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    else:
        audio = audio + np.random.randn(len(audio)) * 0.002
    return audio

def preprocess_function(examples):
    audio_arrays = []

    for x in examples["audio_filepath"]:
        audio = x["array"]
        audio = augment_audio(audio)   
        audio_arrays.append(audio)

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]

    return inputs
def preprocess_function_validation(examples):
    audio_arrays = [x["array"] for x in examples["audio_filepath"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    return inputs
train_ds_encoded = train_ds.map(
    preprocess_function,
    remove_columns=train_ds.column_names,
    batched=True,
    batch_size=32,
)

valid_ds_encoded = valid_ds.map(
    preprocess_function_validation,
    remove_columns=valid_ds.column_names,
    batched=True,
    batch_size=32,
)

config = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id = str_to_int
config.id2label = int_to_str

config.hidden_dropout = 0.05
config.attention_dropout = 0.05
config.activation_dropout = 0.1
config.feat_proj_dropout = 0.1

slid_model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config,
)

slid_model.freeze_feature_encoder()
for name, param in slid_model.named_parameters():
    if "encoder.layers.10" in name or "encoder.layers.11" in name:
        param.requires_grad = True
labels_list = train_ds["language"]
numeric_labels = [str_to_int[l] for l in labels_list]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(numeric_labels),
    y=numeric_labels,
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")

class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]):
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }

        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [f["label"] for f in features],
            dtype=torch.long,
        )

        return batch

data_collator = AudioDataCollator(feature_extractor)

batch_size = 4
gradient_accumulation_steps = 8
num_train_epochs = 10
lr = 3e-5


training_args = TrainingArguments(
    output_dir="./kaggle_run",
    report_to="none",
    logging_steps=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    learning_rate=lr,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    label_smoothing_factor=0.1,
    fp16=True,
    push_to_hub=False,
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids,
    )

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=slid_model,
    args=training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training starting...")
trainer.train()

print("Final evaluation...")
trainer.evaluate()

save_dir = "./indic-SLID/inprogress"
slid_model.save_pretrained(save_dir)