import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorWithPadding, AutoModelForSequenceClassification)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    # The experiment variables
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    # The rest of arguments
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)

    # Additional for analyzing use
    parser.add_argument("--save_as_csv", action="store_true")

    return parser.parse_args()

def get_train_val_test_datasets(tokenize_function: callable, args: argparse):
    dataset = load_dataset("glue", "mrpc")
    encoded = dataset.map(tokenize_function, batched=True)
    train_dataset = encoded["train"].select(range(args.max_train_samples)) if args.max_train_samples != -1 else encoded["train"]
    eval_dataset = encoded["validation"].select(range(args.max_eval_samples)) if args.max_eval_samples != -1 else encoded["validation"]
    test_dataset = encoded["test"].select(range(args.max_predict_samples)) if args.max_predict_samples != -1 else encoded["test"]

    return train_dataset, eval_dataset, test_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Created tokneizer from pretrained: \"bert-base-uncased\"\n")

def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,  # We'll use dynamic padding
    )

# Choosing device (this code was built on a MacBook)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)
print(f"The model \"bert-base-uncased\" has been loaded and sent to the device {device}.\n")

if __name__ == "__main__":
    # Get args
    args = parse_args()
    
    # Connect wandb
    keys_file = 'dev_keys.json' if os.path.isfile('dev_keys.json') else 'keys.json'
    with open(keys_file, 'r') as f:
        keys = json.load(f)

    wandb.login(key=keys['wandb'])
    run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
    wandb.init(
        project="anlp-ex1",
        name=run_name,
        config={
            "epochs": 2,
            "learning_rate": 0.01,
            "batch_size": 32
            }
    )

    # Read datasets
    train_dataset, eval_dataset, test_dataset = get_train_val_test_datasets(tokenize_function, args)

    # Define args
    training_args = TrainingArguments(
        learning_rate=args.lr,                          # --lr
        per_device_train_batch_size=args.batch_size,    # --batch_size
        per_device_eval_batch_size=args.batch_size,     # ""
        num_train_epochs=args.num_train_epochs,         # --num_train_epochs
        
        output_dir="./results",
        overwrite_output_dir=True,
        
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="no",
        report_to="wandb",
        do_train=args.do_train,
        do_predict=args.do_predict,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    model_path = args.model_path if args.model_path else "final_model"

    # Train
    if args.do_train:
        training_result = trainer.train()
        trainer.save_model(model_path)
        eval_result = trainer.evaluate()

        final_val_acc = eval_result["eval_accuracy"]
        with open('res.txt', 'a') as f:
            new_line = ', '.join([
                f'epoch_num: {args.num_train_epochs}',
                f'lr: {args.lr}',
                f'batch_size: {args.batch_size}',
                f'eval_acc: {final_val_acc:.4f}\n'
            ])
            f.write(new_line)
        

        # Save results of validation as csv
        if args.save_as_csv:
            # Run predictions - trainer.predict() calls model.eval() internally
            predictions = trainer.predict(eval_dataset)
            predicted_labels = np.argmax(predictions.predictions, axis=1)
            
            pred_output_data = []
            for s1, s2, label, pred_label in zip(eval_dataset['sentence1'], eval_dataset['sentence2'],
                                     eval_dataset['label'], predicted_labels):
                pred_output_data.append([s1, s2, label, pred_label])
            df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label', 'predicted_label'], data=pred_output_data)
            df.to_csv(f'saved_pred_eval_{run_name}.csv', index=False)

    # Test
    if args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        trainer.model = model  # update trainer with reloaded model

        # Run predictions - trainer.predict() calls model.eval() internally
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Create predictions text
        pred_output_lines = []
        for s1, s2, pred_label in zip(test_dataset['sentence1'], test_dataset['sentence2'], predicted_labels):
            pred_output_lines.append(f"<{s1}>###<{s2}>###<{pred_label}>")
        
        pred_output_lines = '\n'.join(pred_output_lines)

        # Save predictions to file
        with open("predictions.txt", "w") as f:
            f.write(pred_output_lines)

"""
An example:
python ex1.py --lr 2e-5 --num_train_epochs 3 --batch_size 16 --do_train --do_predict --save_as_csv
"""