import argparse
import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./glue-mrpc-results"


def parse_args():
    """
    parses the args that were required in the ex
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_predict_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--model_path', type=str)

    return parser.parse_args()


def preprocess_function(examples, tokenizer):
    """
    runs tokenizer on data
    """
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


def main():
    args = parse_args()

    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    if not args.do_train and not args.do_predict:
        print("Error:  must choose one of [do_train, do_predict]")

    if args.do_train:
        if not args.num_train_epochs or not args.batch_size or not args.lr:
            print("Error: --num_train_epocs, --batch_size and --lr are required if do_train is set to True.")

    if args.do_predict:
        if not args.model_path:
            print("Error: --model_path is required if do_train is set to True.")

    # if I had more time these were all done by a single function :)
    if args.max_train_samples != -1:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        tokenized_datasets["test"] = tokenized_datasets["test"].select(range(args.max_predict_samples))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path if args.do_predict else MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,  
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    if args.do_train:
        trainer.train()
        model.save_pretrained("trained_model")
        tokenizer.save_pretrained("trained_model")

    if args.do_predict:
        test_dataset = tokenized_datasets["test"]
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        with open("predictions.txt", "w") as f:
            for example, pred in zip(raw_datasets["test"], preds):
                line = f"{example['sentence1']}###{example['sentence2']}###{pred}\n"
                f.write(line)


if __name__ == "__main__":
    main()
