"""
ex1.py — Fine-tune bert-base-uncased on MRPC (paraphrase detection).

Training example:
    python ex1.py --max_train_samples -1 --max_eval_samples -1 \
                  --num_train_epochs 3 --lr 2e-5 --batch_size 16 --do_train

Prediction example (after training):
    python ex1.py --max_predict_samples -1 --do_predict \
                  --model_path ./results/epoch_num_3_lr_2e-05_batch_size_16/final_model

W&B authentication:
    Set the WANDB_API_KEY environment variable before running:
        export WANDB_API_KEY="your_key_here"
"""

import os
import argparse
import numpy as np


# ── 1. Arguments ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC.")

    # Sample limits
    parser.add_argument("--max_train_samples",   type=int, default=-1,
                        help="Number of training samples (-1 = all).")
    parser.add_argument("--max_eval_samples",    type=int, default=-1,
                        help="Number of validation samples (-1 = all).")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of test samples for prediction (-1 = all).")

    # Hyperparameters
    parser.add_argument("--num_train_epochs", type=int,   default=3)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--batch_size",       type=int,   default=16)

    # Run modes
    parser.add_argument("--do_train",   action="store_true", help="Run fine-tuning.")
    parser.add_argument("--do_predict", action="store_true", help="Run prediction.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Saved model checkpoint path (required for --do_predict).")

    return parser.parse_args()


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2   # 0 = not paraphrase, 1 = paraphrase


# ── 2. Load Dataset ───────────────────────────────────────────────────────────

def load_data(args):
    """Load MRPC from GLUE and optionally slice each split."""
    from datasets import load_dataset

    raw_datasets = load_dataset("glue", "mrpc")

    # Select the first n samples when a limit is given (spec requirement)
    if args.max_train_samples != -1:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_train_samples, len(raw_datasets["train"])))
        )
    if args.max_eval_samples != -1:
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(args.max_eval_samples, len(raw_datasets["validation"])))
        )
    if args.max_predict_samples != -1:
        raw_datasets["test"] = raw_datasets["test"].select(
            range(min(args.max_predict_samples, len(raw_datasets["test"])))
        )

    return raw_datasets


# ── 3. Load Model & Tokenizer ─────────────────────────────────────────────────

def load_model_and_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Tokenizer converts raw text to token IDs understood by BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # AutoModelForSequenceClassification adds a 2-class linear head on top of
    # BERT's [CLS] token output. The classifier weights are not in the pretrained
    # checkpoint (they are MISSING/randomly initialised) and are learned during
    # fine-tuning. The MLM head weights show as UNEXPECTED and are safely ignored.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=NUM_LABELS,
    )

    return model, tokenizer


# ── 4. Tokenize Dataset ───────────────────────────────────────────────────────

def tokenize_dataset(raw_datasets, tokenizer):
    from transformers import DataCollatorWithPadding

    def tokenize_fn(examples):
        # Feed both sentences together so BERT sees [CLS] s1 [SEP] s2 [SEP].
        # truncation=True clips to BERT's 512-token limit.
        # Padding is NOT done here — DataCollatorWithPadding pads each batch
        # only to its longest sequence at collation time (dynamic padding),
        # which is more efficient than padding everything to the global maximum.
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
        )

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

    # Dynamic padding collator: pads each mini-batch to its own longest sequence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, data_collator


# ── 5. Metrics ────────────────────────────────────────────────────────────────

def make_compute_metrics():
    import evaluate

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        # eval_pred is (logits, labels) — both numpy arrays.
        # argmax over the 2 class logits gives the predicted label (0 or 1).
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


# ── 6 & 7. Train + Evaluate ───────────────────────────────────────────────────

def run_training(args):
    import wandb
    from transformers import TrainingArguments, Trainer

    # wandb.login() picks up WANDB_API_KEY from the environment automatically.
    # Set it before running: export WANDB_API_KEY=your_key_here
    wandb.login()

    # Build a unique run name so each W&B run and output directory is labelled
    # by its hyperparameters — no manual renaming needed when comparing runs.
    run_name   = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
    output_dir = os.path.join("results", run_name)

    # Steps 2-5
    raw_datasets                      = load_data(args)
    model, tokenizer                  = load_model_and_tokenizer(MODEL_NAME)
    tokenized_datasets, data_collator = tokenize_dataset(raw_datasets, tokenizer)
    compute_metrics                   = make_compute_metrics()

    # ── 6. Train ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Evaluate on the validation split at the end of every epoch
        eval_strategy="epoch",

        # Do not save per-epoch checkpoints — we only need the final model and
        # intermediate checkpoints waste disk space (see spec note 2.3)
        save_strategy="no",

        # Hyperparameters passed in via CLI
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,   # L2 regularisation on all non-bias weights

        # Log training loss at every optimiser step → full dense loss curve in W&B
        logging_strategy="steps",
        logging_steps=1,

        run_name=run_name,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    val_acc = eval_results["eval_accuracy"]

    print(f"\n{'='*50}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"{'='*50}\n")

    # Append one line to res.txt — matches the format in the Moodle template
    res_line = (
        f"epoch_num: {args.num_train_epochs}, "
        f"lr: {args.lr}, "
        f"batch_size: {args.batch_size}, "
        f"eval_acc: {val_acc:.4f}\n"
    )
    with open("res.txt", "a") as f:
        f.write(res_line)
    print(f"Appended to res.txt: {res_line.strip()}")

    # Save the final model so it can be loaded for --do_predict later.
    # We save manually here rather than via save_strategy to keep only one copy.
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\nModel saved to: {final_model_dir}")
    print(f"Predict: python ex1.py --do_predict --model_path {final_model_dir}")

    wandb.finish()   # close the W&B run cleanly


# ── Predict ───────────────────────────────────────────────────────────────────

def run_prediction(args):
    from transformers import Trainer, DataCollatorWithPadding

    if args.model_path is None:
        raise ValueError("--model_path is required when using --do_predict.")

    # Steps 2-4
    raw_datasets                      = load_data(args)
    model, tokenizer                  = load_model_and_tokenizer(args.model_path)
    tokenized_datasets, data_collator = tokenize_dataset(raw_datasets, tokenizer)

    # Switch to inference mode: disables Dropout so predictions are deterministic.
    # Layers like Dropout and BatchNorm behave differently during training vs
    # inference — model.eval() signals that we are in inference mode.
    model.eval()

    # Reuse Trainer just for its .predict() convenience; no training happens here.
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )

    predictions_output = trainer.predict(tokenized_datasets["test"])
    # predictions_output.predictions shape: (n_samples, 2) — one logit per class
    pred_labels = np.argmax(predictions_output.predictions, axis=-1)

    # Write predictions.txt — format required by the spec: sentence1###sentence2###label
    with open("predictions.txt", "w") as f:
        for i, label in enumerate(pred_labels):
            s1 = raw_datasets["test"][i]["sentence1"]
            s2 = raw_datasets["test"][i]["sentence2"]
            f.write(f"{s1}###{s2}###{label}\n")

    print(f"predictions.txt written ({len(pred_labels)} samples).")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.do_train and not args.do_predict:
        raise ValueError("Specify at least one of --do_train or --do_predict.")

    if args.do_train:
        run_training(args)

    if args.do_predict:
        run_prediction(args)


if __name__ == "__main__":
    main()
