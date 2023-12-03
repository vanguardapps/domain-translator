from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import json
import evaluate
from functools import partial


def load_primary_components(max_sequence_length):
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "test": "data/test.csv",
            "validate": "data/validate.csv",
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=max_sequence_length,
    )
    return dataset, tokenizer, model, data_collator


def tokenize_function_generic(
    tokenizer, max_sequence_length, input_property, label_property, example
):
    return tokenizer(
        example[input_property],
        example[label_property],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
    )


def main():
    # Primary configuration
    max_sequence_length = 128  # Must set in model, then Seq2SeqTrainingArguments will default its generation_max_length parameter to model.max_length

    dataset, tokenizer, model, data_collator = load_primary_components(
        max_sequence_length
    )

    tokenize_function = partial(
        tokenize_function_generic, tokenizer, max_sequence_length, "header1", "labels"
    )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Trainer configuration
    train_args = Seq2SeqTrainingArguments(
        use_cpu=True,  # Set to False to automatically enable CUDA / mps device
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        output_dir="output",
        overwrite_output_dir=True,
        push_to_hub=False,
        learning_rate=5e-5,
        logging_strategy="epoch",
        optim="adamw_torch",
        warmup_steps=200,
        predict_with_generate=False,  ## ??? Set to True??
        adam_beta1=0.9,
        adam_beta2=0.999,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,  # Set to True to improve memory utilization (though will slow training by 20%)
        torch_compile=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validate"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
