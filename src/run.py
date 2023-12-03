from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
from functools import partial
from utils import relative_path
import torch


def get_split_dataset(dataset_filepath):
    dataset_all = load_dataset("csv", data_files=relative_path(dataset_filepath))
    train_testvalidate = dataset_all["train"].train_test_split(
        test_size=0.3, shuffle=True
    )
    test_validate = train_testvalidate["test"].train_test_split(
        test_size=0.5, shuffle=True
    )
    return DatasetDict(
        {
            "train": train_testvalidate["train"],
            "test": test_validate["train"],
            "validate": test_validate["test"],
        }
    )


def compute_metrics_generic(tokenizer, metrics_list, eval_preds):
    metric = evaluate.load(*metrics_list)
    predictions, references = eval_preds
    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = tokenizer.batch_decode(
        references, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = [[reference] for reference in references]
    return metric.compute(predictions=predictions, references=references)


def load_primary_components(
    model_name, max_sequence_length, dataset_filepath, metrics_list
):
    dataset = get_split_dataset(dataset_filepath)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=max_sequence_length,
    )
    compute_metrics = partial(compute_metrics_generic, tokenizer, metrics_list)
    return dataset, tokenizer, model, data_collator, compute_metrics


def tokenize_function_generic(
    tokenizer, max_sequence_length, input_property, labels_property, batch
):
    input_feature = tokenizer(
        batch[input_property],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
    )
    labels = tokenizer(
        batch[labels_property],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
    )
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": labels["input_ids"],
    }


def main():
    max_sequence_length = 128

    dataset, tokenizer, model, data_collator, compute_metrics = load_primary_components(
        model_name="google/mt5-base",
        max_sequence_length=max_sequence_length,
        dataset_filepath="data/english_to_spanish.csv",
        metrics_list=["bleu"],
    )

    tokenize_function = partial(
        tokenize_function_generic, tokenizer, max_sequence_length, "english", "spanish"
    )

    # model = model.to(device)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=2)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     tokenized_dataset["train"].set_format("torch", device=device)
    #     tokenized_dataset["test"].set_format("torch", device=device)
    #     tokenized_dataset["validate"].set_format("torch", device=device)

    train_args = Seq2SeqTrainingArguments(
        use_cpu=False,  # Set to False to automatically enable CUDA / mps device
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        output_dir="model_output",
        overwrite_output_dir=True,
        push_to_hub=False,
        learning_rate=5e-5,
        logging_strategy="epoch",
        optim="adamw_torch",
        # warmup_steps=200,
        predict_with_generate=True,  ## ??? Set to True??
        adam_beta1=0.9,
        adam_beta2=0.999,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,  # Set to True to improve memory utilization (though will slow training by 20%)
        torch_compile=False,
    )

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
