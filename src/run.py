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


def load_primary_components():
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
    return dataset, tokenizer, model


def main():
    # Primary configuration
    max_sequence_length = 128  # Must set in model, then Seq2SeqTrainingArguments will default its generation_max_length parameter to model.max_length

    dataset, tokenizer, model = load_primary_components()

    print(tokenizer.model_max_length)

    # TODO: Code to tokenize the dataset here, store in tokenized_dataset (use dataset.map, see hf docs)

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
        predict_with_generate=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,  # Set to True to improve memory utilization (though will slow training by 20%)
        torch_compile=True,
    )

    # TODO: Define data_collator from DataCollatorForSeq2Seq (will have to look this up)

    # TODO: uncomment when above TODOs are done
    # trainer = Seq2SeqTrainer(
    #     model,
    #     train_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["validation"],
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    # )

    # Just an example of using the tokenizer
    # result = tokenizer(
    #     batch_sentences, truncation=True, max_length=max_sequence_length, padding="max_length"
    # )
    # print(dataset)


if __name__ == "__main__":
    main()
