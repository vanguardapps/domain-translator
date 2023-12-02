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


def main():
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "test": "data/test.csv",
            "validate": "data/validate.csv",
        },
    )

    


if __name__ == "__main__":
    main()
