### Three ways to convert HuggingFace dataset to Pandas dataframe:

1.  import pandas as pd  # v1.3.5
    from datasets import load_dataset # v2.11.0

2.  df = load_dataset('squad_v2')  # Loading the SQuAD dataset from huggingface.
    pandas_data = pd.DataFrame(df['train'])

3.  df_pandas = train_data_s1.to_pandas()


### Adding special new tokens to the tokenizer

    LANG_TOKEN_MAPPING = {"identify language": ""}
    special_tokens_dict = {
        "additional_special_tokens": list(LANG_TOKEN_MAPPING.values())
    }
    tokenizer.add_special_tokens(special_tokens_dict)


### Customizing a Trainer with Novel Loss Function
    from torch import nn
    from transformers import Trainer


    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

### Further Trainer Customizations
    The Trainer contains the basic training loop which supports the above features. To inject custom behavior you can subclass them and override the following methods:

    get_train_dataloader — Creates the training DataLoader.
    get_eval_dataloader — Creates the evaluation DataLoader.
    get_test_dataloader — Creates the test DataLoader.
    log — Logs information on the various objects watching training.
    create_optimizer_and_scheduler — Sets up the optimizer and learning rate scheduler if they were not passed at init. Note, that you can also subclass or override the create_optimizer and create_scheduler methods separately.
    create_optimizer — Sets up the optimizer if it wasn’t passed at init.
    create_scheduler — Sets up the learning rate scheduler if it wasn’t passed at init.
    compute_loss - Computes the loss on a batch of training inputs.
    training_step — Performs a training step.
    prediction_step — Performs an evaluation/test step.
    evaluate — Runs an evaluation loop and returns metrics.
    predict — Returns predictions (with metrics if labels are available) on a test set.