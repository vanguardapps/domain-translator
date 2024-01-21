### Three ways to convert HuggingFace dataset to Pandas dataframe:

1.  import pandas as pd # v1.3.5
    from datasets import load_dataset # v2.11.0

2.  df = load_dataset('squad_v2') # Loading the SQuAD dataset from huggingface.
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

### Notes on Training for the Task of Multilingual Translation

It's important to note that several major breakthroughs have occurred in the academic realm of transformer-based multilingual translation. Of these, a couple are of particular note:

1. It has been found that training on bilingual in-domain corpus data is far inferior to using multilingual in-domain corpus data, even if the target language is _not_ related to the additional languages. Just having in-domain corpora is helpful, and going from any language to another, was the finding (Verma et al., 2022)
2. Furthermore, it has been found that a _domain first_ approach requires significantly less in-domain data than a language first approach. With a minimal amount of in-domain multilingual text, it is possible to attain results that require much more bitext in the actual bilingual target. This allows us to leverage many in-domain multilingual corpora that are not specifically in our target language set.
3. GPT-J has been used to augment multilingual data or even create entire in-domain corpora for free. It is possible to make as much of this data as you want, according to Moslem et al., 2022. This is also a technique we seek to explore here, perhaps with GPT-J itself, or perhaps with other GPT-3-like models.
4. Chu et al. (2017) mixes fine-tuning on in-domain along with out-of-domain data in an effort to avoid overfitting, and this too is something we wish to explore in this project.

### Steps to Achieve Results

1. Find in-domain multilingual corpora, whether they are in the language target or not.
2. Read and test the generation of in-domain bitext using GPT-J or similar.
3. Follow steps outlined in Verma et al. 2022 for overal _domain first_ training.
4. If overfitting presents itself as a problem, try the approach used by Chu et al. (4 above) and mix in-domain and out-of-domain data.
5. If needed, continue to fine-tune in a language first manner.
6. Compute metrics and iterate as necessary.
7. For final inference endpoint, see infrastructure generated as Terraform by Amazon Q below

### Infrastructure Generated as Terraform by Amazon Q

```terraform
# Here is some Terraform code that would build out the baseline MVP translation system we discussed:

# Create a VPC to launch resources into
resource "aws_vpc" "translation" { cidr_block = "10.0.0.0/16"

tags = { Name = "translation" }}

# Create an internet gateway to give the VPC internet access
resource "aws_internet_gateway" "igw" { vpc_id = aws_vpc.translation.id

tags = { Name = "translation" }}

# Route table for public subnets
resource "aws_route_table" "public" { vpc_id = aws_vpc.translation.id

route { cidr_block = "0.0.0.0/0" gateway_id = aws_internet_gateway.igw.id }

tags = { Name = "public" }}

# Create a public subnet to launch the EC2 instance into
resource "aws_subnet" "public" { vpc_id = aws_vpc.translation.id cidr_block = "10.0.1.0/24" availability_zone = "us-east-1a" map_public_ip_on_launch = true

tags = { Name = "public" }}

# Associate the public route table to the public subnet
resource "aws_route_table_association" "public" { subnet_id = aws_subnet.public.id route_table_id = aws_route_table.public.id}

# Launch a t2.micro EC2 instance
resource "aws_instance" "translation" { ami = "ami-1234567890abcdef0" instance_type = "t2.micro" subnet_id = aws_subnet.public.id

tags = { Name = "translation" }}

# This sets up the basic VPC, public subnet, internet gateway and launches a t2.micro instance as specified. Let me know if any part
# needs more explanation or if you need help adding additional components like S3, ALB etc.
```

### Useful Scripts

- Drop all lines from bash history involving `pip` into a file called output.txt:

```shell
history | grep "pip" > output.txt
```

### Related Repos

- https://github.com/OpenNMT/OpenNMT-tf
- https://github.com/ymoslem/MT-Preparation
- https://github.com/ymoslem/MT-LM
- https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models
- https://github.com/OpenNMT/CTranslate2
- https://github.com/mjpost/sacrebleu
- https://github.com/Unbabel/COMET
