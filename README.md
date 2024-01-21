## Setup / Installation

If this is your first time using this repo, run the following installations within your preferred virtual environment:

```shell
pip install sentencepiece
pip install tokenizers
pip install evaluate
pip install torch tensorflow
pip install protobuf
pip install accelerate -U
pip install sacrebleu
pip install regex
pip install nltk
```

## Environment

There are a few environment variables that need to be set for this repo to run:

```shell
export TOKENIZERS_PARALLELISM=false
```

## Usage

To train the model on the dataset, activate the virtual environment if you have not like so (from the project root):

```shell
source src/<your-env>/bin/activate
```

And then run the following to perform the actual training:

```shell
python3 src/run.py
```
