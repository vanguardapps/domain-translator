## Setup / Installation
If this is your first time using this repo, run the executable bash file `install.sh` like so from the project root:
```shell
./install.sh
```
Note: you may have to run `chmod 755 install.sh` to make `install.sh` executable on your system, depending on how you downloaded this repo.

## Usage
To train the model on the dataset, activate the virtual environment if you have not like so (from the project root):
```shell
source src/.env/bin/activate
```
And then run the following to perform the actual training:
```shell
python3 src/run.py
```
