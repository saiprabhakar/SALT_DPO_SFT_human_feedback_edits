# LearnFromHumanEdit

## Installation
If using `conda`, you can get this to work as follows:

```
conda create -n salt python=3.8
conda activate salt
```

We have experimented with 11.7 and 10.2 cuda version, but this release should work with more recent versions as well.
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```
or 

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
```

Install other packages:
```
conda install -c conda-forge matplotlib
conda install -c conda-forge spacy
conda install -c conda-forge scipy
python -m spacy download en_core_web_sm
pip install nltk
pip install ipdb
pip install rouge
pip install rouge-score
pip install trl
pip install minineedle
pip install nltk

pip install datasets
pip install transformers
```
If you want to use qlora for llm:
```
pip install -q -U bitsandbytes 
pip install -q -U git+https://github.com/huggingface/peft.git 
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

## Run the trainer

```
python DPO_trainer.py
python SFT_trainer.py
python SALT_trainer.py
```

## TODO
- Adapt the codes *_trainer.py 
    - Save output models
    - Save outputs
- Modify the classes in dpo.py and rename it to be more generic
- Add link to paper and bib
- Add dataset
- Do we need wandb instructions



## Poetry installation

If you encountered a problem with poetry installation with torch versions (python 3.10)

`poetry run pip install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`