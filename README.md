# LearnFromHumanEdit

## Poetry installation

`poetry install`

- Add HG auth token to the project by creating a `hg_secret` file
- `python -m spacy download en_core_web_sm`
- If you encountered a problem with poetry installation with torch versions (python 3.10) do:

```poetry run pip install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```


## Run training

```
python DPO_trainer.py
python SFT_trainer.py
python SALT_trainer.py
```

## TODO
- Add link to paper and bib
- Add full dataset