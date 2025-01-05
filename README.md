#  Improving Summarization with Human Edits

Abstract

Recent work has shown the promise of learning with human feedback paradigms to produce human-determined high-quality text. Existing works use human feedback to train large
language models (LLMs) in general domain
abstractive summarization and have obtained
summary quality exceeding traditional likelihood training. In this paper, we focus on a
less explored form of human feedback – Human Edits. We propose Sequence Alignment
(un)Likelihood Training (SALT), a novel technique to use both the human-edited and modelgenerated data together in the training loop.
In addition, we demonstrate simulating Human Edits with ground truth summaries coming from existing training data – Imitation edits,
along with the model-generated summaries obtained after the training, to reduce the need
for expensive human-edit data. In our experiments, we extend human feedback exploration
from general domain summarization to medical
domain summarization. Our results1 demonstrate the effectiveness of SALT in improving
the summary quality with Human and Imitation Edits. Through additional experiments, we
show that SALT outperforms the conventional
RLHF method (designed for human preferences) – DPO, when applied to human-edit data.
We hope the evidence in our paper prompts
researchers to explore, collect and better use
different human feedback approaches scalably

- Yao, Zonghai, Benjamin Schloss, and Sai Selvaraj. "Improving Summarization with Human Edits." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023.

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

## Citing SALT
If SALT or this repository is useful in your own research, you can use the following BibTeX entry:

```
@inproceedings{yao2023improving,
  title={Improving Summarization with Human Edits},
  author={Yao, Zonghai and Schloss, Benjamin and Selvaraj, Sai},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={2604--2620},
  year={2023}
}
```