# Improving Summarization with Human Edits

## Abstract
Recent work has highlighted the potential of learning paradigms incorporating human feedback to generate high-quality text. While most approaches leverage human preferences to fine-tune large language models (LLMs) for abstractive summarization, this work explores a less-studied form of feedback: **Human Edits**. 

We introduce **Sequence Alignment (un)Likelihood Training (SALT)**, a novel training technique that integrates human-edited and model-generated data into the training loop. To address the scarcity of human-edited data, we propose **Imitation Edits**, where ground truth summaries from training data simulate the editing process. These edits, combined with model-generated summaries, reduce the need for costly human feedback.

Through experiments, we extend human feedback exploration to the **medical domain** summarization task. Our results demonstrate that SALT improves summary quality and outperforms the conventional **Direct Preference Optimization (DPO)** method when applied to human-edited data. We hope this work inspires further research into scalable, effective ways to incorporate human feedback for text summarization.

- **Yao, Zonghai, Benjamin Schloss, and Sai Selvaraj.** "Improving Summarization with Human Edits." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023.

---

## What is SALT?
**Sequence Alignment (un)Likelihood Training (SALT)** is a training framework designed to improve text summarization models by leveraging both human edits and model-generated outputs. SALT encourages models to align with high-quality, human-edited summaries while disfavoring undesirable outputs.

Key components of SALT include:
- **Human Edits:** Real-world edits applied to model-generated summaries.
- **Imitation Edits:** Simulated edits derived from ground truth summaries to reduce dependency on expensive human feedback.
- **Likelihood and Unlikelihood Training:** A dual approach that encourages desirable edits while penalizing undesirable ones.

---

## What is DPO?
**Direct Preference Optimization (DPO)** is a method that fine-tunes LLMs based on human preference scores. DPO focuses on aligning the model's output distribution with human-preferred samples (chosen responses) and away from less-preferred ones (rejected responses). While effective for general human preference feedback, our experiments show that SALT achieves better performance on human-edited data.

---

## Setup and Installation

1. Install dependencies with [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```

2. Add your Hugging Face authentication token to the project by creating an `hg_secret` file.

3. Download spaCy's English language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. If you encounter compatibility issues with Poetry and PyTorch (Python 3.10), run the following command:
   ```bash
   poetry run pip install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Training
Run the following scripts to train models:

- **Train DPO:**
  ```bash
  python train_DPO.py
  ```

- **Train SALT:**
  ```bash
  python train_SALT.py
  ```

- **Train SFT (Supervised Fine-Tuning):**
  ```bash
  python train_SFT.py
  ```

---

## Metrics
The following metrics are used to evaluate summarization quality:

- **ROUGE:** Standard metric for summarization quality.
- **ConceptRouge:** Evaluates the inclusion of domain-specific concepts. 
  - Implementation: Refer to `AutomaticConceptEval` in `utils/metrics.py`.
  - Setup: Install [quickumls](https://pypi.org/project/medspacy-quickumls/) and set up its API endpoint at `http://localhost:8123/quickumls`.
- **SAGE:** A novel metric introduced in our paper to evaluate summary quality.
  - Implementation: Refer to `cal_SAGE` in `utils/metrics.py`.

---

## Example Setting
In our training framework, we define:
- **Chosen Sentences:** Sentences preferred based on human edits.
- **Rejected Sentences:** Sentences identified as suboptimal by human edits.
- **Edit Simulation:** Ground truth summaries are transformed to simulate human edits, reducing reliance on human feedback.

During training, SALT optimizes the likelihood of chosen sentences while penalizing the unlikelihood of rejected sentences, guiding the model toward generating high-quality summaries.

---

## Citation
If SALT or this repository is useful in your research, please cite our work:

```bibtex
@inproceedings{yao2023improving,
  title={Improving Summarization with Human Edits},
  author={Yao, Zonghai and Schloss, Benjamin and Selvaraj, Sai},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={2604--2620},
  year={2023}
}
