# Metric
import requests
import json
import string

import numpy as np
import nltk

nltk.download("wordnet")
from nltk.stem import porter
from nltk.corpus import stopwords
from rouge import Rouge
from rouge_score import tokenize


class Rouge(Rouge):
    """Rouge metric class. Modified for handling bad inputs like empty sentences and white space sentences."""

    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}

            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            if len(hyp) <= 0 or len(ref) <= 0:
                scores.append(
                    {
                        "rouge-1": {"f": np.nan},
                        "rouge-2": {"f": np.nan},
                        "rouge-l": {"f": np.nan},
                        "lengths": {
                            "hyp": len(" ".join(hyp).split()),
                            "ref": len(" ".join(ref).split()),
                        },
                    }
                )
                continue

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(
                    hyp, ref, raw_results=self.raw_results, exclusive=self.exclusive
                )
                sen_score[m] = {s: sc[s] for s in self.stats}

            if self.return_lengths:
                lengths = {
                    "hyp": len(" ".join(hyp).split()),
                    "ref": len(" ".join(ref).split()),
                }
                sen_score["lengths"] = lengths
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        if self.return_lengths:
            scores["lengths"] = {"hyp": 0, "ref": 0}

        count = 0
        for hyp, ref in zip(hyps, refs):
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            if len(hyp) <= 0 or len(ref) <= 0:
                continue

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, exclusive=self.exclusive)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}

            if self.return_lengths:
                scores["lengths"]["hyp"] += len(" ".join(hyp).split())
                scores["lengths"]["ref"] += len(" ".join(ref).split())

            count += 1
        avg_scores = {
            m: {s: scores[m][s] / count for s in self.stats} for m in self.metrics
        }

        if self.return_lengths:
            avg_scores["lengths"] = {
                k: scores["lengths"][k] / count for k in ["hyp", "ref"]
            }

        return avg_scores


class AutomaticNgramEval:
    """
    Automatic N-gram evaluation class using the Rouge and Meteor metrics.
    """

    def __init__(self):
        self.rouge_scorer = Rouge()
        return

    def run_rouge(self, ref_texts, gen_texts, use_aggregator):
        scores = self.rouge_scorer.get_scores(gen_texts, ref_texts, avg=use_aggregator)
        if use_aggregator:
            rouge_1 = scores["rouge-1"]["f"]
            rouge_2 = scores["rouge-2"]["f"]
            rouge_l = scores["rouge-l"]["f"]
            return rouge_1, rouge_2, rouge_l
        else:
            return scores

    def run_meteor(self, ref_texts, gen_texts, use_aggregator):
        scores = []
        for ref_text, gen_text in list(zip(ref_texts, gen_texts)):
            score = round(
                nltk.translate.meteor_score.meteor_score(
                    [ref_text.split()], gen_text.split()
                ),
                3,
            )
            scores.append(score)
        if use_aggregator:
            return np.mean(scores)
        else:
            return scores

    def run_all_evaluation(self, ref_texts, gen_texts, use_aggregator=True):
        if use_aggregator:
            meteor_score = self.run_meteor(ref_texts, gen_texts, use_aggregator)
            rouge_1, rouge_2, rouge_l = self.run_rouge(
                ref_texts, gen_texts, use_aggregator
            )
            return {
                "rouge1": rouge_1,
                "rouge2": rouge_2,
                "rougeL": rouge_l,
                "meteor": meteor_score,
            }
        else:
            meteor_score = self.run_meteor(ref_texts, gen_texts, use_aggregator)
            rouge_score = self.run_rouge(ref_texts, gen_texts, use_aggregator)
            return {"rouge": rouge_score, "meteor": meteor_score}


class AutomaticConceptEval:
    """
    Automatic fact evaluation class using the ROUGE and METEOR metrics on UMLS medical concepts extracted from the summaries.
    """

    def __init__(self):
        return

    def _get_umls_concepts(self, inp, all_concepts=False):
        inp = inp if type(inp) is list else [inp]
        response = requests.post(
            "http://localhost:8123/quickumls",
            data=json.dumps({"data": inp}),
            headers={"FILETYPE": "text_list"},
        )
        outs = json.loads(response.text)
        outs_concepts = (
            outs["concepts"]
            if all_concepts == False
            else outs["concepts"] + outs["raw_concepts"]
        )
        umls_outs = {"term": [], "cuis": []}
        for cnx in outs_concepts:
            for cnx_dict in cnx:
                if cnx_dict["term"] not in umls_outs["term"]:
                    umls_outs["term"].append(cnx_dict["term"])

                umls_outs["cuis"] = list(set(umls_outs["cuis"] + cnx_dict["cuis"]))
        return umls_outs

    def process(self, concepts):
        concepts = [each.lower().strip(string.punctuation).strip() for each in concepts]
        concepts = list(set(concepts))
        return concepts

    def compare(self, ref_concepts, gen_concepts):
        precision = 0
        recall = 0
        fscore = 0

        ## precision is out of all predicted, how many were accurate or found in ref
        true_positives = list(set(ref_concepts).intersection(set(gen_concepts)))
        if gen_concepts:
            precision = len(true_positives) / len(gen_concepts)

        ## recall is out of all in reference, how many was predicted
        if ref_concepts:
            recall = len(true_positives) / len(ref_concepts)

        if precision + recall:
            fscore = (2 * precision * recall) / (precision + recall)
        return precision, recall, fscore

    def run_source_concept_faithfulness(self, ref_sums, gen_sums, use_aggregator=True):
        # df_errors = {'Evidence_Utterances': [], 'Summaries' : [], 'Generated_Summaries' : [], 'Ref_concepts' : [], 'Gen_concepts' : [], 'UMLS_score' : [],}
        all_precision_term = []
        all_recall_term = []
        all_fscore_term = []
        all_precision_cuis = []
        all_recall_cuis = []
        all_fscore_cuis = []

        all_gen_concepts_term = []
        all_gen_concepts_cuis = []

        for ref, gen in zip(ref_sums, gen_sums):
            ref_concepts = self._get_umls_concepts(ref, all_concepts=True)
            gen_concepts = self._get_umls_concepts(gen, all_concepts=True)

            ref_concepts_term = ref_concepts["term"]
            gen_concepts_term = gen_concepts["term"]
            # ref_concepts_term = self.process(ref_concepts['term'])
            # gen_concepts_term = self.process(gen_concepts['term'])
            precision_term, recall_term, fscore_term = self.compare(
                ref_concepts_term, gen_concepts_term
            )
            all_precision_term += [precision_term]
            all_recall_term += [recall_term]
            all_fscore_term += [fscore_term]
            all_gen_concepts_term += [gen_concepts_term]

            ref_concepts_cuis = ref_concepts["cuis"]
            gen_concepts_cuis = gen_concepts["cuis"]
            # ref_concepts_cuis = self.process(ref_concepts['cuis'])
            # gen_concepts_cuis = self.process(gen_concepts['cuis'])
            precision_cuis, recall_cuis, fscore_cuis = self.compare(
                ref_concepts_cuis, gen_concepts_cuis
            )
            all_precision_cuis += [precision_cuis]
            all_recall_cuis += [recall_cuis]
            all_fscore_cuis += [fscore_cuis]
            all_gen_concepts_cuis += [gen_concepts_cuis]

        if use_aggregator:
            return {
                "UMLS_term_f": np.mean(all_fscore_term),
                "UMLS_cuis_f": np.mean(all_fscore_cuis),
                "pred_concepts_term": all_gen_concepts_term,
                "pred_concepts_cuis": all_gen_concepts_cuis,
            }

        else:
            return {
                "UMLS_term_f": all_fscore_term,
                "UMLS_cuis_f": all_fscore_cuis,
                "pred_concepts_term": all_gen_concepts_term,
                "pred_concepts_cuis": all_gen_concepts_cuis,
            }


def remove_stopword_and_punc_in_list(word_tokens):
    stop_words = set(stopwords.words("english"))
    words = [w for w in word_tokens if not w.lower() in stop_words]

    words = []

    for w in word_tokens:
        if w not in stop_words:
            words.append(w)

    words = [
        "".join(c.lower() for c in s if c not in string.punctuation).strip()
        for s in words
    ]
    words = list(set(words))
    words = [s for s in words if s]

    return words


def cal_SAGE(x):
    """
    Compute SAGE metric as per the paper "Improving Summarization with Human Edits"
    """
    # word_level
    pred_words = tokenize.tokenize(x["decoded_preds"], porter.PorterStemmer())
    pred_words = remove_stopword_and_punc_in_list(pred_words)
    word_group1_count = 0
    word_group2_count = 0
    word_group3_count = 0

    group1_words_in_pred = []
    group2_words_in_pred = []

    for word in x["word_group1"]:
        if word in pred_words:
            word_group1_count += 1
            group1_words_in_pred.append(word)
    for word in x["word_group2"]:
        if word in pred_words:
            word_group2_count += 1
            group2_words_in_pred.append(word)
    for word in x["word_group3"]:
        if word in pred_words:
            word_group3_count += 1

    # concept_level
    concept_group1_term_count = 0
    concept_group2_term_count = 0
    concept_group3_term_count = 0

    for concept in x["concepts_group1_term"]:
        if concept in x["pred_concepts_term"]:
            concept_group1_term_count += 1
    for concept in x["concepts_group2_term"]:
        if concept in x["pred_concepts_term"]:
            concept_group2_term_count += 1
    for concept in x["concepts_group3_term"]:
        if concept in x["pred_concepts_term"]:
            concept_group3_term_count += 1

    concept_group1_cuis_count = 0
    concept_group2_cuis_count = 0
    concept_group3_cuis_count = 0
    for concept in x["concepts_group1_cuis"]:
        if concept in x["pred_concepts_cuis"]:
            concept_group1_cuis_count += 1
    for concept in x["concepts_group2_cuis"]:
        if concept in x["pred_concepts_cuis"]:
            concept_group2_cuis_count += 1
    for concept in x["concepts_group3_cuis"]:
        if concept in x["pred_concepts_cuis"]:
            concept_group3_cuis_count += 1

    return (
        group1_words_in_pred,
        group2_words_in_pred,
        word_group1_count,
        word_group2_count,
        word_group3_count,
        concept_group1_term_count,
        concept_group2_term_count,
        concept_group3_term_count,
        concept_group1_cuis_count,
        concept_group2_cuis_count,
        concept_group3_cuis_count,
    )
