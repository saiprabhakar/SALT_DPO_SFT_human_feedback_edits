# Parts of the code are adapted from the Hugging Face TRL library

import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from transformers.utils import is_peft_available
from trl.trainer.utils import pad_to_length

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_int8_training

from torch.nn.utils.rnn import pad_sequence

from sequence_alignment import needle
from metrics import AutomaticNgramEval

ngram_eval = AutomaticNgramEval()

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt + chosen/rejected responses.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

        assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
        assert (
            self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
        ), f"Chosen response contains EOS token: {chosen}"
        assert (
            self.tokenizer.eos_token_id not in rejected_tokens["input_ids"]
        ), f"Rejected response contains EOS token: {rejected}"

        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}
            
        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        batch = {}

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
    

class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        alpha1 (`float`, defaults to 0):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        alpha2 (`float`, defaults to 1):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        omega1 (`float`, defaults to 1):
            The omeg1a factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        omega2 (`float`, defaults to 1):
            The omega2 factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        alignment_function (`str`, defaults to `None`):
            which alignment_function will be used
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        alpha1: float = 0.0,
        alpha2: float = 1.0,
        # omega1: float = 1.0,
        # omega2: float = 1.0,
        # S_generated_C_weight: float = 1.0,
        # S_generated_D_weight: float = -0.1,
        # S_generated_S_weight: float = -0.1,
        # S_edited_C_weight: float = 1.0,
        # S_edited_I_weight: float = 1.0,
        # S_edited_S_weight: float = 1.0,
        output_dir: str = './',
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        # alignment_function: Optional[str] = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
                self.ref_model = prepare_model_for_int8_training(ref_model)
            else:
                self.ref_model = ref_model
            model = get_peft_model(model, peft_config)
            self.ref_model = get_peft_model(self.ref_model, peft_config)
            model.print_trainable_parameters()
        else:
            self.ref_model = ref_model

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.omega1 = omega1
        self.omega2 = omega2

        # self.S_generated_C_weight = S_generated_C_weight
        # self.S_generated_D_weight = S_generated_D_weight
        # self.S_generated_S_weight = S_generated_S_weight
        # self.S_edited_C_weight = S_edited_C_weight
        # self.S_edited_I_weight = S_edited_I_weight
        # self.S_edited_S_weight = S_edited_S_weight

        self.output_dir = output_dir

        # if alignment_function is None:
        #     self.alignment_function = 'dpo'
        # else:
        #     self.alignment_function = alignment_function

        self.generate_max_length=max_length

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    # def sequence_alignment(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    #     rejected_salt_weight = []
    #     chosen_salt_weight = []
        
    #     chosen_align_mask = batch['chosen_labels'] != -100
    #     chosen_input_ids = batch['chosen_input_ids'] * chosen_align_mask
        
    #     rejected_align_mask = batch['rejected_labels'] != -100
    #     rejected_input_ids = batch['rejected_input_ids'] * rejected_align_mask
        
    #     for S_generated, S_edited in zip(rejected_input_ids.tolist(), chosen_input_ids.tolist()):
    #         # Create the instance
    #         alignment = needle.NeedlemanWunsch(S_generated, S_edited) #needle.NeedlemanWunsch, smith.SmithWaterman
    #         alignment.gap_character = -100
    #         # Make the alignment
    #         alignment.align()
    #         # Get the score
    #         alignment.get_score()
    #         # Get the sequences aligned as lists
    #         al_generated, al_edited = alignment.get_aligned_sequences("list_of_int")
    #         w_generated, w_edited = needle.get_position_status(al_generated, 
    #                                                     al_edited,
    #                                                     S_generated_C_weight = self.S_generated_C_weight,
    #                                                     S_generated_D_weight = self.S_generated_D_weight,
    #                                                     S_generated_S_weight = self.S_generated_S_weight,
    #                                                     S_edited_C_weight = self.S_edited_C_weight,
    #                                                     S_edited_I_weight = self.S_edited_I_weight,
    #                                                     S_edited_S_weight = self.S_edited_S_weight)
        
    #         rejected_salt_weight.append(w_generated)
    #         chosen_salt_weight.append(w_edited)

    #     batch['rejected_salt_weight'] = torch.tensor(rejected_salt_weight, dtype=torch.float32).to(self.accelerator.device)
    #     batch['chosen_salt_weight'] = torch.tensor(chosen_salt_weight, dtype=torch.float32).to(self.accelerator.device)
            
    #     return batch

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        # if self.alignment_function == 'salt':
        #     batch = self.sequence_alignment(batch)
        
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        
        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        dpo_losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        rewards_mask = pi_logratios != 0

        return dpo_losses, chosen_rewards, rejected_rewards, rewards_mask

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = all_logits[batch["chosen_input_ids"].shape[0] :]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)

        dpo_losses, chosen_rewards, rejected_rewards, rewards_mask = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # if self.alignment_function == 'dpo':
        losses = (-1) * self.alpha1 * policy_chosen_logps + self.alpha2 * dpo_losses
        losses = losses.mean()
        # elif self.alignment_function == 'sft':
        #     losses = (-1) * policy_chosen_logps
        #     losses = losses.mean()
        # elif self.alignment_function == 'salt':
        #     losses = (-1) * self.omega1 * policy_chosen_salt_logps + (-1) * self.omega2 * policy_rejected_salt_logps

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards)[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits[rewards_mask].detach().cpu().numpy().mean()

        if train_eval == "eval":
            policy_output_decoded, reference_output_decoded = self.get_batch_samples(model, batch)

            # Create a DataFrame from the dictionary
            batch.update({
               'policy_output': policy_output_decoded, 
               'reference_output': reference_output_decoded 
            })

            # eval generated summary for policy
            eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], policy_output_decoded)

            eval_dict = {'eval_metrics_policy_'+k: round(v, 4) for k, v in eval_dict.items()}
    
            # eval generated summary for ref model
            ref_eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], reference_output_decoded)
            ref_eval_dict = {'eval_metrics_ref_'+k: round(v, 4) for k, v in ref_eval_dict.items()}
    
            eval_dict.update(ref_eval_dict)
            metrics.update(eval_dict)

        return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        # print(loss, metrics)

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # torch.autograd.set_detect_anomaly(True)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # with torch.autograd.detect_anomaly():
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.generate_max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.generate_max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.generate_max_length, self.tokenizer.pad_token_id)
        reference_output = pad_to_length(reference_output, self.generate_max_length, self.tokenizer.pad_token_id)

        policy_output = policy_output[:, batch["prompt_input_ids"].shape[-1]:]
        reference_output = reference_output[:, batch["prompt_input_ids"].shape[-1]:]

        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)



class SALTTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        alpha1 (`float`, defaults to 0):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        alpha2 (`float`, defaults to 1):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        omega1 (`float`, defaults to 1):
            The omeg1a factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        omega2 (`float`, defaults to 1):
            The omega2 factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        alignment_function (`str`, defaults to `None`):
            which alignment_function will be used
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        # beta: float = 0.1,
        # alpha1: float = 0.0,
        # alpha2: float = 1.0,
        omega1: float = 1.0,
        omega2: float = 1.0,
        S_generated_C_weight: float = 1.0,
        S_generated_D_weight: float = -0.1,
        S_generated_S_weight: float = -0.1,
        S_edited_C_weight: float = 1.0,
        S_edited_I_weight: float = 1.0,
        S_edited_S_weight: float = 1.0,
        output_dir: str = './',
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        # alignment_function: Optional[str] = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
                self.ref_model = prepare_model_for_int8_training(ref_model)
            else:
                self.ref_model = ref_model
            model = get_peft_model(model, peft_config)
            self.ref_model = get_peft_model(self.ref_model, peft_config)
            model.print_trainable_parameters()
        else:
            self.ref_model = ref_model

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        # self.beta = beta
        # self.alpha1 = alpha1
        # self.alpha2 = alpha2
        self.omega1 = omega1
        self.omega2 = omega2

        self.S_generated_C_weight = S_generated_C_weight
        self.S_generated_D_weight = S_generated_D_weight
        self.S_generated_S_weight = S_generated_S_weight
        self.S_edited_C_weight = S_edited_C_weight
        self.S_edited_I_weight = S_edited_I_weight
        self.S_edited_S_weight = S_edited_S_weight

        self.output_dir = output_dir

        # if alignment_function is None:
        #     self.alignment_function = 'dpo'
        # else:
        #     self.alignment_function = alignment_function

        self.generate_max_length=max_length

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def sequence_alignment(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        rejected_salt_weight = []
        chosen_salt_weight = []
        
        chosen_align_mask = batch['chosen_labels'] != -100
        chosen_input_ids = batch['chosen_input_ids'] * chosen_align_mask
        
        rejected_align_mask = batch['rejected_labels'] != -100
        rejected_input_ids = batch['rejected_input_ids'] * rejected_align_mask
        
        for S_generated, S_edited in zip(rejected_input_ids.tolist(), chosen_input_ids.tolist()):
            # Create the instance
            alignment = needle.NeedlemanWunsch(S_generated, S_edited) #needle.NeedlemanWunsch, smith.SmithWaterman
            alignment.gap_character = -100
            # Make the alignment
            alignment.align()
            # Get the score
            alignment.get_score()
            # Get the sequences aligned as lists
            al_generated, al_edited = alignment.get_aligned_sequences("list_of_int")
            w_generated, w_edited = needle.get_position_status(al_generated, 
                                                        al_edited,
                                                        S_generated_C_weight = self.S_generated_C_weight,
                                                        S_generated_D_weight = self.S_generated_D_weight,
                                                        S_generated_S_weight = self.S_generated_S_weight,
                                                        S_edited_C_weight = self.S_edited_C_weight,
                                                        S_edited_I_weight = self.S_edited_I_weight,
                                                        S_edited_S_weight = self.S_edited_S_weight)
        
            rejected_salt_weight.append(w_generated)
            chosen_salt_weight.append(w_edited)

        batch['rejected_salt_weight'] = torch.tensor(rejected_salt_weight, dtype=torch.float32).to(self.accelerator.device)
        batch['chosen_salt_weight'] = torch.tensor(chosen_salt_weight, dtype=torch.float32).to(self.accelerator.device)
            
        return batch

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        # if self.alignment_function == 'salt':
        batch = self.sequence_alignment(batch)
        
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        
        return concatenated_batch

    # def dpo_loss(
    #     self,
    #     policy_chosen_logps: torch.FloatTensor,
    #     policy_rejected_logps: torch.FloatTensor,
    #     reference_chosen_logps: torch.FloatTensor,
    #     reference_rejected_logps: torch.FloatTensor,
    #     reference_free: bool = False,
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Compute the DPO loss for a batch of policy and reference model log probabilities.

    #     Args:
    #         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
    #         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
    #         reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
    #         reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
    #         beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
    #         reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    #     Returns:
    #         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
    #         The losses tensor contains the DPO loss for each example in the batch.
    #         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    #     """
    #     pi_logratios = policy_chosen_logps - policy_rejected_logps
    #     ref_logratios = reference_chosen_logps - reference_rejected_logps

    #     if reference_free:
    #         ref_logratios = 0

    #     logits = pi_logratios - ref_logratios

    #     dpo_losses = -F.logsigmoid(self.beta * logits)
    #     chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
    #     rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
    #     rewards_mask = pi_logratios != 0

    #     return dpo_losses, chosen_rewards, rejected_rewards, rewards_mask

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _get_batch_chosen_salt_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        labels_weight: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        labels_weight = labels_weight[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        zero_for_here = torch.tensor(0, dtype=labels_weight.dtype, device=labels_weight.device)
        likelihood_weight = torch.where(labels_weight > zero_for_here, labels_weight, zero_for_here)
        
        likelihood_weight = likelihood_weight*loss_mask
        per_token_logps = per_token_logps*likelihood_weight
        
        if average_log_prob:
            likelihood_token_num = (likelihood_weight!=0).sum()
            if likelihood_token_num == 0:
                return torch.tensor(0, device=logits.device)
            else:
                return (per_token_logps.sum(-1) / likelihood_token_num).mean()
        else:
            return per_token_logps.sum(-1).mean()


    def _get_batch_rejected_salt_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        labels_weight: torch.LongTensor,
        average_log_prob: bool = True,
        calculate_liklihood_token_log_prob: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        labels_weight = labels_weight[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        # likelihood loss for rejected
        if calculate_liklihood_token_log_prob:
            likelihood_per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
            zero_for_here = torch.tensor(0, dtype=labels_weight.dtype, device=labels_weight.device)
            likelihood_weight = torch.where(labels_weight > zero_for_here, labels_weight, zero_for_here)
            
            likelihood_weight = likelihood_weight*loss_mask
            likelihood_per_token_logps = likelihood_per_token_logps*likelihood_weight
            
            if average_log_prob:
                likelihood_token_num = (likelihood_weight!=0).sum()
                if likelihood_token_num == 0:
                    likelihood_per_token_logps = torch.tensor(0, device=logits.device)
                else:
                    likelihood_per_token_logps = (likelihood_per_token_logps.sum(-1) / likelihood_token_num).mean()
            else:
                likelihood_per_token_logps = likelihood_per_token_logps.sum(-1).mean()

        # unlikelihood loss for rejected
        probs = F.softmax(logits, dim=-1)  
        #torch.log1p(x) := log(x+1)
        #log_one_minus_probs = torch.log1p(-probs)
        one_minus_probs = 1.0 - probs
        one_minus_probs = one_minus_probs + (one_minus_probs==0).float() * 1e-8
        log_one_minus_probs = torch.log(one_minus_probs)
        unlikelihood_per_token_logps = torch.gather(log_one_minus_probs.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        zero_for_here = torch.tensor(0, dtype=labels_weight.dtype, device=labels_weight.device)
        unlikelihood_weight = torch.where(labels_weight < zero_for_here, -1 * labels_weight, zero_for_here)
        
        unlikelihood_weight = unlikelihood_weight*loss_mask
        unlikelihood_per_token_logps = unlikelihood_per_token_logps*unlikelihood_weight

        if average_log_prob:
            unlikelihood_token_num = (unlikelihood_weight!=0).sum()
            if unlikelihood_token_num == 0:
                unlikelihood_per_token_logps = torch.tensor(0.0, device=logits.device)
            else:
                unlikelihood_per_token_logps = (unlikelihood_per_token_logps.sum(-1) / unlikelihood_token_num).mean()
        else:
            unlikelihood_per_token_logps = unlikelihood_per_token_logps.sum(-1).mean()

        if calculate_liklihood_token_log_prob:
            return likelihood_per_token_logps + unlikelihood_per_token_logps
        else:
            return unlikelihood_per_token_logps


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = all_logits[batch["chosen_input_ids"].shape[0] :]
        
        # if self.alignment_function in ['salt']:
        chosen_salt_logps = self._get_batch_chosen_salt_logps(
            chosen_logits,
            concatenated_batch["concatenated_labels"][: batch["chosen_input_ids"].shape[0]],
            concatenated_batch["concatenated_salt_weight"][: batch["chosen_input_ids"].shape[0]],
            average_log_prob=True,
        )
        rejected_salt_logps = self._get_batch_rejected_salt_logps(
            rejected_logits,
            concatenated_batch["concatenated_labels"][batch["chosen_input_ids"].shape[0] :],
            concatenated_batch["concatenated_salt_weight"][batch["chosen_input_ids"].shape[0] :],
            average_log_prob=True,
            calculate_liklihood_token_log_prob=False
        )
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_salt_logps, rejected_salt_logps)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_salt_logps,
            policy_rejected_salt_logps
        ) = self.concatenated_forward(model, batch)
        # with torch.no_grad():
        #     (
        #         reference_chosen_logps,
        #         reference_rejected_logps,
        #         _,
        #         _,
        #         _,
        #         _,
        #     ) = self.concatenated_forward(self.ref_model, batch)

        # dpo_losses, chosen_rewards, rejected_rewards, rewards_mask = self.dpo_loss(
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     reference_chosen_logps,
        #     reference_rejected_logps,
        # )
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # if self.alignment_function == 'dpo':
        # losses = (-1) * self.alpha1 * policy_chosen_logps + self.alpha2 * dpo_losses
        # losses = losses.mean()
        # elif self.alignment_function == 'sft':
        #     losses = (-1) * policy_chosen_logps
        #     losses = losses.mean()
        # elif self.alignment_function == 'salt':

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        rewards_mask = pi_logratios != 0
        losses = (-1) * self.omega1 * policy_chosen_salt_logps + (-1) * self.omega2 * policy_rejected_salt_logps

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards)[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits[rewards_mask].detach().cpu().numpy().mean()

        if train_eval == "eval":
            policy_output_decoded, reference_output_decoded = self.get_batch_samples(model, batch)

            # Create a DataFrame from the dictionary
            batch.update({
               'policy_output': policy_output_decoded, 
               'reference_output': reference_output_decoded 
            })

            # eval generated summary for policy
            eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], policy_output_decoded)

            eval_dict = {'eval_metrics_policy_'+k: round(v, 4) for k, v in eval_dict.items()}
    
            # eval generated summary for ref model
            ref_eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], reference_output_decoded)
            ref_eval_dict = {'eval_metrics_ref_'+k: round(v, 4) for k, v in ref_eval_dict.items()}
    
            eval_dict.update(ref_eval_dict)
            metrics.update(eval_dict)

        return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        # print(loss, metrics)

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # torch.autograd.set_detect_anomaly(True)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # with torch.autograd.detect_anomaly():
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.generate_max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.generate_max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.generate_max_length, self.tokenizer.pad_token_id)
        reference_output = pad_to_length(reference_output, self.generate_max_length, self.tokenizer.pad_token_id)

        policy_output = policy_output[:, batch["prompt_input_ids"].shape[-1]:]
        reference_output = reference_output[:, batch["prompt_input_ids"].shape[-1]:]

        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)


class SFTTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        alpha1 (`float`, defaults to 0):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        alpha2 (`float`, defaults to 1):
            The alpha factor in Edit-DPO loss (alpha1 * chosen-SFT-loss + alpha2 * DPO-loss).
        omega1 (`float`, defaults to 1):
            The omeg1a factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        omega2 (`float`, defaults to 1):
            The omega2 factor in SALT loss: -self.omega1 * policy_chosen_salt_logps - self.omega2 * policy_rejected_salt_logps
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        alignment_function (`str`, defaults to `None`):
            which alignment_function will be used
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        # ref_model: Union[PreTrainedModel, nn.Module] = None,
        # beta: float = 0.1,
        # alpha1: float = 0.0,
        # alpha2: float = 1.0,
        # omega1: float = 1.0,
        # omega2: float = 1.0,
        # S_generated_C_weight: float = 1.0,
        # S_generated_D_weight: float = -0.1,
        # S_generated_S_weight: float = -0.1,
        # S_edited_C_weight: float = 1.0,
        # S_edited_I_weight: float = 1.0,
        # S_edited_S_weight: float = 1.0,
        output_dir: str = './',
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        # alignment_function: Optional[str] = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
                # self.ref_model = prepare_model_for_int8_training(ref_model)
            # else:
            #     self.ref_model = ref_model
            model = get_peft_model(model, peft_config)
            # self.ref_model = get_peft_model(self.ref_model, peft_config)
            model.print_trainable_parameters()
        # else:
            # self.ref_model = ref_model

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        # self.beta = beta
        # self.alpha1 = alpha1
        # self.alpha2 = alpha2
        # self.omega1 = omega1
        # self.omega2 = omega2

        # self.S_generated_C_weight = S_generated_C_weight
        # self.S_generated_D_weight = S_generated_D_weight
        # self.S_generated_S_weight = S_generated_S_weight
        # self.S_edited_C_weight = S_edited_C_weight
        # self.S_edited_I_weight = S_edited_I_weight
        # self.S_edited_S_weight = S_edited_S_weight

        self.output_dir = output_dir

        # if alignment_function is None:
        #     self.alignment_function = 'dpo'
        # else:
        #     self.alignment_function = alignment_function

        self.generate_max_length=max_length

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            # self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        
        return concatenated_batch

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = all_logits[batch["chosen_input_ids"].shape[0] :]
        
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        # with torch.no_grad():
        #     (
        #         reference_chosen_logps,
        #         reference_rejected_logps,
        #         _,
        #         _,
        #         _,
        #         _,
        #     ) = self.concatenated_forward(self.ref_model, batch)

        # dpo_losses, chosen_rewards, rejected_rewards, rewards_mask = self.dpo_loss(
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     reference_chosen_logps,
        #     reference_rejected_logps,
        # )
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # if self.alignment_function == 'dpo':
        # losses = (-1) * self.alpha1 * policy_chosen_logps + self.alpha2 * dpo_losses
        # losses = losses.mean()
        # elif self.alignment_function == 'sft':
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        rewards_mask = pi_logratios != 0
        losses = (-1) * policy_chosen_logps
        losses = losses.mean()
        # elif self.alignment_function == 'salt':


        # losses = (-1) * self.omega1 * policy_chosen_salt_logps + (-1) * self.omega2 * policy_rejected_salt_logps

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies[rewards_mask].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards)[rewards_mask].cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits[rewards_mask].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits[rewards_mask].detach().cpu().numpy().mean()

        if train_eval == "eval":
            policy_output_decoded = self.get_batch_samples(model, batch)

            # Create a DataFrame from the dictionary
            batch.update({
               'policy_output': policy_output_decoded, 
            #    'reference_output': reference_output_decoded 
            })

            # eval generated summary for policy
            eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], policy_output_decoded)

            eval_dict = {'eval_metrics_policy_'+k: round(v, 4) for k, v in eval_dict.items()}
    
            # eval generated summary for ref model
            # ref_eval_dict = ngram_eval.run_all_evaluation(batch['chosen_response_only'], reference_output_decoded)
            # ref_eval_dict = {'eval_metrics_ref_'+k: round(v, 4) for k, v in ref_eval_dict.items()}
    
            # eval_dict.update(ref_eval_dict)
            metrics.update(eval_dict)

        return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # torch.autograd.set_detect_anomaly(True)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # with torch.autograd.detect_anomaly():
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.generate_max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # reference_output = self.ref_model.generate(
        #     input_ids=batch["prompt_input_ids"],
        #     attention_mask=batch["prompt_attention_mask"],
        #     max_length=self.generate_max_length,
        #     do_sample=False,
        #     pad_token_id=self.tokenizer.pad_token_id,
        # )

        policy_output = pad_to_length(policy_output, self.generate_max_length, self.tokenizer.pad_token_id)
        # reference_output = pad_to_length(reference_output, self.generate_max_length, self.tokenizer.pad_token_id)

        policy_output = policy_output[:, batch["prompt_input_ids"].shape[-1]:]
        # reference_output = reference_output[:, batch["prompt_input_ids"].shape[-1]:]

        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        # reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded#, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

