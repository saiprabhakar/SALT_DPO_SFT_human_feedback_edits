import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_from_disk  # , load_dataset, load_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

# from transformers.trainer_utils import EvalPrediction# , EvalLoopOutput
# from transformers.trainer_pt_utils import find_batch_size, nested_concat

# import pandas as pd

from peft import LoraConfig, get_peft_model

# from torch.utils.data import DataLoader

from trainer.dpo import DPOTrainer, SALTTrainer, SFTTrainer


def extract_prompt(prompt_and_response):
    search_term = "\n\nGenerate the corresponding Discharge Instructions according to the input article:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def load_dataset(
    split: str,
    sanity_check: bool = False,
    alignment_function: str = "sft",
    silent: bool = False,
    cache_dir: str = None,
) -> Dataset:
    """Load the dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      Conversation <prompt>\n\nSummary
    """
    # dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if alignment_function in ["sft", "dpo", "salt"]:
        dataset = load_from_disk("data/avs/" + split)

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 10)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    dataset = dataset.map(split_prompt_and_responses)

    return dataset


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    alpha1: Optional[float] = field(
        default=0.0, metadata={"help": "the alpha parameter for Edit-DPO loss"}
    )
    alpha2: Optional[float] = field(
        default=1.0, metadata={"help": "the alpha parameter for Edit-DPO loss"}
    )
    omega1: Optional[float] = field(
        default=1.0, metadata={"help": "the omega parameter for SALT loss"}
    )
    omega2: Optional[float] = field(
        default=1.0, metadata={"help": "the omega parameter for SALT loss"}
    )
    S_generated_C_weight: Optional[float] = field(
        default=1.0, metadata={"help": "sequence alignment weights"}
    )
    S_generated_D_weight: Optional[float] = field(
        default=-0.1, metadata={"help": "sequence alignment weights"}
    )
    S_generated_S_weight: Optional[float] = field(
        default=-0.1, metadata={"help": "sequence alignment weights"}
    )
    S_edited_C_weight: Optional[float] = field(
        default=1.0, metadata={"help": "sequence alignment weights"}
    )
    S_edited_I_weight: Optional[float] = field(
        default=1.0, metadata={"help": "sequence alignment weights"}
    )
    S_edited_S_weight: Optional[float] = field(
        default=1.0, metadata={"help": "sequence alignment weights"}
    )

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "the model name"}
    )
    learning_rate: Optional[float] = field(
        default=1e-3, metadata={"help": "optimizer learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    max_prompt_length: Optional[int] = field(
        default=128, metadata={"help": "max length of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"}
    )
    max_steps: Optional[int] = field(
        default=1000, metadata={"help": "max number of training steps"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    evaluation_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "the evaluation strategy, None, epoch, or steps"},
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "Number of update steps between two evaluations if evaluation_strategy=steps"
        },
    )
    eval_first_step: Optional[bool] = field(
        default=False, metadata={"help": "Wether to eval first step"}
    )
    logging_strategy: Optional[str] = field(
        default=None, metadata={"help": "the logging strategy, None, epoch, or steps"}
    )
    log_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "Number of update steps between two logging if logging_strategy=steps"
        },
    )
    logging_first_step: Optional[bool] = field(
        default=False, metadata={"help": "Wether to log first step"}
    )
    save_strategy: Optional[str] = field(
        default=None, metadata={"help": "the saving strategy, None, epoch, or steps"}
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "Number of update steps between two saving if save_strategy=steps"
        },
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    alignment_function: Optional[str] = field(
        default="dpo", metadata={"help": "alignment function will be used"}
    )
    output_dir: Optional[str] = field(
        default="./test", metadata={"help": "output path"}
    )
    run_name: Optional[str] = field(
        default="test",
        metadata={
            "help": "A descriptor for the run. Typically used for wandb and mlflow logging."
        },
    )
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": "If a value is passed, will limit the total amount of checkpoints."
        },
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models."
        },
    )
    # instrumentation
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def trainer(script_args, train_dataset, eval_dataset):
    with open("hg_secret", "r") as f:
        hg_auth_token = f.read()

    # 1. load a pretrained model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # This means: fit the entire model on the GPU:0
        device_map = {"": 0}
    else:
        device_map = None
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_auth_token=hg_auth_token,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_auth_token=hg_auth_token,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, use_auth_token=hg_auth_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        logging_strategy=script_args.logging_strategy,
        logging_steps=script_args.log_steps,
        logging_first_step=script_args.logging_first_step,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        run_name=script_args.run_name,
        save_total_limit=script_args.save_total_limit,
        load_best_model_at_end=script_args.load_best_model_at_end,
        metric_for_best_model=script_args.metric_for_best_model,
    )

    # 5. initialize the DPO trainer

    if script_args.use_peft:
        lora_config = LoraConfig(
            r=256,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None

    if script_args.alignment_function == "dpo":
        trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            beta=script_args.beta,
            alpha1=script_args.alpha1,
            alpha2=script_args.alpha2,
            output_dir=script_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            max_prompt_length=script_args.max_prompt_length,
            peft_config=lora_config,
        )
    elif script_args.alignment_function == "salt":
        trainer = SALTTrainer(
            model=model,
            args=training_args,
            omega1=script_args.omega1,
            omega2=script_args.omega2,
            S_generated_C_weight=script_args.S_generated_C_weight,
            S_generated_D_weight=script_args.S_generated_D_weight,
            S_generated_S_weight=script_args.S_generated_S_weight,
            S_edited_C_weight=script_args.S_edited_C_weight,
            S_edited_I_weight=script_args.S_edited_I_weight,
            S_edited_S_weight=script_args.S_edited_S_weight,
            output_dir=script_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            max_prompt_length=script_args.max_prompt_length,
            peft_config=lora_config,
        )
    elif script_args.alignment_function == "sft":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            output_dir=script_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            max_prompt_length=script_args.max_prompt_length,
            peft_config=lora_config,
        )
    else:
        raise ValueError("Invalid alignment function")

    # if script_args.eval_first_step or 1:
    #     print('evaluating')
    #     print(trainer.evaluate())

    # 6. train
    trainer.train()

    trainer.evaluate()
