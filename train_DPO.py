from transformers import HfArgumentParser
import wandb
from trainer.trainer import ScriptArguments, load_dataset_hg_local, trainer

parser = HfArgumentParser(ScriptArguments)

script_args = parser.parse_args_into_dataclasses(
    args=[
            '--per_device_train_batch_size', '2',
            '--per_device_eval_batch_size', '2',
            '--gradient_accumulation_steps', '4',
            '--model_name_or_path', 'gpt2',
            # '--model_name_or_path', 'sshleifer/tiny-gpt2',
            # '--model_name_or_path', 'huggy llama/llama-7b',
            # '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
            '--load_in_4bit',
            '--use_peft',
            '--learning_rate', '1e-4',
            # '--report_to', 'wandb',
            '--run_name', 'DPO-avs-gpt2',
            '--max_length', '1024',
            '--max_prompt_length', '768',
            '--num_train_epochs', '1',
            '--max_steps', '-1',
            '--evaluation_strategy', 'epoch',
            '--eval_steps', '-1',
            '--logging_strategy', 'steps',
            '--log_steps', '10',
            '--logging_first_step',
            '--save_strategy', 'epoch',
            '--save_steps', '-1',
            '--save_total_limit', '3',
            '--load_best_model_at_end',
            '--metric_for_best_model', 'metrics_policy_rouge1',
            '--output_dir', './results/avs/DPO_model/DPO-avs-gpt2(1|1|0.3)',
            "--alpha1", "1.0",  # sft loss
            "--alpha2", "1.0",  # dpo loss
            "--beta", "0.3",
        ]
    )[0]

# Initialize wandb if reporting to wandb
if script_args.report_to == "wandb":
    wandb.init(project=script_args.run_name)

data_subset = "sub_eval_w_simulated_edits"
train_dataset = load_dataset_hg_local(
    data_subset,
    sanity_check=script_args.sanity_check,
    alignment_function=script_args.alignment_function,
)

# 3. Load evaluation dataset
eval_dataset = load_dataset_hg_local(
    data_subset,
    sanity_check=True,
    alignment_function=script_args.alignment_function,
)

dpo_trainer = trainer(script_args, train_dataset, eval_dataset)