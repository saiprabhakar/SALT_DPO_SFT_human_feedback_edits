from transformers import HfArgumentParser
from trainer import ScriptArguments, load_dataset, trainer
import wandb

parser = HfArgumentParser(ScriptArguments)

# for DPO
script_args = parser.parse_args_into_dataclasses(
    args=[
         '--per_device_train_batch_size', '1',
         '--per_device_eval_batch_size', '1',
         '--gradient_accumulation_steps', '1',
         '--model_name_or_path', 'sshleifer/tiny-gpt2',
         # '--model_name_or_path', 'results/avs/BASE_model/gpt2',
         # '--model_name_or_path', 'gpt2',
         # '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
         '--load_in_4bit',
         '--use_peft',
         # '--learning_rate', '1e-3',
         '--learning_rate', '1e-4',
         # '--report_to', 'wandb',
         '--run_name', 'DPO-avs-gpt2',
         '--max_length', '1024',
         '--max_prompt_length', '768',
         '--num_train_epochs', '5',
         '--max_steps', '-1',
         '--evaluation_strategy', 'epoch',
         '--eval_steps', '-1',
         # '--eval_first_step',
         '--logging_strategy', 'steps',
         '--log_steps', '20',
         '--logging_first_step',
         # '--save_strategy', 'epoch',
         '--save_strategy', 'steps',
         '--save_steps', '10000000',
         # '--save_total_limit', '3',
         # '--load_best_model_at_end',
         # '--metric_for_best_model', 'metrics_policy_rouge1',
         #  '--alignment_function', 'dpo',
         '--output_dir', './results/avs/DPO_model/DPO-avs-gpt2(1|1|0.3)',
         '--alpha1', '1.0', #sft loss
         '--alpha2', '1.0', #dpo loss
         '--beta', '0.3',
         ])[0]

# Initialize wandb if reporting to wandb
if script_args.report_to == 'wandb':
    wandb.init(project=script_args.run_name)

# 2. Load training dataset
# train_dataset = load_dataset("train", sanity_check=script_args.sanity_check)
train_dataset = load_dataset("sub_eval", sanity_check=script_args.sanity_check, alignment_function=script_args.alignment_function)

# 3. Load evaluation dataset
eval_dataset = load_dataset("sub_eval", sanity_check=script_args.sanity_check, alignment_function=script_args.alignment_function)

dpo_trainer = trainer(script_args, train_dataset, eval_dataset)