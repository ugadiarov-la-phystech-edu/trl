from dataclasses import dataclass
from datetime import datetime
import logging
import os
import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = (
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])

from run.multiplication4x4_utils import reward_step_answer, reward_step_expression, reward_step_tag, \
    reward_answer_number, reward_answer_tag, reward_response_length, reward_step_answer_sparse, \
    reward_step_expression_sparse, reward_step_tag_sparse, reward_answer_number_sparse, reward_answer_tag_sparse, \
    reward_response_length_sparse

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "elfray/multiplication_3x2"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


SYSTEM_PROMPT = """
Step by step, multiply the numbers. For each place value of the multiplier, provide the partial products, the partial sums, and the final result in your response. For example, for the multiplication 813 * 45, the expected response is:
<step>813 * 5 = 4065</step>
<step>813 * 40 = 32520</step>
<step>4065 + 32520 = 36585</step>
<answer>36585</answer>
"""


########################
# Helper functions
########################


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_size='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'pad_token_id={tokenizer.pad_token_id} eos_token_id={tokenizer.eos_token_id}')

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(script_args.dataset_id_or_path, None, split="train")
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = load_dataset(script_args.dataset_id_or_path, None, split="valid")

    #####################
    # Prepare and format dataset
    #####################

    # generate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(example):
        multiplier, multiplicand = example['task'][::-1].replace(' ', '').split('*')
        answer = example['labels'][::-1].replace(' ', '')
        multiplier_num = int(multiplier)
        multiplicand_num = int(multiplicand)
        answer_num = int(answer)
        assert multiplicand_num * multiplier_num == answer_num

        gt_step_strings = []
        step_answers_num = []
        for place, digit in enumerate(multiplier[::-1]):
            place_value_num = int(f'{digit}{"0" * place}')
            step_mult_answer_num = int(multiplicand) * place_value_num
            step_answers_num.append(step_mult_answer_num)
            gt_step_strings.append(f'{multiplicand} * {place_value_num} = {step_mult_answer_num}')
            if place > 0:
                step_sum_answer_num = step_answers_num[-2] + step_mult_answer_num
                gt_step_strings.append(f'{step_answers_num[-2]} + {step_mult_answer_num} = {step_sum_answer_num}')
                step_answers_num.append(step_sum_answer_num)

        gt_steps = '\n'.join(gt_step_strings)
        return {'prompt': f'{SYSTEM_PROMPT.strip()}\nTask: {multiplicand} * {multiplier}',
                'answer': f'{gt_steps}\n{answer_num}'}

    # convert our dataset to the r1 prompt
    train_dataset = train_dataset.map(lambda x: generate_r1_prompt(x))
    test_dataset = test_dataset.map(lambda x: generate_r1_prompt(x))

    #########################
    # Instantiate DPO trainer
    #########################

    reward_funcs = [reward_step_answer, reward_step_expression, reward_step_tag, reward_answer_number,
                      reward_answer_tag, reward_response_length, ]
    if not training_args.dense_reward:
        reward_funcs = [reward_step_answer_sparse, reward_step_expression_sparse, reward_step_tag_sparse,
                        reward_answer_number_sparse, reward_answer_tag_sparse, reward_response_length_sparse, ]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # # Save everything else on main process
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card({"tags": ["rl", "grpo", "tutorial", "philschmid"]})
    # # push to hub if needed
    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()