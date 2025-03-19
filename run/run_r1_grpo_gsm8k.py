import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os

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
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
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
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


########################
# Helper functions
########################


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.

    Args:
       text (str): The model-generated text containing XML-style <answer> tags.

    Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

    Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
       return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
       return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_last_number(text):
    """
    Extracts the last number appearing in the text.

    Args:
       text (str): The text to extract a number from.

    Returns:
       float or None: The last number in the text, or None if no number is found.

    Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
    """
    text = text.replace('$', '').replace('%', '')
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    """
    Extracts a single number from text if exactly one number is present.

    Args:
       text (str): The text to extract a number from.

    Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

    Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
    """
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None


def correctness_reward_func(completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
       completions (list): List of model completions, each containing content.
       answer (list): List of expected answers.
       **kwargs: Additional keyword arguments.

    Returns:
       list: List of numerical rewards for each completion.

    Explanation:
       1. Extracts the content from each completion.
       2. Extracts the answer portion from each response using extract_answer_from_model_output.
       3. Assigns rewards based on matching criteria:
          - 2.0 points for an exact match
          - 1.5 points for numeric equivalence (when values match but format differs)
          - 0.0 points for incorrect answers
       4. Tracks completion lengths for analysis.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)

    return rewards


def format_reward_func(completions, answer, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.

    Args:
       completions (list): List of model completions, each containing content.
       **kwargs: Additional keyword arguments.

    Returns:
       list: List of format compliance scores for each completion.

    Explanation:
       1. Extracts the content from each completion.
       2. Evaluates format compliance by checking for required XML tags:
          - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
          - Maximum score of 0.8 for perfect format compliance
       3. Stores and returns the format compliance scores.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)

    return rewards


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
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, 'main', split=script_args.dataset_splits)
    # dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42)

    #####################
    # Prepare and format dataset
    #####################

    def build_prompt(messages):
        """
        Build a single prompt string from a list of messages.

        Args:
           messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

        Returns:
           str: A concatenated string of all message contents.

        Explanation:
           1. Takes a list of message dictionaries in the typical chat format.
           2. Extracts the 'content' field from each message and strips whitespace.
           3. Joins all content strings with newlines to create a single prompt.
           4. This preserves the training format while converting from structured messages to a string.
        """
        return "\n".join([msg["content"].strip() for msg in messages])

    def extract_answer_from_dataset(text):
        """
        Extracts the answer from the GSM8K dataset examples.

        Args:
           text (str): The dataset example text containing a question and answer.

        Returns:
           str or None: The extracted answer part after the '####' delimiter, or None if not found.

        Explanation:
           1. Checks if the text contains the '####' delimiter that separates question from answer.
           2. If found, splits the text at this delimiter and returns the second part (the answer).
           3. The answer is stripped of leading/trailing whitespace.
           4. Returns None if no delimiter is present.
        """
        if "####" not in text:
           return None
        return text.split("####")[1].strip()

    # generate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(example):
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
           "prompt": prompt_str,  # Now a string rather than a list.
           "answer": extract_answer_from_dataset(example["answer"])
        }
        return formatted_example

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
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

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()