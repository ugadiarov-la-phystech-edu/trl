import re
from typing import List, Collection


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


def contains_any(string: str, substrings: Collection[str]):
    for s in substrings:
        if s in string:
            return True

    return False


def strict_reward_func(completions: List[str], answer, **kwargs):
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
    responses = completions
    rewards = []
    r = 1 / 12
    tags = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
    for response, a in zip(responses, answer):
        score = 0.0
        reasoning_open_index = response.find("<reasoning>")
        if reasoning_open_index > -1:
            score += r

            if response[:reasoning_open_index].strip() == "":
                score += r

        reasoning_close_index = response.rfind("</reasoning>")
        if reasoning_close_index > -1:
            score += r

        if reasoning_open_index > -1 and reasoning_close_index > -1:
            if reasoning_open_index < reasoning_close_index:
                score += r

            if not contains_any(response[reasoning_open_index + len("<reasoning>"):reasoning_close_index], tags):
                score += r

        answer_open_index = response.find("<answer>")
        if answer_open_index > -1:
            score += r

        answer_close_index = response.rfind("</answer>")
        if answer_close_index > -1:
            score += r

            if response[answer_close_index + len("</answer>"):].strip() == "":
                score += r

        if answer_open_index > -1 and answer_close_index > -1:
            if answer_open_index < answer_close_index:
                score += r

            answer_text = response[answer_open_index + len("<answer>"):answer_close_index].strip()
            if not contains_any(answer_text, tags):
                score += r

                answer_number = extract_single_number(answer_text)
                if answer_number is not None and answer_number == float(a.replace(',', "")):
                    score += 1

        if reasoning_close_index > -1 and answer_open_index > -1:
            if reasoning_close_index < answer_open_index:
                score += r

                if response[reasoning_close_index + len("</reasoning>"):answer_open_index].strip() == "":
                    score += r

        rewards.append(score)

    return rewards


if __name__ == '__main__':
    example = """
    
    
    
        <reasoning>
        
        
        Some reasoning
        
        
        </reasoning>
        
        
        
        <answer>
        
        
        Some answer
        
        
        </answer>
        
        
        
        
        """
    example2 = """
        <reasoning>
        Some reasoning
        <reasoning>
        </answer>
        </reasoning>
        <answer>
        Some answer
        </answer>
        """
