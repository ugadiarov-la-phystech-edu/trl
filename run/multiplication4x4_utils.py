import re
from typing import List


WHITESPACE = re.compile('\\s')
INTEGER = re.compile('(\\d+)')


def reward_step_answer(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    step_answers = []
    for a in answer:
        step_answers.append([step_answer.split('=')[1].strip() for step_answer in a.strip().split('\n')[:-1]])

    completion_answers = [[False] * len(step_answer) for step_answer in step_answers]
    rewards = []
    for step_completion in completions:
        assert len(step_completion) == len(step_answers), f'{len(step_completion)} != {len(step_answers)}'
        assert len(step_completion) == len(completion_answers), f'{len(step_completion)} != {len(completion_answers)}'
        step_rewards = []
        for completion, step_answer, completion_answer in zip(step_completion, step_answers, completion_answers):
            assert len(step_answer) == len(completion_answer), f'{len(step_answer)} != {len(completion_answer)}'
            reward = 0
            for i, sa in enumerate(step_answer):
                if not completion_answer[i]:
                    has_step_answer = False
                    index = completion.find(sa)
                    if index > -1:
                        match = INTEGER.match(completion[index:])
                        has_step_answer = match is not None and match.group(1) == sa

                    completion_answer[i] = has_step_answer
                    reward += float(has_step_answer)

            step_rewards.append(reward)

        assert len(step_rewards) == n_generations, f'{len(step_rewards)} != {n_generations}'
        rewards.append(step_rewards)

    return rewards


def reward_step_expression(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    step_answers = []
    for a in answer:
        step_answers.append([re.sub(WHITESPACE, '', step_answer) for step_answer in a.strip().split('\n')[:-1]])

    completion_answers = [[False] * len(step_answer) for step_answer in step_answers]
    rewards = []
    for step_completion in completions:
        assert len(step_completion) == len(step_answers), f'{len(step_completion)} != {len(step_answers)}'
        assert len(step_completion) == len(completion_answers), f'{len(step_completion)} != {len(completion_answers)}'
        step_rewards = []
        for completion, step_answer, completion_answer in zip(step_completion, step_answers, completion_answers):
            assert len(step_answer) == len(completion_answer), f'{len(step_answer)} != {len(completion_answer)}'
            reward = 0
            for i, sa in enumerate(step_answer):
                if not completion_answer[i]:
                    has_step_answer = sa in re.sub(WHITESPACE, '', completion)
                    completion_answer[i] = has_step_answer
                    reward += float(has_step_answer)

            step_rewards.append(reward)

        assert len(step_rewards) == n_generations, f'{len(step_rewards)} != {n_generations}'
        rewards.append(step_rewards)

    return rewards


def reward_step_tag(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    step_answers = []
    for a in answer:
        step_answers.append([f'<step>{re.sub(WHITESPACE, "", step_answer)}</step>' for step_answer in a.strip().split('\n')[:-1]])

    completion_answers = [[False] * len(step_answer) for step_answer in step_answers]
    rewards = []
    for step_completion in completions:
        assert len(step_completion) == len(step_answers), f'{len(step_completion)} != {len(step_answers)}'
        assert len(step_completion) == len(completion_answers), f'{len(step_completion)} != {len(completion_answers)}'
        step_rewards = []
        for completion, step_answer, completion_answer in zip(step_completion, step_answers, completion_answers):
            assert len(step_answer) == len(completion_answer), f'{len(step_answer)} != {len(completion_answer)}'
            reward = 0
            for i, sa in enumerate(step_answer):
                if not completion_answer[i]:
                    has_step_answer = sa in re.sub(WHITESPACE, '', completion)
                    completion_answer[i] = has_step_answer
                    reward += float(has_step_answer)

            step_rewards.append(reward)

        assert len(step_rewards) == n_generations, f'{len(step_rewards)} != {n_generations}'
        rewards.append(step_rewards)

    return rewards


def reward_answer_number(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    step_answers = []
    for a in answer:
        step_answers.append([step_answer.strip() for step_answer in a.strip().split('\n')[-1:]])

    completion_answers = [[False] * len(step_answer) for step_answer in step_answers]
    rewards = []
    for step_completion in completions:
        assert len(step_completion) == len(step_answers), f'{len(step_completion)} != {len(step_answers)}'
        assert len(step_completion) == len(completion_answers), f'{len(step_completion)} != {len(completion_answers)}'
        step_rewards = []
        for completion, step_answer, completion_answer in zip(step_completion, step_answers, completion_answers):
            assert len(step_answer) == len(completion_answer), f'{len(step_answer)} != {len(completion_answer)}'
            reward = 0.
            for i, sa in enumerate(step_answer):
                if not completion_answer[i]:
                    has_step_answer = False
                    index = completion.find(sa)
                    if index > -1:
                        match = INTEGER.match(completion[index:])
                        has_step_answer = match is not None and match.group(1) == sa

                    completion_answer[i] = has_step_answer
                    reward += float(has_step_answer)

            step_rewards.append(reward)

        assert len(step_rewards) == n_generations, f'{len(step_rewards)} != {n_generations}'
        rewards.append(step_rewards)

    return rewards


def reward_answer_tag(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    step_answers = []
    for a in answer:
        step_answers.append([f'<answer>{re.sub(WHITESPACE, "", step_answer)}</answer>' for step_answer in a.strip().split('\n')[-1:]])

    completion_answers = [[False] * len(step_answer) for step_answer in step_answers]
    rewards = []
    for step_completion in completions:
        assert len(step_completion) == len(step_answers), f'{len(step_completion)} != {len(step_answers)}'
        assert len(step_completion) == len(completion_answers), f'{len(step_completion)} != {len(completion_answers)}'
        step_rewards = []
        for completion, step_answer, completion_answer in zip(step_completion, step_answers, completion_answers):
            assert len(step_answer) == len(completion_answer), f'{len(step_answer)} != {len(completion_answer)}'
            reward = 0.
            for i, sa in enumerate(step_answer):
                if not completion_answer[i]:
                    has_step_answer = sa in re.sub(WHITESPACE, '', completion)
                    completion_answer[i] = has_step_answer
                    reward += float(has_step_answer)

            step_rewards.append(reward)

        assert len(step_rewards) == n_generations, f'{len(step_rewards)} != {n_generations}'
        rewards.append(step_rewards)

    return rewards


def reward_response_length(completions: List[List[str]], answer, **kwargs):
    n_generations = len(completions[0])
    assert len(answer) == n_generations

    rewards = [[0] * n_generations for _ in range(len(completions))]
    for i, (a, completion) in enumerate(zip(answer, completions[-1])):
        n_completion_lines = len(completion.split('\n'))
        n_answer_lines = len(a.split('\n'))
        rew = n_completion_lines / n_answer_lines
        if rew > 1:
            rew = 1 / rew

        rewards[-1][i] = rew

    return rewards


if __name__ == '__main__':
    completions = [
        ['a', 'b', 'c'], ['a ', 'b ', 'c '], ['a *', 'b +', 'c -'], ['a * ', 'b + ', 'c - '], ['a * 1', 'b + 2', 'c - 3'],
        ['a * 1 ', 'b + 2 ', 'c - 3 '], ['a * 1 =', 'b + 2 =', 'c - 3 ='], ['a * 1 = ', 'b + 2 = ', 'c - 3 = '],
        ['a * 1 = x', 'b + 2 = y', '   <step>\nc - 3   = \tz   </step>'], ['a * 1 = x\n<answer> 100 </answer>', 'b + 2 = y \n 200', '   <step>\nc - 3   = \tz   </step>333'],
    ]
    answer = ['a * 1 = u\n100', 'b + 0 = y\n200', 'c - 3 = z\n300']

    print(reward_step_answer(completions, answer))
    print(reward_step_expression(completions, answer))
    print(reward_step_tag(completions, answer))
    print(reward_answer_number(completions, answer))
    print(reward_answer_tag(completions, answer))
    print(reward_response_length(completions, answer))
