import os
from copy import copy
import re
from utils import decimal_to_binary, llama, mixtral
import pickle
import torch
from tqdm import tqdm
from draw import draw_distribution, draw_agent, draw_turn, draw_three_strategy, draw_10_turn, draw_10_agent, draw_radar, draw_10_agent_consistent_line, draw_word
import numpy as np
import pandas as pd
import scipy.stats as stats

import fire

def majority_ans(answer:list, return_answer_count:bool=True):
    assert len(answer) > 1
    if len(answer) == 2:
        if answer[0] == answer[1]:
            if return_answer_count:
                return 1
            return [answer[0]]
        else:
            if return_answer_count:
                return 2
            return []
    else:
        mapping = {}
        for ans in answer:
            if ans not in mapping:
                mapping[ans] = 1
            else:
                mapping[ans] += 1
        if return_answer_count:
            assert len(mapping) <= 10
            return len(mapping)

def parse_answer(dataset:str, content:str, task_info:tuple=None):
    def extract_math(string):
        results = []
        stack = []
        flag = r"\boxed{"
        string_copy = copy(string)
        idx = string_copy.find(flag)
        if idx == -1:
            # print(f"math parse failed: \"{string}\"")
            return []
            # assert False, f"math parse failed: \"{string}\""
        output_flag = 0
        idx += len(flag)
        while idx < len(string):
            if string[idx] == '{':
                output_flag += 1
            elif string[idx] == '}' and output_flag == 0:
                results.append("".join(stack))
                stack.clear()
                idx = string_copy.find(flag, idx)
                if idx == -1:
                    break
                else:
                    idx += len(flag)
                    continue
            elif string[idx] == '}' and output_flag != 0:
                output_flag -= 1
            stack.append(string[idx])
            idx += 1
        return results

    if dataset == "mmlu":
        assert len(task_info) == 5
        if "Combined Answer".lower() in content.lower():
            return None
        pattern = r"\((\w+)\)|(\w+)\)"
        matches = re.findall(pattern, content)
        matches = [match[0] or match[1] for match in matches]
        solution_by_re = None
        # assert len(matches)<=1, str(len(matches))
        for match_str in matches[::-1]:
            solution_by_re = match_str.upper()
            if solution_by_re >= 'A' and solution_by_re <= 'D':
                break
            else:
                solution_by_re = None
        solution_by_item = [-1,-1,-1,-1]
        idx = 0
        for item in task_info[1:]:
            pos = content.lower().rfind(item.lower().strip())
            if pos >= 0:
                solution_by_item[idx] = pos
            idx += 1
        if max(solution_by_item) == -1:
            solution_by_item = None
        else:
            solution_by_item = ["A","B","C","D"][
                solution_by_item.index(max(solution_by_item))
            ]
        if solution_by_item is None and solution_by_re is not None:
            return solution_by_re
        elif solution_by_item is not None and solution_by_re is None:
            return solution_by_item
        elif solution_by_item is None and solution_by_re is None:
            return None
        elif solution_by_item is not None and solution_by_re is not None:
            if solution_by_item == solution_by_re:
                return solution_by_item
            else:
                return solution_by_item
    elif dataset == "math":
        # pattern = r"\\boxed{([^{}]+|(?R))*}"
        # matches = re.findall(pattern, content)
        matches = extract_math(string=content)
        if len(matches)==0:
            return None
            # assert False, f"math parse failed: \"{content}\""
        else:
            return matches[-1]
    elif dataset == "chess":
        none_responese = [
            "i am unable to provide a valid destination square based on the given chess game and moves",
            "none",
            "no valid",
            "no specific valid",
            "invalid",
            "n/a",
            "unable to provide",
            "game sequence contains errors",
            "i cannot provide"
        ]
        content = content.lower()
        pattern = r"[a-h][1-8]"
        pos = content.rfind("final answer")
        if pos != -1:
            """there exist final answer"""
            item = content.split("final answer")[-1].strip()
            matches = re.findall(pattern, item)
            if len(matches) == 1:
                return matches[0].lower()
            elif len(matches) > 1:
                """set the last one to answer"""
                # print([content])
                # print("*"*100)
                return matches[-1].lower()
            else:
                for valid_case in none_responese:
                    if valid_case in content:
                        return None
                return None
                assert False, f"chess parse failed 1: \"{content}\""
        else:
            matches = re.findall(pattern, content)
            if len(matches) == 0:
                for valid_case in none_responese:
                    if valid_case in content:
                        return None
                return None
                # assert False, f"chess parse failed 2: \"{content}\""
            else:
                return matches[-1]
    else:
        assert False

def load_dataset(args=None, dataset=None):
    # file_name = f"./results/{args.experiment_type}/{args.dataset}_data.pkl"
    if args is not None:
        file_name = f"../eval_data/{args.dataset}.pkl"
    elif dataset is not None:
        file_name = f'../eval_data/{dataset}.pkl'
    else:
        assert False
    database = pickle.load(open(file_name, "rb"))
    return database

def _most_frequence_answer(answers:list):
    # print(List)
    counter = 0
    if answers is None:
        return None
    num = [answers[0]]

    for i in answers:
        current_frequency = answers.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = [i]
        elif current_frequency == counter:
            num.append(i)

    num = list(set(num))
    if counter == 1:
        return None
    elif len(num) != 1:
        return None
    else:
        return num[0]

def _judge_answer(args, gt_answer, agent_final_answer, agent_answer, math_pattern):
    def check_math_answer(answer):
        if answer == '{2^3 \\times 3^2 \\times 5 \\times 7}':
            return '2520'
        for _ in math_pattern:
            answer = answer.replace(_, "")
        # answer = answer.replace("^\\circ", "").replace(" \\, \\text{cm}","").replace("{\\text{Degree of }} g(x) = ","")
        answer = answer.replace("{", "").replace("}", "")
        if answer in ["978,121", "99,940,009", "979,121"]:
            return answer.replace(",", "")
        return answer

    if agent_final_answer is None:
        return 0
    if args.dataset == "chess":
        assert isinstance(gt_answer, list)
        if agent_final_answer in gt_answer:
            return 1
        else:
            return 0
    elif args.dataset == "mmlu":
        if agent_final_answer.lower() == gt_answer.lower():
            return 1
        else:
            return 0
    elif args.dataset == "math":
        if isinstance(gt_answer, tuple):
            gt_answer = gt_answer[0]
        agent_final_answer = check_math_answer(agent_final_answer)
        if agent_final_answer.lower() == gt_answer.lower():
            # print(agent_answer)
            # print(agent_final_answer, gt_answer, "True")
            return 1
        else:
            # print(agent_answer)
            # print(agent_final_answer, gt_answer, "False")
            return 0
    else:
        assert False

def judge_answer(agent_answer:list, gt_answer, args, weight=False, converge=False):
    math_pattern = [
        " \\text{ positive integers, } 12 \\text{ negative integers}", " \\, \\text{cm}", "\\, \\text{cm}",
        "^\\circ", "{\\text{Degree of }} g(x) = ", "°", " \\text{ cm}", "\\text{ cm}", "b = ", "r = ", "x = ", "m+n = ", "\\text{ degrees}", "x + y + z = "
    ]
    
    def check_include(answers: list):
        """应对['{12}', '12', '24']情况"""
        for i in range(len(answers)):
            if answers[i] is not None:
                for _ in math_pattern:
                    answers[i] = answers[i].replace(_, "")
        for i in range(len(answers)):
            for j in range(len(answers)):
                if i != j and answers[i] is not None and answers[j] is not None and answers[i] in answers[j] and answers[i] != answers[j]:
                    # print("mike-check-before:", answers)
                    if "{" + answers[i] + "}" == answers[j]:
                        answers[i] = answers[j]
        # print("mike-check-now:", answers)
        return answers

    # if args.dataset == "math":
    #     # _agent_answer = 
    #     agent_final_answer = _most_frequence_answer(answers=[check_math_answer(_) for _ in agent_answer])
    #     print(agent_final_answer)
    # else:
    if args.dataset == "math":
        agent_answer = check_include(agent_answer)
    if not weight and not converge:
        agent_final_answer = _most_frequence_answer(answers=agent_answer)
        # if args.dataset == 'chess':

            # for i in agent_answer:
            #     if i is None:
            #         return 0
            #     if i.lower() not in gt_answer:
            #         return 0
            # return 1

            # cnt = 0
            # for i in agent_answer:
            #     if i is None:
            #         continue
            #     if i.lower() in gt_answer:
            #         cnt += 1
            #     # if i.lower() not in gt_answer:
            #     #     return 0
            # if cnt >= 2:
            #     return 1
            # else:
            #     return 0
        return _judge_answer(args, gt_answer, agent_final_answer, agent_answer, math_pattern)
    elif weight==True and not converge:
        value = 0
        for ans in agent_answer:
            value += _judge_answer(args, gt_answer, ans, agent_answer, math_pattern)
        return value / 3
    elif not weight and converge:
        value = 0

    # print(agent_answer)
    # print(agent_final_answer, gt_answer)
    # assert False
    
def _behaviour(memory):
    n_role, n_strategy, n_repeat, n_case, n_turn = memory.shape
    """wrong wrong wrong wrong=0000"""
    """correct correct correct correct=1111"""
    results = torch.zeros([n_role, n_strategy, 2**n_turn], dtype=torch.int)
    for role_id in range(n_role):
        print(f"==========={role_id} Haromony===========")
        for strategy_id in range(n_strategy):
            for repeat_id in range(n_repeat):
                for case_id in range(n_case):
                    value = ""
                    for turn_id in range(n_turn):
                        if memory[role_id, strategy_id, repeat_id, case_id, turn_id] >= 0:
                            value = f"{value}{memory[role_id, strategy_id, repeat_id, case_id, turn_id]}"
                        else:
                            break
                    if len(value) != n_turn:
                        continue
                    assert len(value) == n_turn, f"{value} != {n_turn}"
                    flag = int(value, 2)
                    results[role_id, strategy_id, flag] += 1
            # print(decimal_to_binary(strategy_id, n_turn-1), ":", results[role_id, strategy_id].tolist())
            # 0000, 0001, ..., 1111  1-correct 0-mistake
            # print(results[role_id, strategy_id].tolist())
    # prepare for draw()
    output_data = {
        '知错能改': [0 for i in range(8)],     # [n_dataset, n_strategy]   0111, 0011, 0001
        '对了也改': [0 for i in range(8)],     # 1000, 1100, 1110
        '漂浮不定': [0 for i in range(8)],     # 1010, 1001, 1010, 1001
        '起始是对': [0 for i in range(8)],     # 1*
        '结果是对': [0 for i in range(8)]      # *1
    }
    category = [
        ('知错能改', ('0111','0011','0001')),
        ('对了也改', ('1000','1100','1110')),
        ('漂浮不定', ('1010','1001','0110','0101')),
        ('起始是对', ('1000','1001','1010','1011','1100','1101','1110','1111')),
        ('结果是对', ('0001','0011','0101','0111','1001','1011','1101','1111'))
    ]
    results = results.sum(dim=0)    # [strategy_id, 2**4]
    for label, types in category:
        for strategies in range(8):
            for i in types:
                v = int(i, 2)
                output_data[label][strategies] += results[strategies, v].item()
    tokenizer = None
    dict2str = None

    collaboration_tokens, self_consistent_tokens = get_token_for_distribute(Args, tokenizer, dict2str)
    # print(self_consistent_tokens)
    # a = input('check')
    output_data['cost'] = collaboration_tokens.tolist()
    output_data['self-consistent cost'] = self_consistent_tokens.tolist()[0]
    # print(output_data)
    return output_data
         
def behaviour(args, only_value=False):
    """only_value主要是对group有用"""
    """对某个的行为进行分析，分析策略对行为是否会产生共性"""    
    """直接就对4轮的进行分析吧，因为4轮的前3轮就是3轮拿过来的，算了，还是让参数来决定吧，因为4轮的只做了一个社会的"""
    """
    主要是得到下面的这个矩阵，然后对其进行分析即可：
        [n_role, n_strategy, n_repeat, n_case, n_turn]   里面存的整体的结果是对不对     
        [n_role, n_strategy, n_repeat, n_case, n_turn]   里面存的是有几个agent答对了
    1. 按照repeat、role加载数据文件
    2. 拿到某个数据文件进行读取
    3. 
    """
    if isinstance(args.dataset, list):
        args.dataset = args.dataset[0]
    assert isinstance(args.dataset, str)
    n_role = 4 # if args.experiment_type == "main" else 1
    assert len(args.role) == n_role
    n_repeat = 5
    n_case = 50
    n_turn = 3 # if args.experiment_type == "main" else 4
    n_turn += 1
    n_strategy = 2 ** args.turn
    print(n_role, n_strategy, n_repeat, n_case, n_turn)
    # assert False
    mem_results = torch.zeros([n_role, n_strategy, n_repeat, n_case, n_turn], dtype=torch.int)
    mem_agent = torch.zeros([n_role, n_strategy, n_repeat, n_case, n_turn], dtype=torch.int)
    mem_results[:] = -1
    mem_agent[:] = -1
    results_value = []
    results_name = []
    database = load_dataset(args)
    file_dir = f"./results/{args.experiment_type}/{args.dataset}"
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2 ** args.turn)]
    # if args.dataset == "mmlu":
    #     repeat_dir_name = ['2','3','4','5','7']
    # elif args.dataset == "chess":
    #     repeat_dir_name = ['2','3','4','5','6']
    # elif args.dataset == "math":
    #     repeat_dir_name = ['1','2','3','4','5']
    repeat_dir_name = ['1','2','3','4','5']
    for repeat_id, repeat in enumerate(repeat_dir_name):
        args.repeat = int(repeat)
        for role_id, role in enumerate(args.role):
            for strategy_id, strategy in enumerate(strategies):
                current_acc = []
                for case_id in range(args.n_case):
                    origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                    origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                    now_full_path = origin_full_path
                    try:
                        agent_center = pickle.load(open(now_full_path, "rb"))
                    except:
                        print("shutdown:", now_full_path)
                        continue
                        assert False
                    agent_answers = []
                    gt_answers = database["answer"][case_id]
                    """0,1 2,3 [4,5 6,7 8,9]"""
                    for turn_id, which_turn in enumerate(range(3, len(agent_center[0]), 2)):
                        for agent_id in range(args.agent):
                            context = agent_center[agent_id][which_turn]
                            assert context["role"] == "assistant"
                            agent_answers.append(
                                parse_answer(dataset=args.dataset, content=context["content"], task_info=database["task_info"][case_id])
                            )
                            # print(context, agent_answers[-1], gt_answers)
                        # print([n_role, n_strategy, n_repeat, n_case, n_turn])
                        # print([role_id, strategy_id, repeat_id, case_id, turn_id])
                        mem_results[role_id, strategy_id, repeat_id, case_id, turn_id] = judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, args=args, weight=False)
                        """0/1/2/.../n_agent"""
                        mem_agent[role_id, strategy_id, repeat_id, case_id, turn_id] = judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, args=args, weight=True)*3
                        # current_acc += judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, args=args)
                        agent_answers = []
    if only_value:
        return mem_results
    else:
        args.repeat = [1,2,3,4,5]
        return _behaviour(mem_results)
    # _check_acc(mem_results)

class Args:
    which_turn = 9
    experiment_type = None 
    dataset = None
    repeat = [1,2,3,4,5]
    society = [0, 1, 2, 3]
    strategy = [decimal_to_binary(i, 3) for i in range(8)]
    n_case=50
    case_id = list(range(n_case))
    n_agent=3
    n_turn=3

    role = society
    turn = n_turn
    agent = n_agent

def wrap4latex(mean:list, var:list, output_var=True):
    def wrap(_mean:float, _var:float, color:str=None, bold:bool=True) -> str:
        if output_var:
            value = f'{round(_mean, 1)}±{round(_var, 1)}'
        else:
            value = f'{round(_mean, 1)}'
        if color is None:
            return value
        else:
            # return value
            if bold:
                return r'\colorbox{' + color + r'}{\textbf{' + value + r'}}'
            else:
                return r'\colorbox{' + color + r'}{' + value + r'}'

    def find_indices(mean_values, variance_values):
        # Combining mean and variance into a list of tuples
        combined_data = list(zip(mean_values, variance_values, list(range(len(mean_values)))))
        combined_data = sorted(combined_data, key=lambda x: (-x[0], x[1]))
        max_indices, second_max_indices, min_indices = [combined_data[0][-1]], [], [combined_data[-1][-1]]
        second_start_idx = None
        for i in range(1, len(mean_values)):
            if combined_data[i][0] == combined_data[0][0] and combined_data[i][1] == combined_data[0][1]:
                max_indices.append(combined_data[i][-1])
            else:
                second_max_indices.append(combined_data[i][-1])
                second_start_idx = i
                break
        for i in range(second_start_idx+1, len(mean_values)):
            if combined_data[i][0] == combined_data[second_start_idx][0] and combined_data[i][1] == combined_data[second_start_idx][1]:
                second_max_indices.append(combined_data[i][-1])
            else:
                break
        for i in range(len(mean_values)-2, -1, -1):
            if combined_data[i][0] == combined_data[-1][0] and combined_data[i][1] == combined_data[-1][1]:
                min_indices.append(combined_data[i][-1])
            else:
                break
        return max_indices, second_max_indices, min_indices
    
    color_mode = {
        'max': 'Mycolor1',
        'min': 'Mycolor2',
        'second': 'Mycolor3'
    }
    assert len(mean) == len(var)
    max_indices, second_max_indices, min_indices = find_indices(mean, var)
    outputs = ""
    for i in range(len(mean)):
        m, v = mean[i],var[i]
        if i in max_indices:
            color = color_mode['max']
        elif i in min_indices:
            color = color_mode['min']
        elif i in second_max_indices:
            color = color_mode['second']
        else:
            color = None
        outputs = f"{outputs} & {wrap(m, v, color, bold=not (color_mode['min']==color))}"
    return outputs

def get_token(args, tokenizer=None, dict2str=None):
    def load_data(args, repeat, role, strategy, case_id, using_token_file:bool):
        dataset = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
        if using_token_file:
            file_name =  f"./results/{args.experiment_type}/{dataset}/{repeat}/{role}_harmony_{args.n_agent}_agents_{args.n_turn}_turns_{strategy}_strategy_case_{case_id}_token.pkl"
            assert os.path.exists(file_name), f"{file_name}"
            consume = 0
            data = pickle.load(open(file_name, "rb"))

            for a in range(args.n_agent):
                consume += data[a][-1]['total_tokens']
            return consume
        file_name = f"./results/{args.experiment_type}/{dataset}/{repeat}/{role}_harmony_{args.n_agent}_agents_{args.n_turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
        assert os.path.exists(file_name), f"{file_name}"
        consume = 0
        data = pickle.load(open(file_name, "rb"))
        for a in range(args.n_agent):
            consume += len(tokenizer(dict2str(data[a]))['input_ids'])
            # consume += data[a][-1]["total_tokens"]
        return consume
    
    total_tokens = torch.zeros([len(args.repeat), len(args.society), len(args.strategy)])  # [n_repeat, n_role, n_strategy]
    pbar = tqdm(total = len(args.repeat)*len(args.society)*len(args.strategy)*len(args.case_id))
    for role_id, role in enumerate(args.society):
        for repeat_id, repeat in enumerate(args.repeat):
            for strategy_id, strategy in enumerate(args.strategy):
                for case_id in range(args.n_case):
                    pbar.update(1)
                    total_tokens[repeat_id, role_id, strategy_id] += load_data(args, repeat, role, strategy, case_id, using_token_file= tokenizer == None)
                    # total_tokens[-1].append(load_data(args, repeat, role, strategy, case_id))
                total_tokens[repeat_id, role_id, strategy_id] /= args.n_case
    pbar.close()
    # total_tokens: [n_rpeat, n_society, n_strategy]
    # n_strategy, n_society
    return total_tokens.mean(dim=[0,1]), total_tokens.mean(dim=[0,2])

def get_token_for_distribute(args, tokenizer=None, dict2str=None) -> list:
    def load_data(args, repeat, role, strategy, case_id, using_token_file:bool):
        dataset = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
        if using_token_file:
            file_name =  f"./results/{args.experiment_type}/{dataset}/{repeat}/{role}_harmony_{args.n_agent}_agents_{args.n_turn}_turns_{strategy}_strategy_case_{case_id}_token.pkl"
            assert os.path.exists(file_name), f"{file_name}"
            consume = [0 for _ in range(args.n_turn+2)]
            data = pickle.load(open(file_name, "rb"))
            for t in range(args.n_turn+2):
                for a in range(args.n_agent):
                    consume[t] += data[a][t]['total_tokens']
                    # consume += data[a][-1]['total_tokens']
            return consume
        file_name = f"./results/{args.experiment_type}/{dataset}/{repeat}/{role}_harmony_{args.n_agent}_agents_{args.n_turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
        assert os.path.exists(file_name), f"{file_name}"
        consume = [0 for _ in range(args.n_turn+2)]
        data = pickle.load(open(file_name, "rb"))
        for t in range(args.n_turn+2):
            for a in range(args.n_agent):
                assert data[a][0:t*2+2]['role'] == 'assistant'
                consume[t] += len(tokenizer(dict2str(data[a][0:t*2+2]))['input_ids'])
                # consume += len(tokenizer(dict2str(data[a]))['input_ids'])
            # consume += data[a][-1]["total_tokens"]
        return consume
    # print(args.repeat)
    # print(args.society)
    # print(args.strategy)
    # print(args.dataset)
    # a=input('a')
    total_tokens = torch.zeros([len(args.repeat), len(args.society), len(args.strategy), args.turn+2])  # [n_repeat, n_role, n_strategy, n_turn]
    pbar = tqdm(total = len(args.repeat)*len(args.society)*len(args.strategy)*len(args.case_id))
    for role_id, role in enumerate(args.society):
        for repeat_id, repeat in enumerate(args.repeat):
            for strategy_id, strategy in enumerate(args.strategy):
                for case_id in range(args.n_case):
                    pbar.update(1)
                    total_tokens[repeat_id, role_id, strategy_id] += torch.as_tensor(load_data(args, repeat, role, strategy, case_id, using_token_file= tokenizer == None))
                    # total_tokens[-1].append(load_data(args, repeat, role, strategy, case_id))
                total_tokens[repeat_id, role_id, strategy_id] /= args.n_case
    pbar.close()
    # total_tokens: [n_rpeat, n_society, n_strategy]
    # n_strategy, n_society
    return total_tokens[:,:,:,-1].mean(dim=[0,1]), total_tokens[:,:,:,1].mean(dim=[0,1])
    # return total_tokens.mean(dim=[0,1]), total_tokens.mean(dim=[0,2])

def win_lose(data):
    # data: [n_strategy, n_society, n_repeat]
    n_strategy, n_society, n_repeat = data.shape[0], data.shape[1], data.shape[2]
    winlose = torch.zeros([n_society, n_strategy])
    for soc in range(n_society):
        for stgy in range(1, n_strategy):
            cnt = 0
            for r in range(n_repeat):
                if data[stgy, soc, r] >= data[0, soc, r]:
                    cnt += 1
            winlose[soc, stgy] = cnt
    # [n_strategy], [n_society]
    return winlose.sum(dim=0), winlose.sum(dim=1)
            
def return_data(return_mean_and_var=True, return_origin=False, value:str='performance'):
    """
    return_mean_and_va, return_origin
    True, False/True -> merge the repeat and case dim: [n_strategy, n_society] for draw figure
    False, False -> keep the repeat dim [n_repeat, n_society, n_strategy] for siginificant test
    False, True -> return the meta data: [n_repeat, n_society, n_strategy, n_case] for drawng radar
    """
    assert value in ['performance','consistent']
    which_turn = Args.which_turn
    experiment_type = Args.experiment_type
    # dataset = ['math', 'mmlu', 'chess']
    dataset = Args.dataset
    repeat = Args.repeat
    society = Args.society
    strategy = Args.strategy
    case_id = Args.case_id
    n_agent = Args.n_agent
    n_turn = Args.n_turn

    # if 'llama' in experiment_type:
    #     tokenizer = LlamaTokenizer.from_pretrained('...')
    #     dict2str = llama().messages2str
    # elif 'mixtral' in experiment_type or 'mistral' in experiment_type:
    #     tokenizer = AutoTokenizer.from_pretrained('...')
    #     dict2str = mixtral().messages2str
    # else:
    tokenizer = None
    dict2str = None

    data = torch.zeros([len(dataset), len(repeat), len(society), len(strategy), len(case_id)])
    pbar = tqdm(total=data.numel())
    for idx_ds, ds in enumerate(dataset):
        database = load_dataset(dataset=ds)
        class args:
            dataset=ds
        for idx_rpt, rpt in enumerate(repeat):
            file_dir = f'./results/{experiment_type}/{ds}/{rpt}'
            for idx_ro, ro in enumerate(society):
                for idx_stgy, stgy in enumerate(strategy):
                    for idx_case, idx in enumerate(case_id):
                        # load file
                        file_name = f'{ro}_harmony_{n_agent}_agents_{n_turn}_turns_{stgy}_strategy_case_{idx}.pkl'
                        full_path = f'{file_dir}/{file_name}'
                        try:
                            agent_center = pickle.load(open(full_path, 'rb'))
                        except:
                            print(full_path)
                            assert False
                        # agent's answer
                        agent_answers = []
                        # ground truth 
                        gt_answers = database['answer'][idx]
                        for agent_id in range(n_agent):
                            context = agent_center[agent_id][which_turn]
                            assert context['role'] == 'assistant'
                            agent_answers.append(
                                parse_answer(dataset=ds, content=context['content'], task_info=database['task_info'][idx])
                            )
                        if value == 'performance':
                            data[idx_ds, idx_rpt, idx_ro, idx_stgy, idx_case] = \
                                judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, weight=False, args=args)
                        elif value == 'consistent':
                            data[idx_ds, idx_rpt, idx_ro, idx_stgy, idx_case] = majority_ans(agent_answers, return_answer_count=True)
                        # if idx_rpt == 0 and idx_case == 30 and idx_stgy ==0:
                            # print(agent_answers, gt_answers, data[idx_ds, idx_rpt, idx_ro, idx_stgy, idx_case])
                            # a=input('test')
                        pbar.update(1)
    pbar.close()
    # [n_dataset, n_repeat, n_society, n_strategy, n_case]
    if not return_mean_and_var:
        # return meta data for significance test
        # data: [len(dataset), len(repeat), len(society), len(strategy), len(case_id)]
        if return_origin == False:
            print('data for draw significance test')
            assert data.shape[0] == 1
            return data[0].sum(-1)  # [n_repeat, n_society, n_strategy]
        else:
            print('data for draw radar figure')
            assert data.shape[0] == 1
            return data[0]  # [n_repeat, n_society, n_strategy, n_case]
    means = []
    vars = []
    for idx_ds, ds in enumerate(dataset):
        print('='*20, ds, '='*20)
        cur_data = data[idx_ds].sum(-1) # [n_repeat, n_society, n_strategy]
        if value == 'performance':
            cur_data = cur_data / len(case_id) * 100    
        elif value == 'consistent':
            cur_data = cur_data / len(case_id)
        cur_data = cur_data.permute(2,1,0)   # [n_strategy, n_society, n_repeat]
        # winlose_strategy, winlose_society = win_lose(cur_data)
        # token_strategy, token_society = get_token(Args, tokenizer, dict2str)
        mean = cur_data.mean(-1)            # [n_strategy, n_society]
        var = cur_data.std(-1)              # [n_strategy, n_society]
        means.append(mean)
        vars.append(var)
    return means[0], vars[0]

def main_table(
    experiment_type: str,
    dataset: str=['mmlu','math','chess'][0],
    repeat: list=[1,2,3,4,5],
    society: list=[0,1,2,3],
    strategy: list=[decimal_to_binary(i, 3) for i in range(8)],
    n_agent: int=3,
    n_turn: int=3,
    which_turn: int=9,
):
    Args.which_turn = which_turn
    Args.experiment_type = experiment_type
    assert os.path.exists(f'results/{experiment_type}') and os.path.isdir(f'results/{experiment_type}'), \
        f"The folder `results/{experiment_type}` don't exist. Please make sure the `--experiment_type` is valid."
    # dataset = ['math', 'mmlu', 'chess']
    Args.dataset = [dataset]
    assert isinstance(dataset, str)
    dataset = Args.dataset
    for d in Args.dataset:
        assert os.path.exists(f'results/{experiment_type}/{d}') and os.path.isdir(f'results/{experiment_type}/{d}'),\
            f"The folder `results/{experiment_type}/{d}` don't exist. Please make sure the `--dataset` is valid."
    Args.repeat = repeat
    assert isinstance(repeat, list)
    for r in Args.repeat:
        assert os.path.exists(f'results/{experiment_type}/{d}/{r}') and os.path.isdir(f'results/{experiment_type}/{d}/{r}'),\
            f"The folder `results/{experiment_type}/{d}/{r}` don't exist. Please make sure the `--repeat` is valid."
    Args.society = society
    assert isinstance(Args.society, list)
    Args.strategy = strategy
    assert isinstance(Args.strategy, list)
    Args.n_agent = n_agent
    Args.n_turn = n_turn
    case_id = Args.case_id

    tokenizer = None
    dict2str = None

    data = torch.zeros([len(dataset), len(repeat), len(society), len(strategy), len(case_id)])
    pbar = tqdm(total=data.numel())
    for idx_ds, ds in enumerate(dataset):
        database = load_dataset(dataset=ds)
        class args:
            dataset=ds
        for idx_rpt, rpt in enumerate(repeat):
            file_dir = f'./results/{experiment_type}/{ds}/{rpt}'
            for idx_ro, ro in enumerate(society):
                for idx_stgy, stgy in enumerate(strategy):
                    for idx_case, idx in enumerate(case_id):
                        file_name = f'{ro}_harmony_{n_agent}_agents_{n_turn}_turns_{stgy}_strategy_case_{idx}.pkl'
                        full_path = f'{file_dir}/{file_name}'
                        try:
                            agent_center = pickle.load(open(full_path, 'rb'))
                        except:
                            assert False
                        agent_answers = []
                        gt_answers = database['answer'][idx]
                        for agent_id in range(n_agent):
                            context = agent_center[agent_id][which_turn]
                            assert context['role'] == 'assistant'
                            agent_answers.append(
                                parse_answer(dataset=ds, content=context['content'], task_info=database['task_info'][idx])
                            )
                        # print(agent_answers, gt_answers)
                        data[idx_ds, idx_rpt, idx_ro, idx_stgy, idx_case] = \
                            judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, weight=False, args=args)
                        pbar.update(1)
    pbar.close()
    # [n_dataset, n_repeat, n_society, n_strategy, n_case]
    for idx_ds, ds in enumerate(dataset):
        print('='*20, ds, '='*20)
        cur_data = data[idx_ds].sum(-1) # [n_repeat, n_society, n_strategy]
        cur_data = cur_data / len(case_id) * 100    
        cur_data = cur_data.permute(2,1,0)   # [n_strategy, n_society, n_repeat]
        winlose_strategy, winlose_society = win_lose(cur_data)
        token_strategy, token_society = get_token(Args, tokenizer, dict2str)
        mean = cur_data.mean(-1)            # [n_strategy, n_society]
        var = cur_data.std(-1)              # [n_strategy, n_society]
        # 输出
        for i in range(len(society)):
            mean_list = []
            var_list = []
            for j in range(len(strategy)):
                mean_list.append(round(mean[j,i].item(), 2))
                var_list.append(round(var[j,i].item(), 2))
            # print()
            print(wrap4latex(mean_list, var_list), f"& {int(token_society[i].item())} & {int(winlose_society[i].item())}")
        print(' & '.join([str(int(token_strategy[i])) for i in range(len(Args.strategy))]))
        print(' & '.join([str(int(winlose_strategy[i])) for i in range(1, len(Args.strategy))]))
        print()

def draw(
    types:str,
    experiment_type: str,
    dataset: str = None
):
    if types == '10-turn':
        assert dataset is not None
        assert dataset in ['chess','math','mmlu']
    Args.experiment_type = experiment_type
    if types == 'distribute':
        distribute_data = {
            '知错能改': [],     
            '对了也改': [],
            '漂浮不定': [],
            '起始是对': [],
            '结果是对': [],
            'cost': [],
            'self-consistent cost': []  
        }
        dataset = ['mmlu', 'math', 'chess']
        for ds in dataset:
            Args.dataset = [ds]
            data = behaviour(Args)
            for label in distribute_data:
                distribute_data[label].append(data[label])
        # print(distribute_data)
        # a=input('1')
        draw_distribution(distribute_data, f"final-graph/distribute/{Args.experiment_type}-distribute.pdf")
    elif types == 'agent':
        # 2-agent 3-agent 4-agent
        agent_data = {
            # 'data_mmlu': np.random.uniform(30,60, (8,3)).reshape(-1),
            # 'data_chess': np.random.uniform(30,60, (8,3)).reshape(-1),
            # 'errors_mmlu': np.random.uniform(5,10, (8,3)).reshape(-1),
            # 'errors_chess': np.random.uniform(5,10, (8,3)).reshape(-1),
            'data_mmlu': torch.zeros([8,3]),
            'data_chess': torch.zeros([8,3]),
            'errors_mmlu': torch.zeros([8,3]),
            'errors_chess': torch.zeros([8,3]),
        }
        dataset = ['mmlu', 'chess']
        # model = ['llama13-main', 'llama70-main'][0]
        assert experiment_type in ['llama13-main', 'llama70-main']
        model = experiment_type
        for ds in dataset:
            Args.dataset = [ds]
            for idx, n_agent in enumerate([2,3,4]):
                if n_agent == 3:
                    Args.experiment_type = model
                else:
                    Args.experiment_type = f"{model.split('-')[0]}-agent-{n_agent}"
                # Args.society = [1] if not (model == 'gpt-3.5' and n_agent==2) else [2]
                # Args.society = [2] if model == 'gpt-3.5' and n_agent==2 else [1]
                Args.society = [1]
                Args.n_agent = n_agent
                Args.agent = n_agent
                mean, var = return_data()   # [n_strategy, n_society]
                agent_data[f'data_{ds}'][:, idx] = mean.reshape(-1)
                agent_data[f'errors_{ds}'][:, idx] = var.reshape(-1)
        agent_data['data_chess'] = agent_data['data_chess'].reshape(-1)
        agent_data['data_mmlu'] = agent_data['data_mmlu'].reshape(-1)
        agent_data['errors_mmlu'] = agent_data['errors_mmlu'].reshape(-1)
        agent_data['errors_chess'] = agent_data['errors_chess'].reshape(-1)
        draw_agent(agent_data, f'final-graph/agent/{model}-agent.pdf')
    elif types == 'turn':
        turn_data = {
            # 'data_mmlu': np.random.uniform(30,60, (16,3)).reshape(-1).tolist(), # [[策略1-t2, 策略1-t3, 策略1-t4]]
            # 'data_chess': np.random.uniform(30,60, (16,3)).reshape(-1).tolist(),
            # 'errors_mmlu': np.random.uniform(5,10, (16,3)).reshape(-1).tolist(),
            # 'errors_chess': np.random.uniform(5,10, (16,3)).reshape(-1).tolist(),
            'data_mmlu': torch.zeros([16,3]),
            'data_chess': torch.zeros([16,3]),
            'errors_mmlu': torch.zeros([16,3]),
            'errors_chess': torch.zeros([16,3]),
        }
        dataset = ['mmlu', 'chess']
        assert experiment_type in ['llama13-main', 'llama70-main']
        model = experiment_type
        for ds in dataset:
            Args.dataset = [ds]
            Args.experiment_type = f"{model.split('-')[0]}-turn-4"
            for idx, which_turn in enumerate([7,9,11]):
                Args.society = [1]
                Args.strategy = [decimal_to_binary(i, 4) for i in range(16)]
                Args.n_turn = 4
                Args.turn = 4
                Args.which_turn = which_turn
                mean, var = return_data()   # [n_strategy, n_society]
                turn_data[f'data_{ds}'][:, idx] = mean.reshape(-1)
                turn_data[f'errors_{ds}'][:, idx] = var.reshape(-1)
        turn_data['data_chess'] = turn_data['data_chess'].reshape(-1)
        turn_data['data_mmlu'] = turn_data['data_mmlu'].reshape(-1)
        turn_data['errors_mmlu'] = turn_data['errors_mmlu'].reshape(-1)
        turn_data['errors_chess'] = turn_data['errors_chess'].reshape(-1)
        draw_turn(turn_data, f'final-graph/turn/{model}-turn.pdf')
    elif types == 'strategy':
        strategy_data = {
            'data_mmlu': torch.zeros([8,2]),# np.random.uniform(30,60,(8,2)).reshape(-1).tolist(),       # [[strategy1-all,strategy1-part],[strategy2-all,strategy2-part]]
            'data_math': torch.zeros([8,2]),
            'data_chess': torch.zeros([8,2]),
            'errors_mmlu': torch.zeros([8,2]),
            'errors_math': torch.zeros([8,2]),
            'errors_chess': torch.zeros([8,2]),
        }
        dataset = ['mmlu', 'math', 'chess']
        strategymapping = {
            'gpt-1106-main': 'gpt-1106-strategy',
            'llama13-main': 'llama13-strategy', 
            'llama70-main': 'llama70-strategy', 
            'mixtral-main': 'mixtral-strategy', 
            'qwen-main': 'qwen-strategy'
        }
        model = list(strategymapping.keys())[4]
        Args.strategy = [decimal_to_binary(i, 3) for i in range(8)]
        Args.society = [1]
        Args.n_turn = 3
        Args.turn = 3
        Args.which_turn = 9
        for ds in dataset:
            Args.dataset = [ds]
            for idx, exp_type in enumerate([model, strategymapping[model]]):
                Args.experiment_type = exp_type
                mean, var = return_data()   # [n_strategy, n_society]
                strategy_data[f'data_{ds}'][:, idx] = mean.reshape(-1)
                strategy_data[f'errors_{ds}'][:, idx] = var.reshape(-1)
        strategy_data['data_chess'] = strategy_data['data_chess'].reshape(-1)
        strategy_data['data_mmlu'] = strategy_data['data_mmlu'].reshape(-1)
        strategy_data['data_math'] = strategy_data['data_math'].reshape(-1)
        strategy_data['errors_mmlu'] = strategy_data['errors_mmlu'].reshape(-1)
        strategy_data['errors_chess'] = strategy_data['errors_chess'].reshape(-1)
        strategy_data['errors_math'] = strategy_data['errors_math'].reshape(-1)
        draw_three_strategy(strategy_data, f'final-graph/strategy/{model}-strategy.pdf')
    elif types == '10-agent':
        mask = np.random.rand(9,5)
        mask[:] = 1
        for i in range(9):
            if i % 2 == 1:
                mask[i,2] = 0   # odd number of agent mask middle
        mask[0,0:2] = 0     # 2 agent, mask the first two

        agent_data = {
            'data': torch.zeros([8,5,9]),   # [n_strategy, n_society, n_agent]
            'error': torch.zeros([8,5,9]),
            'legend_label': None,
            'x_label': None,
            'title': None,
            'mask': mask
        }
        dataset = ['chess']
        assert experiment_type in ['gpt-1106-main','mixtral-main','qwen-main']
        model = '-'.join(experiment_type.split('-')[0:-1])
        society = {
            2: ['eo', 'eo', 'eo', 'ee', 'oo'], 
            3: ['eoe', 'eoo', 'eoe', 'eee', 'ooo'], 
            4: ['eoee', 'eooo', 'eoeo', 'eeee', 'oooo'], 
            5: ['eoeee', 'eoooo', 'eoeoe', 'eeeee', 'ooooo'], 
            6: ['eoeeee', 'eooooo', 'eoeoeo', 'eeeeee', 'oooooo'], 
            7: ['eoeeeee', 'eoooooo', 'eoeoeoe', 'eeeeeee', 'ooooooo'], 
            8: ['eoeeeeee', 'eooooooo', 'eoeoeoeo', 'eeeeeeee', 'oooooooo'], 
            9: ['eoeeeeeee', 'eoooooooo', 'eoeoeoeoe', 'eeeeeeeee', 'ooooooooo'], 
            10: ['eoeeeeeeee', 'eooooooooo', 'eoeoeoeoeo', 'eeeeeeeeee', 'oooooooooo']
        }
        for ds in dataset:
            Args.dataset = [ds]
            for idx_agent, n_agent in enumerate([2,3,4,5,6,7,8,9,10]):
                for idx_society, soc in enumerate(society[n_agent]):
                    Args.experiment_type = f"{model}-agent-{n_agent}"
                    if "gpt" not in Args.experiment_type:
                        Args.experiment_type = Args.experiment_type 
                    Args.society = [soc]
                    Args.n_agent = Args.agent = n_agent
                    mean, var = return_data()   # [n_strategy, n_society]  
                    agent_data['data'][:, idx_society, idx_agent] = mean.reshape(-1)
                    agent_data['error'][:, idx_society, idx_agent] = var.reshape(-1)
        agent_data['data'] = [agent_data['data'][i].numpy() for i in range(8)]
        agent_data['error'] = [agent_data['error'][i].numpy() for i in range(8)]
        label_set = {
            True: [f'{i} agents' for i in [2,3,4,5,6,7,8,9,10]],
            # True: [f'Society {i}' for i in range(5)],
            False: [f'Society: {i}' for i in ['$(eo)(\dot{e})$', '$(eo)(\dot{o})$', '$\dot{eo}$', '$\dot{e}$', '$\dot{o}$']]
        }
        agent_data['mask'] = agent_data['mask'].transpose(1,0)
        for idx, transpose in enumerate([False, True]):
        # transpose = True
            x_labels = label_set[transpose]
            legend_labels = label_set[transpose==False]
            sub_title = ['$p_0p_0p_0$', '$p_0p_0p_1$', '$p_0p_1p_0$', '$p_0p_1p_1$', '$p_1p_0p_0$', '$p_1p_0p_1$', '$p_1p_1p_0$', '$p_1p_1p_1$']
            sub_title = [f'({chr(ord("a")+idx)}) Strategy {i}' for idx, i in enumerate(sub_title)]
            agent_data['legend_label'] = legend_labels
            agent_data['x_label'] = x_labels
            agent_data['title'] = sub_title
            draw_10_agent(database=agent_data, transpose=transpose, file_name=f'final-graph/agent/{model}-10_agent_{idx}-new.pdf')
    elif types == '10-turn':
        
        Args.society = [1]
        Args.strategy = [
            "0111111111","1011111111","0100000000","1000000000",
            "0101010101","1010101010","1111111111","0000000000"
        ]
        Args.repeat = [1,2,3,4,5]
        turn_data = {
            'data': torch.zeros([8, 8]), # [n_strategy, n_turn]
            'error': torch.zeros([8, 8]), # []
            'title': []
        }
        # dataset = [['chess','math','mmlu'][0]]
        assert dataset in ['chess','math','mmlu']
        dataset = [dataset]
        assert len(dataset) == 1
        assert experiment_type in ['gpt-1106-main','mixtral-main','qwen-main']
        model = '-'.join(experiment_type.split('-')[0:-1])
        # model = 'qwen-max-1201'
        for ds in dataset:
            Args.dataset = [ds]
            Args.experiment_type = f"{model}-turn-10"
            for idx, which_turn in enumerate([9,11,13,15,17,19,21,23]):   # [3,5,7,9,11,13,15,17,19,21,23]   # [5,7,9,11,13,15,17,19,21]
                Args.n_turn = Args.turn = 10
                Args.which_turn = which_turn
                mean, var = return_data()   # [n_strategy, n_society]
                turn_data['data'][:, idx] = mean.reshape(-1)
                turn_data['error'][:, idx] = var.reshape(-1)
        turn_data['data'] = [turn_data['data'][i].numpy() for i in range(8)]
        turn_data['error'] = [turn_data['error'][i].numpy() for i in range(8)]
        for item in Args.strategy:
            title = ''
            for i in item:
                title += f'p_{i}'
            title = f'${title}$'
            turn_data['title'].append(title)
        draw_10_turn(database=turn_data, file_name=f'final-graph/turn/{model}-10_turn-{dataset[0]}-update.pdf')
    elif types == 'radar':
        radar_data = {
            'data': [torch.zeros([3, 8]), torch.zeros([3, 8]), torch.zeros([2, 8]), torch.zeros([4, 8])], # [3,8], [3,8], [2,8], [4,]
            'self-consitent': [torch.zeros([3, 8]), torch.zeros([3, 8]), torch.zeros([2, 8]), torch.zeros([4, 8])], # 暂时先这样，然后要取值的
            'titles': ["(a) All Datasets", "(b) MMLU (Part 1)", "(c) MMLU (Part 2)", "(d) Mathematics Domain"],
            'legend_labels':[
                'MMLU', 'MATH', 'Chess Move Validity',
                'MMLU-Chemistry','MMLU-Science','MMLU-Biology',
                'MMLU-Physics', 'MMLU-Statistics',
                'MATH-Level3','MATH-Level4','MATH-Level5','MMLU-Mathematics'
            ],
            'dot_names': [
                "$p_0p_0p_0$", "$p_0p_0p_1$", "$p_0p_1p_1$", "$p_0p_1p_0$", 
                "$p_1p_0p_0$", "$p_1p_0p_1$", "$p_1p_1p_0$", "$p_1p_1p_1$"
            ]
        }

        mapping_index = {
            'MMLU':{
                # 'biology', 'physics', 'statistics', 'mathematics', 'chemistry', 'science'
                # [0,8] [8,16] [16,24] [24,32], [32,41] [41,50]
                'Biology': [0,8],
                'Physics': [8,16],
                'Statistics': [16,24],
                'Mathematics': [24,32],
                'Chemistry': [32,41],
                'Science': [41,50]
            },
            'MATH': {
                # [0,22] [22,44] [44,50]
                'Level3': [0, 22],
                'Level4': [22, 44],
                'Level5': [44, 50]
            }
        }

        dataset = ['mmlu', 'math', 'chess']
        # model = ['prefix-gpt', 'prefix-llama70', 'prefix-llama13', 'prefix-mixtral', 'prefix-qwen', 'gpt-3.5'][5]
        assert experiment_type in ['gpt-1106-main','llama13-main','llama70-main','qwen-main','mixtral-main']
        model = experiment_type
        Args.n_turn = 3
        Args.turn = 3
        Args.society = [0,1,2,3]
        temp_cache = []
        for which_turn in [9,3]:
            cache = {}
            Args.which_turn = which_turn

            for ds in dataset:
                Args.dataset = [ds]
                Args.experiment_type = f'{model}'
                data = return_data(False, return_origin=True) # [n_repeat, n_society, n_strategy, n_case]
                copy_data = data.clone()                         # 复制一份
                assert data.shape[0] == 5 and data.shape[1] == 4 and data.shape[2] == 8 and data.shape[3] == 50
                ds = {'math':'MATH','mmlu':'MMLU','chess':'Chess Move Validity'}[ds]
                if ds in mapping_index:
                    for category in mapping_index[ds]:
                        assert f'{ds}-{category}' in radar_data['legend_labels']
                        start, end = mapping_index[ds][category]    
                        cur_data = data[:,:,:,start:end].sum(-1)       # [n_repeat, n_society, n_strategy]
                        # print(cur_data.shape)
                        cur_data = cur_data / (end-start) * 100    
                        cur_data = cur_data.permute(2,1,0)   # [n_strategy, n_society, n_repeat]
                        mean = cur_data.mean(-1)            # [n_strategy, n_society]
                        mean = mean.mean(-1)            # [n_strategy]
                        assert mean.shape[0] == 8
                        cache[f'{ds}-{category}'] = mean
                        # if ds == 'math':
                        #     print(mean)
                        #     a=input('hello')
                _ = copy_data.sum(-1) / Args.n_case * 100   # [n_repeat, n_society, n_strategy]
                _ = _.permute(2,1,0)                        # [n_strategy, n_society, n_repeat]
                _ = _.mean(dim=[1,2])                       # [n_strategy]
                # _ = _.mean(-1).mean(-1)
                cache[ds] = _
                assert ds in radar_data['legend_labels']
            assert len(cache) == len(radar_data['legend_labels']), f"{len(cache)}!={len(radar_data['legend_labels'])}, {list(cache.keys())}"

            temp_cache.append(cache)
        
        cache = temp_cache[0]
        allocate = [data.shape[0] for data in radar_data['data']]
        assert sum(allocate) == len(cache)
        pointer = 0
        for idx, fisrt in enumerate(allocate):
            for i in range(fisrt):
                assert cache[radar_data['legend_labels'][pointer]].shape[0] == radar_data['data'][idx][i].shape[0]
                # print(radar_data['data'][idx][i].shape, cache[radar_data['legend_labels'][pointer]].shape, radar_data['legend_labels'][pointer])
                radar_data['data'][idx][i] = cache[radar_data['legend_labels'][pointer]]
                pointer += 1
        for i in range(len(radar_data['data'])):
            radar_data['data'][i] = radar_data['data'][i].numpy()
        
        cache = temp_cache[1]
        allocate = [data.shape[0] for data in radar_data['self-consitent']]
        assert sum(allocate) == len(cache)
        pointer = 0
        for idx, fisrt in enumerate(allocate):
            for i in range(fisrt):
                assert cache[radar_data['legend_labels'][pointer]].shape[0] == radar_data['self-consitent'][idx][i].shape[0]
                # print(radar_data['self-consitent'][idx][i].shape, cache[radar_data['legend_labels'][pointer]].shape, radar_data['legend_labels'][pointer])
                radar_data['self-consitent'][idx][i] = cache[radar_data['legend_labels'][pointer]]
                pointer += 1
        _data = []
        for i in range(len(radar_data['self-consitent'])):
            radar_data['self-consitent'][i] = radar_data['self-consitent'][i].numpy()
            _data.append([])
            for j in range(radar_data['self-consitent'][i].shape[0]):
                _data[-1].append(radar_data['self-consitent'][i][j][0])
        # print(radar_data['self-consitent'])
        radar_data['self-consitent'] = _data
        print(radar_data['self-consitent'])
        # a = input("hello")
        # ============================
        print(radar_data['data'])
        draw_radar(radar_data, f'final-graph/radar/{model}-radar.pdf')
    elif types == '10-agent-consistent':
        mask = np.random.rand(9,5)
        mask[:] = 1
        for i in range(9):
            if i % 2 == 1:
                mask[i,2] = 0   
        mask[0,0:2] = 0     

        agent_data = {
            'data': torch.zeros([8,5,9,3]),   # [n_strategy, n_society, n_agent. n_turn]
            'error': torch.zeros([8,5,9,3]),
            'legend_label': None,
            'x_label': None,
            'title': None,
            'mask': mask.transpose(1,0)
        }
        dataset = ['chess']
        assert experiment_type in ['gpt-1106-main','mixtral-main','qwen-main']
        model = '-'.join(experiment_type.split('-')[0:-1])
        society = {
            2: ['eo', 'eo', 'eo', 'ee', 'oo'], 
            3: ['eoe', 'eoo', 'eoe', 'eee', 'ooo'], 
            4: ['eoee', 'eooo', 'eoeo', 'eeee', 'oooo'], 
            5: ['eoeee', 'eoooo', 'eoeoe', 'eeeee', 'ooooo'], 
            6: ['eoeeee', 'eooooo', 'eoeoeo', 'eeeeee', 'oooooo'], 
            7: ['eoeeeee', 'eoooooo', 'eoeoeoe', 'eeeeeee', 'ooooooo'], 
            8: ['eoeeeeee', 'eooooooo', 'eoeoeoeo', 'eeeeeeee', 'oooooooo'], 
            9: ['eoeeeeeee', 'eoooooooo', 'eoeoeoeoe', 'eeeeeeeee', 'ooooooooo'], 
            10: ['eoeeeeeeee', 'eooooooooo', 'eoeoeoeoeo', 'eeeeeeeeee', 'oooooooooo']
        }
        for ds in dataset:
            Args.dataset = [ds]
            for idx_agent, n_agent in enumerate([2,3,4,5,6,7,8,9,10]):
                for idx_society, soc in enumerate(society[n_agent]):
                    for idx_turn, which_turn in enumerate([5,7,9]):
                        Args.which_turn = which_turn
                        Args.experiment_type = f"{model}-agent-{n_agent}"
                        if "gpt" not in Args.experiment_type:
                            Args.experiment_type = Args.experiment_type
                        Args.society = [soc]
                        Args.n_agent = Args.agent = n_agent
                        mean, var = return_data(value='consistent')   # [n_strategy, n_society]
                        assert (mean<=10).all(), f"{mean}"
                        agent_data['data'][:, idx_society, idx_agent, idx_turn] = mean.reshape(-1) / n_agent
                        agent_data['error'][:, idx_society, idx_agent, idx_turn] = var.reshape(-1) / n_agent
        agent_data['data'] = agent_data['data'].numpy()
        agent_data['error'] = agent_data['error'].numpy()
        label_set = {
            True: [f'{i} agents' for i in [2,3,4,5,6,7,8,9,10]],
            # True: [f'Society {i}' for i in range(5)],
            False: [f'Society: {i}' for i in ['$(eo)(\dot{e})$', '$(eo)(\dot{o})$', '$\dot{eo}$', '$\dot{e}$', '$\dot{o}$']]
        }
        for idx, transpose in enumerate([False, True]):
        # transpose = True
            if transpose:
                agent_data['data'] = agent_data['data'].transpose(0,2,1,3)
            x_labels = label_set[transpose]
            legend_labels = label_set[transpose==False]
            sub_title = ['$p_0p_0p_0$', '$p_0p_0p_1$', '$p_0p_1p_0$', '$p_0p_1p_1$', '$p_1p_0p_0$', '$p_1p_0p_1$', '$p_1p_1p_0$', '$p_1p_1p_1$']
            sub_title = [f'({chr(ord("a")+idx)}) Strategy {i}' for idx, i in enumerate(sub_title)]
            agent_data['legend_label'] = legend_labels
            agent_data['x_label'] = x_labels
            agent_data['title'] = sub_title
            # draw_10_agent_consistent(
            #     database=agent_data, 
            #     file_name=f'distribute/10_agent_{idx}-consistent-turn.pdf', 
            #     # ylabel='Consistent'
            # )
            # print(idx)
            draw_10_agent_consistent_line(
                database=agent_data,
                file_name=f'final-graph/consistent/{model}-110_agent_{idx}-consistent-turn-line.pdf',  # -no-division
                ylabel=['Average Ratio of\nConsensus Clusters','Average Quantity of\nConsensus Clusters'][idx],
                plotGroup=[[5],[1,4,4]][idx]
            )
            agent_data['mask'] = agent_data['mask'].transpose(1,0)
    
    elif types == 'word':
        assert experiment_type in ['gpt-1106-main', 'qwen-main', 'llama13-main', 'llama70-main', 'mixtral-main']
        society = ["0_harmony", "3_harmony"]
        datasets = ['mmlu', 'math', 'chess']
        
        exp_types = [experiment_type]
        for social in society:
            for ds in datasets:
                for exp_type in exp_types:
                    prefix = f'results/{exp_type}/{ds}'
                    idx = [1,2,3,4,5]
                    draw_word(
                        folder_list=[f"{prefix}/{i}" for i in idx],
                        filter_str=social,
                        name=f"final-graph/wordcloud/{exp_type}_{social}_{ds}"
                    )
    else:
        assert False

def anova(
    types:str,
    dataset: str,
    experiment_type: list
):
    def main_split_anova(database):
        # database: [n_repeat, n_society, n_strategy]
        assert database.shape[0] == 5
        assert database.shape[1] == 4
        # assert database.shape[2] == 8
        print('='*20, 'main-split', '='*20)
        results = ['='*20+'main-split'+'='*20]
        database = database.permute(1,2,0)  # [n_society, n_strategy, n_repeat]
        # given the society, explore the impact of the strategy
        for society_id in range(database.shape[0]):
            anova_data = []
            for strategy_id in range(database.shape[1]):
                anova_data.append(database[society_id, strategy_id].tolist())
                # [[1,2,3,4,5],[1,2,3,4,5]]
            w, p = stats.levene(*anova_data)
            if p < 0.05:
                results.append('The assumption of homogeneity of variance is not valid.')
            else:
                f_value, p_value = stats.f_oneway(*anova_data)
                print(f"{p_value},$S_{society_id+1}$")
                results.append(f"{p_value},$S_{society_id+1}$")
        # Given the strategy, explore the impact of the society
        for strategy_id in range(database.shape[1]):
            anova_data = []
            strategy = bin(strategy_id)[2:].zfill(3)
            for society_id in range(database.shape[0]):
                anova_data.append(database[society_id, strategy_id].tolist())
            w, p = stats.levene(*anova_data)
            if p < 0.05:
                results.append('The assumption of homogeneity of variance is not valid.')
            else:
                f_value, p_value = stats.f_oneway(*anova_data)
                print(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
                results.append(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
        return results

    def agent_anova(database:list, strategies:list[int]):
        # [n_strategy, n_agent, n_repeat]
        print('='*20, 'agent', '='*20)
        results = ['='*20+'agent'+'='*20]
        for strategy_id in range(len(database)):
            strategy = strategies[strategy_id]
            A = database[strategy_id][0]
            B = database[strategy_id][1]
            C = database[strategy_id][2]
            data = [A, B, C]
            w, p = stats.levene(*data)
            if p < 0.05:
                results.append('The assumption of homogeneity of variance is not valid.')
            else:
                f_value, p_value = stats.f_oneway(*data)
                print(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
                results.append(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
        return results

    def turn_anova(database:list, strategies:list):
        # [n_strategy, n_turn, n_repeat]
        results = ['='*20+'turn'+'='*20]
        print('='*20, 'turn', '='*20)
        for strategy_id in range(len(database)):
            strategy = strategies[strategy_id]
            A = database[strategy_id][0]
            B = database[strategy_id][1]
            C = database[strategy_id][2]
            data = [A, B, C]
            w, p = stats.levene(*data)
            if p < 0.05:
                results.append('The assumption of homogeneity of variance is not valid.')
            else:
                f_value, p_value = stats.f_oneway(*data)
                print(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
                results.append(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
        return results

    def strategy_anova(database:list, strategies:list):
        # [n_strategy, 2, n_repeat]
        results = ['='*20+'strategy'+'='*20]
        print('='*20, 'strategy', '='*20)
        for strategy_id in range(len(database)):
            strategy = strategies[strategy_id]
            A = database[strategy_id][0]
            B = database[strategy_id][1]
            data = [A, B]
            w, p = stats.levene(*data)
            if p < 0.05:
                results.append('The assumption of homogeneity of variance is not valid.')
            else:
                f_value, p_value = stats.f_oneway(*data)
                print(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
                results.append(f"{p_value},${''.join(['p_'+str(i) for i in strategy])}$")
        return results

    assert isinstance(experiment_type, list)
    for exp_type in experiment_type:
        assert os.path.exists(f'results/{exp_type}') and os.path.isdir(f'results/{exp_type}'), \
            f"The folder `results/{exp_type}` don't exist. Please make sure the `--experiment_type` is valid."
    models = experiment_type
    output_results = []
    if types == 'main':
        # dataset = ['mmlu', 'math', 'chess']
        dataset = [dataset]
        Args.society = Args.role = [0,1,2,3]
        Args.turn = Args.n_turn = 3
        Args.n_agent = Args.agent = 3
        Args.strategy = [decimal_to_binary(i, 3) for i in range(2**Args.n_turn)]
        for model in models:
            for ds in dataset:
                Args.which_turn = 9
                Args.dataset = [ds]
                Args.experiment_type = model
                print('='*20, Args.experiment_type, '='*20)
                output_results.append('='*20+Args.experiment_type+'='*20)
                data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                output_results.extend(main_split_anova(data))
    elif types == 'turn':
        # 这里主要就是变换which_turn
        dataset = ['mmlu', 'chess']
        Args.society = Args.role = [1]
        Args.n_agent = Args.agent = 3
        Args.turn = Args.n_turn = 4
        Args.strategy = [decimal_to_binary(i, 4) for i in range(2**Args.n_turn)]
        # model = []
        for model in models:
            assert model in ['llama13-turn-4', 'llama70-turn-4'], \
                "We only test the 4 round collaboration in the llama-series model. If you want to explore the other model including gpt, mixtral or qwen, please set the `--types` to `10-turn`."
        for model in models:
            Args.experiment_type = model
            # print('='*20, Args.experiment_type, '='*20)
            for ds in dataset:
                print('='*20, model, ',', ds, '='*20)
                output_results.append('='*20+model+','+ds+'='*20)
                data = []
                for which_turn in [7,9,11]:
                    Args.which_turn = which_turn
                    Args.dataset = [ds]
                    new_data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                    assert new_data.shape[1] == 1
                    data.append(new_data)
                # data: []
                # target: [n_strategy, n_turn, n_repeat]
                data = torch.cat(data, dim=1)   # [n_repeat, n_turn, n_strategy]
                data = data.permute(2,1,0).tolist()
                output_results.extend(turn_anova(data, Args.strategy))
    elif types == '10-turn':
        # dataset = ['mmlu','math','chess']
        dataset = [dataset]
        Args.society = Args.role = [1]
        Args.n_agent = Args.agent = 3
        Args.turn = Args.n_turn = 10
        Args.strategy = ['0000000000','1000000000','0100000000','1010101010','0101010101','1011111111','0111111111','1111111111']
        for model in models:
            assert model in ['gpt-1106-turn-10', 'qwen-turn-10', 'mixtral-turn-10']
        for model in models:
            for ds in dataset:
                print('='*20, model, ',', ds, '='*20)
                output_results.append('='*20+model+','+ds+'='*20)
                data = []
                Args.experiment_type = model
                for which_turn in [7,9,11,13,15,17,19,21,23]:
                    Args.which_turn = which_turn
                    Args.dataset = [ds]
                    new_data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                    assert new_data.shape[1] == 1
                    data.append(new_data)
                # data: []
                # target: [n_strategy, n_turn, n_repeat]
                data = torch.cat(data, dim=1)   # [n_repeat, n_turn, n_strategy]
                data = data.permute(2,1,0).tolist()
                output_results.extend(turn_anova(data, Args.strategy))
    elif types == 'agent':
        assert dataset in ['mmlu','chess'], "We only support the mmlu and chess, please set the `--dataset` to `mmlu` or `chess`."
        dataset = [dataset]
        Args.society = Args.role = [1]
        Args.turn = Args.n_turn = 3
        Args.strategy = [decimal_to_binary(i, 3) for i in range(2**Args.n_turn)]
        for model in models:
            assert model in ['llama13-main', 'llama70-main'],\
                "We only support the model `llama13` or `llama70` when you set the `--types` to `agent`. If you want to explore the `qwen`, `gpt` or `mixtral`, please set the `--types` to `10-agent`."
        for model in models:
            for ds in dataset:
                print('='*20, model, ',', ds, '='*20)
                output_results.append('='*20+model+','+ds+'='*20)
                Args.dataset = [ds]
                data = []
                for n_agent in [2,3,4]:
                    if n_agent == 3:
                        Args.experiment_type = f"{model}"
                    else:
                        Args.experiment_type = f"{'-'.join(model.split('-')[0:-1])}-agent-{n_agent}"
                    Args.n_agent = Args.agent = n_agent
                    # Args.society = [2] if model == 'gpt-3.5' and n_agent==2 else [1]
                    new_data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                    assert new_data.shape[1] == 1
                    data.append(new_data)
                data = torch.cat(data, dim=1)   # [n_repeat, n_agent, n_strategy]
                # target: [n_strategy, n_agent, n_repeat]
                # data = torch.cat(data, dim=1)   # [n_repeat, n_agent, n_strategy]
                data = data.permute(2,1,0).tolist()
                output_results.extend(agent_anova(data, Args.strategy))
    elif types == '10-agent':
        num2str = {
            2:"eo eo eo ee oo".split(" "),
            3:"eoe eoo eoe eee ooo".split(" "),
            4:"eoee eooo eoeo eeee oooo".split(" "),
            5:"eoeee eoooo eoeoe eeeee ooooo".split(" "),
            6:"eoeeee eooooo eoeoeo eeeeee oooooo".split(" "),
            7:"eoeeeee eoooooo eoeoeoe eeeeeee ooooooo".split(" "),
            8:"eoeeeeee eooooooo eoeoeoeo eeeeeeee oooooooo".split(" "),
            9:"eoeeeeeee eoooooooo eoeoeoeoe eeeeeeeee ooooooooo".split(" "),
            10:"eoeeeeeeee eooooooooo eoeoeoeoeo eeeeeeeeee oooooooooo".split(" "),
        }
        assert dataset == 'chess', 'Only the chess move validity dataset is supported at the 10 agents settings. Please set `--dataset` to `chess` when you set the `types` to `10-agent`.'
        dataset = ['chess']
        Args.society = Args.role = [1]
        Args.turn = Args.n_turn = 3
        Args.strategy = [decimal_to_binary(i, 3) for i in range(2**Args.n_turn)]
        for society_id in range(5):
            for model in models:
                for ds in dataset:
                    print('='*20, model, ',', ds, f'S{society_id}','='*20)
                    output_results.append('='*20+model+','+ds+'='*20)
                    Args.dataset = [ds]
                    data = []
                    for n_agent in [2,3,4,5,6,7,8,9,10]:
                        Args.society = Args.role = [num2str[n_agent][society_id]]
                        Args.experiment_type = f"{'-'.join(model.split('-')[0:-1])}-agent-{n_agent}"
                        Args.n_agent = Args.agent = n_agent
                        new_data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                        assert new_data.shape[1] == 1
                        data.append(new_data)
                    data = torch.cat(data, dim=1)   # [n_repeat, n_agent, n_strategy]
                    # target: [n_strategy, n_agent, n_repeat]
                    # data = torch.cat(data, dim=1)   # [n_repeat, n_agent, n_strategy]
                    data = data.permute(2,1,0).tolist()
                    output_results.extend(agent_anova(data, Args.strategy))
    elif types == 'strategy':
        # dataset = ['mmlu', 'math','chess']
        dataset = [dataset]
        Args.society = Args.role = [1]
        Args.turn = Args.n_turn = 3
        Args.n_agent = Args.agent = 3
        Args.strategy = [decimal_to_binary(i, 3) for i in range(2**Args.n_turn)]
        strategy_file = {
            'gpt-1106-main': 'gpt-1106-strategy',
            'llama13-main': 'llama13-strategy', 
            'llama70-main': 'llama70-strategy', 
            'mixtral-main': 'mixtral-strategy', 
            'qwen-main': 'qwen-strategy'
        }
        for model in models:
            for ds in dataset:
                print('='*20, model,',',ds, '='*20)
                output_results.append('='*20+model+','+ds+'='*20)
                Args.dataset = [ds]
                data = []
                for exp_type in [f'{model}',strategy_file[model]]:
                    Args.experiment_type = exp_type
                    new_data = return_data(return_mean_and_var=False)   # [n_repeat, n_society, n_strategy] torch.Tensor
                    assert new_data.shape[1] == 1
                    data.append(new_data)
                # data = torch.cat(data, dim=1)   # [n_repeat, 2, n_strategy]
                # target: [n_strategy, n_agent, n_repeat]
                data = torch.cat(data, dim=1)   # [n_repeat, 2, n_strategy]
                data = data.permute(2,1,0).tolist()
                output_results.extend(strategy_anova(data,Args.strategy))
    else:
        assert False, "invalid `--types`"

    print('='*100)
    print('||',' '*45,'output',' '*45,'||')
    print('='*100)
    for i in output_results:
        print(i)


if __name__ == '__main__':
    fire.Fire()

"""
python evaluate.py main_table --experiment_type gpt-1106-main --dataset mmlu

python evaluate.py anova --types main --dataset chess --experiment_type "['llama13-main','gpt-1106-main']"
python evaluate.py anova --types turn --dataset chess  --experiment_type "['llama13-turn-4','llama70-turn-4']"
python evaluate.py anova --types 10-turn --dataset chess --experiment_type "['gpt-1106-turn-10', 'qwen-turn-10', 'mixtral-turn-10']"
python evaluate.py anova --types agent --dataset chess --experiment_type "['llama13-main','llama70-main']"
python evaluate.py anova --types 10-agent --dataset chess --experiment_type "['gpt-1106-main','qwen-main']"
python evaluate.py anova --types strategy --dataset chess --experiment_type "['gpt-1106-main','qwen-main']"

python evaluate.py draw --types distribute --experiment_type gpt-1106-main
python evaluate.py draw --types agent --experiment_type llama13-main
python evaluate.py draw --types turn --experiment_type llama70-main
python evaluate.py draw --types strategy --experiment_type gpt-1106-main
python evaluate.py draw --types 10-agent --experiment_type gpt-1106-main
python evaluate.py draw --types 10-turn --experiment_type gpt-1106-main --dataset chess
python evaluate.py draw --types radar --experiment_type gpt-1106-main
python evaluate.py draw --types 10-agent-consistent --experiment_type gpt-1106-main
python evaluate.py draw --types word --experiment_type gpt-1106-main

"""