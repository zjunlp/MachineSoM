import argparse
from evaluate import load_dataset, decimal_to_binary, parse_answer, _judge_answer
from evaluate import judge_answer as js
import os
import pickle
from draw import draw_conformity
from draw import draw_consistent
import numpy as np

class ARGS:
    dataset: str = None

def judge_answer(cur_answer, gt_answer) -> bool:
    if isinstance(cur_answer, list):
        return js(agent_answer=cur_answer, gt_answer=gt_answer[0], args=ARGS) == 1
    else:
        math_pattern = [
            " \\text{ positive integers, } 12 \\text{ negative integers}", " \\, \\text{cm}", "\\, \\text{cm}",
            "^\\circ", "{\\text{Degree of }} g(x) = ", "°", " \\text{ cm}", "\\text{ cm}", "b = ", "r = ", "x = ", "m+n = ", "\\text{ degrees}", "x + y + z = "
        ]
        return _judge_answer(args=ARGS, gt_answer=gt_answer, agent_final_answer=cur_answer, math_pattern=math_pattern, agent_answer=None)
    # return cur_answer == gt_answer

def majority_ans(answer:list, only_same_num:bool=False):
    assert len(answer) > 1
    if len(answer) == 2:
        if answer[0] == answer[1]:
            if only_same_num:
                return 2
            return [answer[0]]
        else:
            if only_same_num:
                return 0
            return []
    else:
        mapping = {}
        for ans in answer:
            if ans not in mapping:
                mapping[ans] = 1
            else:
                mapping[ans] += 1
        max_value = max(mapping.values())
        if max_value == 1:
            if only_same_num:
                return 0
            return []
        else:
            max_keys = [key for key, value in mapping.items() if value == max_value]
        if only_same_num:
            return mapping[max_keys[0]]
        return max_keys

def strip_and_lower(target):
    if isinstance(target, list):
        return [str(_).lower().strip() for _ in target]
    return str(target).lower().strip()

def _judge_single_conformity(pre_answer:list, cur_answer:str, pos:int, true_answer:str=None):
    # pos: start from 0
    # =====Situation 1=====
    # A A C
    #     A
    # =====Situation 2=====
    # A A C
    # A
    results = {
        "1": {
            "conformity": None, # True/False
            "type": None        # tf/ft/ff/tt         t->f f->t
        },
        "2": {
            "conformity": None, # True/False
            "type": None        # tf/ft/ff/tt         t->f f->t
        },
        'answer': {
            'value': None,      # content
            'label': None,      # answer label
            'same': None,       # how many unique answer
        }
    }
    mapping = {True:"t", False:"f"}
    # =====Situation 1=====
    other_agent_answer = pre_answer[0:pos]
    other_agent_answer.extend(pre_answer[pos+1:])
    other_final_answer = majority_ans(other_agent_answer)
    results['1']['conformity'] = cur_answer in other_final_answer
    if results['1']['conformity'] and true_answer is not None:
        results['1']['type'] = f'{mapping[judge_answer(pre_answer[pos], true_answer)]}{mapping[judge_answer(cur_answer, true_answer)]}'
    
    # =====================
    other_agent_answer = pre_answer
    other_final_answer = majority_ans(other_agent_answer)
    results['2']['conformity'] = cur_answer in other_final_answer
    if results['2']['conformity'] and true_answer is not None:
        results['2']['type'] = f'{mapping[judge_answer(pre_answer[pos], true_answer)]}{mapping[judge_answer(cur_answer, true_answer)]}'
        # results['2']['type'] = results['1']['type']
    # =====================
    results['answer']['value'] = pre_answer
    results['answer']['same'] = majority_ans(pre_answer, only_same_num=True)
    results['answer']['label'] = [judge_answer(ans, true_answer) for ans in pre_answer]
    return results

def judge_conformity(pre_answer:list, cur_answer:list, true_answer:str, situation:int=1)->dict:
    """
    {
        "dist": [True, True, False],
        "type": ['tf','ff','ft','tt'],
        "answer": { # pre_answer
            'value': [a1,a2,a3],
            'same': 2   
            'label': [True, True, True] 
        }
    }
    """
    results = {"dist": [], "type":[]}   # "answer":{'value': None, 'same': None, 'label': None}
    assert situation in [1,2]
    situation:str = str(situation)
    assert len(pre_answer) == len(cur_answer)
    pre_answer = strip_and_lower(pre_answer)
    cur_answer = strip_and_lower(cur_answer)
    
    for i in range(len(cur_answer)):
        analyze:dict = _judge_single_conformity(pre_answer, cur_answer[i], i, true_answer)
        results['dist'].append(analyze[situation]['conformity'])
        results['type'].append(analyze[situation]['type'])
    return results

def judge_consistent(answers:list, true_answer:str):
    def counts(data:list):
        mapping = {}
        for i in data:
            mapping[i] = 0
        return len(mapping)
    answers = strip_and_lower(answers)
    analyze = _judge_single_conformity(pre_answer=answers, cur_answer=true_answer, true_answer=true_answer, pos=0)
    # print(analyze)
    # print(analyze['answer']['value'])
    # a = input(f"{counts(analyze['answer']['value'])}")
    return {
        'answers': analyze['answer']['value'],
        'labels': analyze['answer']['label'],
        'same': analyze['answer']['same'],
        'cnt of answers': counts(analyze['answer']['value'])
    }

def parse_agrs():
    parser = argparse.ArgumentParser(description='agent')
    parser.add_argument('--dataset', type=str, default='mmlu')
    parser.add_argument('--repeat', type=str, default="[1,2,3,4,5]")  
    parser.add_argument('--experiment_type', type=str, default='main')
    parser.add_argument('--agent', type=int, default=3) 
    parser.add_argument('--role', type=str, default="[0,1,2,3]") 
    parser.add_argument('--turn', type=int, default=3)  
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--type', type=str, default='consistent')
    return parser.parse_args()

def check_args(args):
    args.role = eval(args.role)
    args.repeat = eval(args.repeat)
    ARGS.dataset = args.dataset
    assert args.type.lower() in ['consistent','conformity']
    print(args.type.lower())

def conformity(args):
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
                    if "{" + answers[i] + "}" == answers[j]:
                        answers[i] = answers[j]
        return answers
    database = load_dataset(args)
    all_results = {}
    file_dir = f'./results/{args.experiment_type}/{args.dataset}'
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2 ** args.turn)]
    if args.repeat != -1:
        assert str(args.repeat) in repeat_dir_name
        repeat_dir_name = [str(args.repeat)]
    
    for repeat in repeat_dir_name:
        for role in args.role:
            all_results[role] = {}
            for strategy in strategies:
                all_results[role][strategy] = {
                    'type': [],
                    'conformity':[],
                }
                for case_id in range(args.n_case):
                    origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                    origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                    now_full_path = origin_full_path
                    try:
                        agent_center = pickle.load(open(now_full_path,'rb'))
                    except:
                        print("shudown:", now_full_path)
                        if args.experiment_type != "turn":
                            assert False
                        else:
                            agent_center = pickle.load(open(
                                f"{file_dir}/{repeat}/{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}_shutdown.pkl"
                                , "rb"
                            ))
                            for i in range(len(agent_center)):
                                agent_center[i] = agent_center[i][0:-1]
                    gt_answers = database['answer'][case_id]
                    agent_turn_answers = []
                    for which_turn in [3,5,7,9]:
                        agent_answers = []
                        for agent_id in range(args.agent):
                            context = agent_center[agent_id][which_turn]
                            assert context["role"] == "assistant"
                            _ = parse_answer(dataset=args.dataset, content=context["content"], task_info=database["task_info"][case_id])
                            agent_answers.append(_)
                        if args.dataset.lower() == 'math':
                            agent_answers = check_include(agent_answers)
                        agent_turn_answers.append(agent_answers)
                    all_results[role][strategy]['conformity'].append([])
                    all_results[role][strategy]['type'].append([])
                    for idx, thinking_pattern in enumerate(strategy):
                        if thinking_pattern == '0':
                            results = judge_conformity(                 # results['dist'], results['type']
                                pre_answer=agent_turn_answers[idx],
                                cur_answer=agent_turn_answers[idx+1],
                                true_answer=gt_answers,
                                situation=2
                            )
                            all_results[role][strategy]['conformity'][-1].append(results['dist'])
                            all_results[role][strategy]['type'][-1].append(results['type'])

    return all_results               

def consistent(args):
    math_pattern = [
        " \\text{ positive integers, } 12 \\text{ negative integers}", " \\, \\text{cm}", "\\, \\text{cm}",
        "^\\circ", "{\\text{Degree of }} g(x) = ", "°", " \\text{ cm}", "\\text{ cm}", "b = ", "r = ", "x = ", "m+n = ", "\\text{ degrees}", "x + y + z = "
    ]
    
    def check_include(answers: list):
        """['{12}', '12', '24']"""
        for i in range(len(answers)):
            if answers[i] is not None:
                for _ in math_pattern:
                    answers[i] = answers[i].replace(_, "")
        for i in range(len(answers)):
            for j in range(len(answers)):
                if i != j and answers[i] is not None and answers[j] is not None and answers[i] in answers[j] and answers[i] != answers[j]:
                    if "{" + answers[i] + "}" == answers[j]:
                        answers[i] = answers[j]
        return answers
    database = load_dataset(args)
    all_results = {}
    file_dir = f'./results/{args.experiment_type}/{args.dataset}'
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2 ** args.turn)]
    if args.repeat != -1:
        assert str(args.repeat) in repeat_dir_name
        repeat_dir_name = [str(args.repeat)]
    
    for repeat in repeat_dir_name:
        for role in args.role:
            all_results[role] = {}
            for strategy in strategies:
                all_results[role][strategy] = {
                    'answers': [],  
                    'labels': [],   
                    'same': [],      
                    'cnt of answers': []    
                }
                for case_id in range(args.n_case):
                    origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                    origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                    now_full_path = origin_full_path
                    try:
                        agent_center = pickle.load(open(now_full_path,'rb'))
                    except:
                        if args.experiment_type != "turn":
                            assert False
                        else:
                            agent_center = pickle.load(open(
                                f"{file_dir}/{repeat}/{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}_shutdown.pkl"
                                , "rb"
                            ))
                            for i in range(len(agent_center)):
                                agent_center[i] = agent_center[i][0:-1]
                    gt_answers = database['answer'][case_id]
                    agent_turn_answers = []
                    for which_turn in [3,5,7,9]:
                        agent_answers = []
                        for agent_id in range(args.agent):
                            context = agent_center[agent_id][which_turn]
                            assert context["role"] == "assistant"
                            _ = parse_answer(dataset=args.dataset, content=context["content"], task_info=database["task_info"][case_id])
                            agent_answers.append(_)
                        if args.dataset.lower() == 'math':
                            agent_answers = check_include(agent_answers)
                        agent_turn_answers.append(agent_answers)
                    all_results[role][strategy]['answers'].append([])
                    all_results[role][strategy]['labels'].append([])
                    all_results[role][strategy]['same'].append([])
                    all_results[role][strategy]['cnt of answers'].append([])
                    for t in range(4):
                        analyze = judge_consistent(agent_turn_answers[t], true_answer=gt_answers)
                        all_results[role][strategy]['answers'][-1].append(analyze['answers'])
                        all_results[role][strategy]['labels'][-1].append(analyze['labels'])
                        all_results[role][strategy]['same'][-1].append(analyze['same'])
                        all_results[role][strategy]['cnt of answers'][-1].append(analyze['cnt of answers'])

    return all_results              
                            
def view1(args, all_results):
    """
    [   
        # repeat 1
        {
            role:{
                strategy:{
                    'conformity': [[[a1,a2,a3],[a1,a2,a3]], [], []],        # [n_case, n_debate, n_agent]
                    'type': [[[a1,a2,a3],[a1,a2,a3]]]
                }
            }
        },
        # repeat 2
        {},
        # repeat 3
        {},
        # repeat 4
        {},
        # repeat 5
        {}
    ]
    """
    data_source = {
        '1-0':{
        },
        '2-0':{
        },
        '3-0':{
        }
    }
    for strategy_id in range(8):
        strategy = decimal_to_binary(strategy_id, 3)
        for repeat in range(5):
            for case_id in range(50):
                for role in range(4):
                    cnt = 0
                    for pos, thinking_pattern in enumerate(strategy):
                        if thinking_pattern == '0':
                            agent_conformity:list = all_results[repeat][role][strategy]['conformity'][case_id][cnt]
                            agent_type:list = all_results[repeat][role][strategy]['type'][case_id][cnt]
                            if strategy[0:pos+1] not in data_source[f'{pos+1}-0']:
                                data_source[f'{pos+1}-0'].update(
                                    {strategy[0:pos+1]: {t:0 for t in ['tf','ft','tt','ff']}}
                                )
                            for t, c in zip(agent_type, agent_conformity):
                                if t is not None and c:
                                    data_source[f'{pos+1}-0'][strategy[0:pos+1]][t] += 1
                            cnt += 1
                            
    data_draw = []
    categories = []
    print(data_source)
    for k1 in data_source:
        for k2 in data_source[k1]:
            categories.append(f'{k1} {k2}')
            data_draw.append([data_source[k1][k2][t] for t in ['tf','ft','tt','ff']])
    import matplotlib.pyplot as plt
    import numpy as np
    bar_width = 0.5  
    bar_positions = np.arange(len(categories))  
    color = ['red','green','blue','black','pink', 'purple','orange']
    data_draw = np.array(data_draw).T   # [7,4] -> [4,7]
    for i in range(len(data_draw)):
        if i == 0:
            plt.bar(bar_positions, data_draw[i], color=color[i], edgecolor='white', width=bar_width)
        else:
            plt.bar(bar_positions, data_draw[i], bottom=np.sum(data_draw[0:i],0),color=color[i], edgecolor='white', width=bar_width)

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Stacked Bar Chart with Different Colors')

    plt.xticks(bar_positions, categories)
    plt.legend(['tf','ft','tt','ff'])
    plt.savefig(f'{args.dataset}_conformity.png')

def view2(args, all_results):
    data_source = {
        'oc':{t:0 for t in ['tf','ft','tt','ff']},    # overconfident
        'eg':{t:0 for t in ['tf','ft','tt','ff']}     # easygoing
    }
    for strategy_id in range(8):
        strategy = decimal_to_binary(strategy_id, 3)
        for repeat in range(5):
            for case_id in range(50):
                for role in range(4):
                    traits = {
                        0:['oc','oc','oc'], 1:['oc','oc','eg'], 2:['oc','eg','eg'], 3:['eg','eg','eg']
                    }[role]
                    cnt = 0
                    for pos, thinking_pattern in enumerate(strategy):
                        if thinking_pattern == '0':
                            agent_conformity:list = all_results[repeat][role][strategy]['conformity'][case_id][cnt]
                            agent_type:list = all_results[repeat][role][strategy]['type'][case_id][cnt]
                            for idx, trait in enumerate(traits):
                                c = agent_conformity[idx]
                                t = agent_type[idx]
                                if t is not None and c:
                                    data_source[trait][t] += 1
                            # if strategy[0:pos+1] not in data_source[f'{pos+1}-0']:
                            #     data_source[f'{pos+1}-0'].update(
                            #         {strategy[0:pos+1]: {t:0 for t in ['tf','ft','tt','ff']}}
                            #     )
                            cnt += 1
    data_draw = []
    categories = []
    print(data_source)
    for k1 in data_source:
        categories.append(k1)
        data_draw.append([data_source[k1][t] for t in ['tf','ft','tt','ff']])
    import matplotlib.pyplot as plt
    import numpy as np
    bar_width = 0.5  
    bar_positions = np.arange(len(categories)) 
    color = ['red','green','blue','black','pink', 'purple','orange']
    data_draw = np.array(data_draw).T   # [2,4] -> [4,2]
    for i in range(len(data_draw)):
        if i == 0:
            plt.bar(bar_positions, data_draw[i], color=color[i], edgecolor='white', width=bar_width)
        else:
            plt.bar(bar_positions, data_draw[i], bottom=np.sum(data_draw[0:i],0),color=color[i], edgecolor='white', width=bar_width)

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Stacked Bar Chart with Different Colors')

    plt.xticks(bar_positions, categories)
    plt.legend(['tf','ft','tt','ff'])
    plt.savefig(f'{args.dataset}_conformity_by_trait.png')

def view3(args, all_results) -> list:
    """
    [   
        # repeat 1
        {
            role:{
                strategy:{
                    'answers': [[[a1,a2,a3],[a1,a2,a3], ...], [], []],          # [n_case, n_round, n_agent]
                    'labels': []                                                # [n_case, n_round, n_agent]
                    'same': []                                                  # [n_case, n_round]
                }
            }
        },
        # repeat 2
        {},
        # repeat 3
        {},
        # repeat 4
        {},
        # repeat 5
        {}
    ]
    """
    data_source = {
        decimal_to_binary(i,3):[] for i in range(8)
    }

    for strategy_id in range(8):
        strategy = decimal_to_binary(strategy_id, 3)
        for repeat in range(5):
            for role in range(4):
                # for case_id in range(50):
                    # print(all_results[repeat][role][strategy])
                # data_source[strategy].extend(all_results[repeat][role][strategy]['same'])
                data_source[strategy].extend(all_results[repeat][role][strategy]['cnt of answers'])
    
    import numpy as np
    import matplotlib.pyplot as plt
    data = []
    for key in data_source:
        data.append(data_source[key])
    data = np.array(data)       # [8, *, 4]
    x = np.array([1, 2, 3, 4])
    y = data.mean(1)
    return y.tolist()

def view5(args, all_results) -> list:
    labels = ['tf','ft','tt','ff']
    data_source = {
        '1-0':{
        },
        '2-0':{
        },
        '3-0':{
        }
    }
    aaa = 0
    for strategy_id in range(8):
        strategy = decimal_to_binary(strategy_id, 3)
        for repeat in range(5):
            for case_id in range(50):
                for role in range(4):
                    cnt = 0
                    for pos, thinking_pattern in enumerate(strategy):
                        if thinking_pattern == '0':
                            agent_conformity:list = all_results[repeat][role][strategy]['conformity'][case_id][cnt]
                            agent_type:list = all_results[repeat][role][strategy]['type'][case_id][cnt]
                            if strategy[0:pos+1] not in data_source[f'{pos+1}-0']:
                                data_source[f'{pos+1}-0'].update(
                                    {strategy[0:pos+1]: {t:0 for t in labels}}
                                )
                            for t, c in zip(agent_type, agent_conformity):
                                aaa += 1
                                if t is not None and c:
                                    data_source[f'{pos+1}-0'][strategy[0:pos+1]][t] += 1
                            cnt += 1
    print(aaa)
    data_draw_split = []        # [7,4]  7=[0,00,10,000,010,100,110] 4=[ft,tf,tt,ff]
    categories_split = []       
    print(data_source)
    for k1 in data_source:
        for k2 in data_source[k1]:
            categories_split.append(f'{k1} {k2}')
            data_draw_split.append([data_source[k1][k2][t] for t in labels])
    data_draw_merge = [
        data_draw_split[0], [i+j for i,j in zip(data_draw_split[1], data_draw_split[2])],
        [i+j+k+l for i,j,k,l in zip(data_draw_split[3], data_draw_split[4], data_draw_split[5], data_draw_split[6])]
    ]   # [3,4]
    categories_merge = ['1','2','3']

    print(data_draw_split)
    print(data_draw_merge)
    return data_draw_merge


if __name__ == '__main__':

    data_list = []
    args = parse_agrs()
    repeats = args.repeat
    roles = args.role
    for dataset in ['mmlu','math','chess']:
        args.repeat = repeats
        args.role = roles
        args.dataset = dataset
        check_args(args)
        print(args.dataset)
        repeat = args.repeat
        all_results = []
        for r in repeat:
            args.repeat = r
            print(f"Repeat {r}")
            if args.type.lower() == 'conformity':
                all_results.append(
                    conformity(args)
                )
            else:
                all_results.append(
                    consistent(args)
                )
        

        if args.type.lower() == 'conformity':
            data = view5(args=args, all_results=all_results)
        else:
            data = view3(args=args, all_results=all_results)
        data_list.append(data)
    if args.type.lower() == 'conformity':
        draw_conformity(np.array(data_list)/12000, f'new/{args.experiment_type}_conformity.pdf')
    else:
        draw_consistent(
            {'MMLU':data_list[0], 'MATH':data_list[1], 'Chess Move Validity':data_list[2]},
            f'new/{args.experiment_type}_consistent.pdf'
        )

'''
python conformity_and_consistency.py --experiment_type 'mixtral-main' --type 'consistent'
python conformity_and_consistency.py --experiment_type 'mixtral-main' --type 'conformity'
python conformity_and_consistency.py --experiment_type 'qwen-main' --type 'consistent'
python conformity_and_consistency.py --experiment_type 'qwen-main' --type 'conformity'
python conformity_and_consistency.py --experiment_type 'llama70' --type 'consistent'
python conformity_and_consistency.py --experiment_type 'llama70' --type 'conformity'

'''