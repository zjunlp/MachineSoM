"""评估代码"""
import os

"""
两种评估方法：
1. 就是传统的，按照不同的策略进行
2. 参考PandaLM，两两进行，使用win lose指标
math那个的结果除了提取出来，还要进行降噪，得到真正可比的数值
"""
import networkx as nx
import matplotlib.pyplot as plt
from copy import copy
import argparse
import re
from utils import decimal_to_binary
import pickle
from regenerate import find_invalid_case
import torch

def parse_answer(dataset:str, content:str, task_info:tuple=None):
    def extract_math(string):
        results = []
        stack = []
        flag = r"\boxed{"
        string_copy = copy(string)
        idx = string_copy.find(flag)
        if idx == -1:
            print(f"math parse failed: \"{string}\"")
            return []
            # assert False, f"math parse failed: \"{string}\""
        output_flag = 0
        idx += len(flag)
        while idx < len(string):
            if string[idx] == '{':
                output_flag += 1
            elif string[idx] == '}' and output_flag == 0:
                """弹出"""
                results.append("".join(stack))
                stack.clear()
                """重新找下一个"""
                idx = string_copy.find(flag, idx)
                if idx == -1:
                    """没有找到"""
                    break
                else:
                    idx += len(flag)
                    continue
            elif string[idx] == '}' and output_flag != 0:
                output_flag -= 1
            stack.append(string[idx])
            idx += 1
        return results

    """解析出待求解的值，每个就是agent一次回答"""
    if dataset == "mmlu":
        """和之前一样的方法吧，稍微不同的是，他有些回答没有按照那个指令，因此可以根据选项来确定"""
        # 正则匹配
        assert len(task_info) == 5
        """多选"""
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
        if len(matches) > 1:
            print("mike:",(content,),"parse:", solution_by_re)
        # 选项匹配            A  B  C  D
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
        """按照之前的设置的prompt，应该是'\boxed{}提取出来就行'"""
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
        """如果有final answer，则根据final answer来解析，如果没有则从后往前进行正则匹配"""
        content = content.lower()
        pattern = r"[a-h][1-8]"
        pos = content.rfind("final answer")
        if pos != -1:
            """说明有final answer"""
            item = content.split("final answer")[-1].strip()
            matches = re.findall(pattern, item)
            if len(matches) == 1:
                return matches[0].lower()
            elif len(matches) > 1:
                """有多个可选，选最后一个吧"""
                print(f"****选最后一个{matches[-1].lower()}*****")
                print([content])
                print("*"*100)
                return matches[-1].lower()
            else:
                for valid_case in none_responese:
                    if valid_case in content:
                        return None
                return None
                assert False, f"chess parse failed 1: \"{content}\""
        else:
            """说明没有final answer，直接匹配"""
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

def parse_args():
    parser = argparse.ArgumentParser(description='agent')
    parser.add_argument('--dataset', type=str, default="mmlu")  # ["mmlu", "chess", "math"]
    parser.add_argument('--metric', type=str, default="dag")    # ["dag", "acc"，"token"]
    # =========================================================================================
    parser.add_argument('--repeat', type=int, default=-1)     # -1: 全部 1~3: 表示对应的
    parser.add_argument('--experiment_type', type=str, default="main")  # 实验类型
    parser.add_argument('--turn', type=int, default=3)  # 轮次
    parser.add_argument('--agent', type=int, default=3) # 智能体个数
    parser.add_argument('--role', type=str, default="[0,1,2,3]")   # [0,1,2,3]
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--which_turn', type=int, default=-1)   # 就是选取哪个进行  角色分配(0,1) 问题提问(2,3) 轮数1(4,5) 轮数2(6,7) 轮数3(8,9)
    return parser.parse_args()

def check_args(args):
    assert args.dataset.lower() in ["mmlu","chess","math"], \
        "invalid dataset"
    assert args.metric.lower() in ["dag", "acc", "token"], \
        "invalid metric"
    assert args.repeat == -1 or args.repeat >= 1, \
        "invalid repeat"
    assert isinstance(eval(args.role), list), \
        "invalid role"
    args.role = eval(args.role)
    return args

def load_dataset(args):
    """加载答案文件"""
    # file_name = f"./results/{args.experiment_type}/{args.dataset}_data.pkl"
    file_name = f"./eval_data/{args.dataset}.pkl"
    database = pickle.load(open(file_name, "rb"))
    """
    {
        "task_info": [(), (), ()],
        "answer": [ , , ,],             # chess是个json
        "ratio": ratio,
        # 一个item有多少个元素
        "item_size": len(database[-1]),
        "role": role,
        "sampled_index": sampled_indexes
    }
    """
    return database

def load_agent_file(args):
    """建立索引"""
    def get_map_id(directory):
        """为超长的找到id"""
        """列举所有的文件"""
        files = os.listdir(directory)
        """保留replace和非token的文件 {旧的:新的}"""
        replace_mapping = {}
        for f in files:
            if "token" in f and "replace" in f:
                """3_harmony_3_agents_3_turns_111_strategy_case_56_replace_23_token.pkl"""
                items = f.split("_")
                print(items)
                if int(items[-2]) in replace_mapping:
                    assert replace_mapping[int(items[-2])] == int(items[-4])
                replace_mapping[int(items[-2])] = int(items[-4])
                assert int(items[-2])<args.n_case and int(items[-4]) >= args.n_case
        print(replace_mapping)
        return replace_mapping

    file_mapping = {}
    file_dir = f"./results/{args.experiment_type}/{args.dataset}"
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2**args.turn)]
    if args.repeat != -1:
        """表示不是全部"""
        assert str(args.repeat) in repeat_dir_name
        repeat_dir_name = [str(args.repeat)]
    """寻找当前数据集的不合法case的id"""
    invalid_case_id = []
    if args.repeat not in [4,5]:
        # 第4、5次都是合法的
        invalid_case_id = find_invalid_case(args)
    print(invalid_case_id)
    """不合法的case的id与新的id的建立的索引"""
    replace_mapping = get_map_id(f"{file_dir}/{repeat_dir_name[0]}")
    assert len(invalid_case_id) == len(replace_mapping)
    for repeat in repeat_dir_name:
        """重复次数"""
        for role in args.role:
            """社会个数"""
            for strategy in strategies:
                """策略个数"""
                for case_id in range(args.n_case):
                    """遍历case"""
                    if case_id in invalid_case_id:
                        """需要重新映射"""
                        origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                        origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                        now_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{replace_mapping[case_id]}_replace_{case_id}.pkl"
                        now_full_path = f"{file_dir}/{repeat}/{now_file_name}"
                        file_mapping[origin_full_path] = now_full_path
                    else:
                        file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                        full_path = f"{file_dir}/{repeat}/{file_name}"
                        file_mapping[full_path] = full_path
    return file_mapping

def _most_frequence_answer(answers:list):
    """统计答案出现最多的"""
    # print(List)
    counter = 0
    if answers is None:
        return None
    num = [answers[0]]

    for i in answers:
        """统计答案i出现了多少次"""
        current_frequency = answers.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = [i]
        elif current_frequency == counter:
            num.append(i)

    num = list(set(num))
    if counter == 1:
        """如果最多的只出现了一次，那么说明大家都不一样，算错"""
        return None
    elif len(num) != 1:
        """说明有多个答案，也算错"""
        return None
    else:
        return num[0]

def judge_answer(agent_answer:list, gt_answer, args):
    math_pattern = [
        " \\text{ positive integers, } 12 \\text{ negative integers}", " \\, \\text{cm}", "\\, \\text{cm}",
        "^\\circ", "{\\text{Degree of }} g(x) = ", "°", " \\text{ cm}", "\\text{ cm}", "b = ", "r = ", "x = ", "m+n = ", "\\text{ degrees}", "x + y + z = "
    ]
    def check_math_answer(answer):
        if answer == '{2^3 \\times 3^2 \\times 5 \\times 7}':
            return '2520'
        for _ in math_pattern:
            answer = answer.replace(_, "")
        # answer = answer.replace("^\\circ", "").replace(" \\, \\text{cm}","").replace("{\\text{Degree of }} g(x) = ","")
        answer = answer.replace("{", "").replace("}", "")
        if answer in ["978,121", "99,940,009"]:
            return answer.replace(",", "")
        return answer

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

    """返回0表示错误，返回1表示正确"""
    # if args.dataset == "math":
    #     # _agent_answer = 
    #     agent_final_answer = _most_frequence_answer(answers=[check_math_answer(_) for _ in agent_answer])
    #     print(agent_final_answer)
    # else:
    if args.dataset == "math":
        agent_answer = check_include(agent_answer)
    agent_final_answer = _most_frequence_answer(answers=agent_answer)
    # print(agent_answer)
    # print(agent_final_answer, gt_answer)
    # assert False
    if agent_final_answer is None:
        return 0
    if args.dataset == "chess":
        """gt_answer是一个list"""
        assert isinstance(gt_answer, list)
        print("chess-check:", agent_final_answer)
        print("check-answer:", gt_answer)
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
            print(agent_answer)
            print(agent_final_answer, gt_answer, "True")
            return 1
        else:
            print(agent_answer)
            print(agent_final_answer, gt_answer, "False")
            return 0
    else:
        assert False

def evaluate_in_acc(args):
    """
    采用准确率指标进行计算
    1. 读取答案文件
    2. 遍历生成的文件，然后确定好replace的文件，直接建立一个mapping:{"source":"actual"}这种
    3. 开始评估
          取出一个文件
          对文件的三个agent的回答进行一个解析
          如果三个都不一样算错，或者三个都解析失败也算错，
          对同一类的准确率进行合并
    4. 输出
    :param args:
    :return:
    """
    results_value = []
    results_name = []
    """step-1 加载答案文件"""
    database = load_dataset(args)
    """step-2 变量生成的文件，并建立mapping，{旧的路径:新的路径}，是全路径"""
    file_mapping = load_agent_file(args)
    """step-3 开始评估"""
    # 加载路径
    file_dir = f"./results/{args.experiment_type}/{args.dataset}"
    # 获取重复的文件夹个数
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2 ** args.turn)]
    if args.repeat != -1:
        """表示不是全部"""
        assert str(args.repeat) in repeat_dir_name
        repeat_dir_name = [str(args.repeat)]
    for repeat in repeat_dir_name:
        """重复次数"""
        for role in args.role:
            """社会个数"""
            for strategy in strategies:
                """策略个数"""
                current_acc = 0
                for case_id in range(args.n_case):
                    """得到真实的地址"""
                    origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                    origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                    now_full_path = file_mapping[origin_full_path]
                    """读取文件并进行操作"""
                    try:
                        agent_center = pickle.load(open(now_full_path, "rb"))
                    except:
                        print("shutdown:",now_full_path)
                        current_acc += 0
                        continue
                    """智能体的答案"""
                    agent_answers = []
                    """实际的答案"""
                    gt_answers = database["answer"][case_id]
                    for agent_id in range(args.agent):
                        context = agent_center[agent_id][args.which_turn]
                        assert context["role"] == "assistant"
                        agent_answers.append(
                            parse_answer(dataset=args.dataset, content=context["content"], task_info=database["task_info"][case_id])
                        )
                        # print(context, agent_answers[-1], gt_answers)
                    """解析答案并返回结果"""
                    current_acc += judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, args=args)
                results_name.append({'repeat': repeat, 'role': role, 'strategy': strategy})
                results_value.append(current_acc)
    for i in range(len(results_value)//(2**args.turn)):
        print(results_value[i*(2**args.turn):(i+1)*(2**args.turn)])
        # print(results_name)

def dag_evaluation(matrix):
    """就是做一下差，然后返回即可"""
    relations = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[0]):
            """行向量"""
            delta = matrix[i] - matrix[j]
            tie = len(torch.where(delta==0)[0])
            win = len(torch.where(delta==1)[0])
            lose = len(torch.where(delta==-1)[0])
            if win > lose:
                relations.append((i+1,j+1))
            elif win < lose:
                relations.append((j+1,i+1))
    return relations

def draw(relations, idx, dataset, args):
    # 创建一个有向图对象
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from(list(range(1,2**args.turn+1)))

    # 添加边
    G.add_edges_from(relations)

    # 设置节点标签
    labels = {}
    for i in range(2**args.turn):
        labels[i+1] = decimal_to_binary(i, args.turn)
    print(labels)
    # labels = {'A': 'Node A', 'B': 'Node B', 'C': 'Node C', 'D': 'Node D', 'E': 'Node E', 'F': 'Node F', 'G': 'Node G', 'H': 'Node H'}
    nx.set_node_attributes(G, labels, 'label')

    # 绘制图形
    pos = nx.spring_layout(G)  # 布局算法，这里使用了Spring布局
    nx.draw_networkx(G, pos=pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightblue', node_size=500, font_size=10, edge_color='gray', arrows=True)

    # 显示图形
    plt.axis('off')
    plt.savefig(f'./dag/{dataset}_{idx}.png')
    plt.cla()

def draw_dag(relations, idx, dataset):
    # 创建一个有向图对象
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from(range(1, 9))

    # 添加边
    G.add_edges_from(relations)

    # 绘制图形
    pos = nx.spring_layout(G)  # 布局算法，这里使用了Spring布局
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10,
                     edge_color='gray', arrows=True)

    # 显示图形
    plt.axis('off')
    plt.savefig(f'./dag/{dataset}_{idx}.png')
    plt.cla()

def evaluate_in_dag(args):
    """
    采用偏序图的win-lose来进行，到时候还可以生成偏序图

    应该就是一个两两比较的过程：
        n_exp * n_case 的一个0-1矩阵   1-正确 0-错误
        然后C(n_exp, 2)种情况进行组合
        用win-lose指标，作差，0表示平，1表示赢，
        然后进行分组就行
    :param args:
    :return:
    """
    results_value = []
    results_name = []
    """step-1 加载答案文件"""
    database = load_dataset(args)
    """step-2 变量生成的文件，并建立mapping，{旧的路径:新的路径}，是全路径"""
    file_mapping = load_agent_file(args)
    """step-3 开始评估"""
    # 加载路径
    file_dir = f"./results/{args.experiment_type}/{args.dataset}"
    # 获取重复的文件夹个数
    repeat_dir_name = os.listdir(file_dir)
    strategies = [decimal_to_binary(_, args.turn) for _ in range(2 ** args.turn)]
    if args.repeat != -1:
        """表示不是全部"""
        assert str(args.repeat) in repeat_dir_name
        repeat_dir_name = [str(args.repeat)]
    for repeat in repeat_dir_name:
        """重复次数"""
        for role in args.role:
            """社会个数"""
            for strategy in strategies:
                """策略个数"""
                current_acc = []
                for case_id in range(args.n_case):
                    """得到真实的地址"""
                    origin_file_name = f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}.pkl"
                    origin_full_path = f"{file_dir}/{repeat}/{origin_file_name}"
                    now_full_path = file_mapping[origin_full_path]
                    """读取文件并进行操作"""
                    agent_center = pickle.load(open(now_full_path, "rb"))
                    """智能体的答案"""
                    agent_answers = []
                    """实际的答案"""
                    gt_answers = database["answer"][case_id]
                    for agent_id in range(args.agent):
                        context = agent_center[agent_id][args.which_turn]
                        assert context["role"] == "assistant"
                        agent_answers.append(
                            parse_answer(dataset=args.dataset, content=context["content"],
                                         task_info=database["task_info"][case_id])
                        )
                        # print(context, agent_answers[-1], gt_answers)
                    """解析答案并返回结果"""
                    current_acc.append(judge_answer(agent_answer=agent_answers, gt_answer=gt_answers, args=args))
                # results_name.append({'repeat': repeat, 'role': role, 'strategy': strategy})
                results_value.append(current_acc)

    results_value = torch.as_tensor(results_value)
    for i in range(len(results_value) // (2 ** args.turn)):
        """每组里面有8个进行比较"""
        relations = dag_evaluation(matrix=results_value[i * (2 ** args.turn):(i + 1) * (2 ** args.turn)])
        print(relations)
        draw(relations, i, args.dataset, args)
        # print(results_value[i * (2 ** args.turn):(i + 1) * (2 ** args.turn)])

def evaluate_in_token(args):
    """输出token"""
    pass

def _evaluate():
    """因为前面的所有流程都是一样的，就是最后评估的时候不太一样，因此这边就放在一起，到时候传入一个函数进行即可"""
    

def evaluate(args):
    func_mapping ={
        "dag": evaluate_in_dag,
        "acc": evaluate_in_acc,
        "token": evaluate_in_token
    }
    func_mapping[args.metric](args=args)

def test():
    """解析mmlu"""
    # print(parse_answer(
    #     dataset="mmlu",
    #     task_info=("question", "8.05", "7.6", "3.95", "3.37"),
    #     content="After examining the solutions provided by other agents, I agree with their approach and reasoning. Here is an updated step-by-step analysis:\n\n1. We need to determine the highest amount of rainfall that would place the month among the 10% driest months.\n2. To do this, we calculate the z-score corresponding to the 10th percentile of the normal distribution.\n3. Using a standard normal distribution table or calculator, we find that the z-score for the 10th percentile is approximately -1.28.\n4. Next, we use the formula X = μ + (z * σ) to find the corresponding rainfall value.\n5. Plugging in the given values, we have X = 6 + (-1.28 * 1.6).\n6. Simplifying the equation, X ≈ 3.952 inches.\n7. Comparing the answer choices:\n   A) 8.05\n   B) 7.6\n   C) 3.95 (X)\n   D) 3.37\n8. Based on our calculations, the correct answer is C) 3.95 inches.\n\nSo, the final answer is C) 3.95 inches."))

    """解析chess"""
    # print(
    #     parse_answer(
    #         dataset="chess",
    #         content='After reviewing the valid move justifications provided by other agents and considering my previous answers, I acknowledge that my earlier suggestion of "d7" as the valid destination square was incorrect.\n\nConsidering the assessments from other agents, I agree that the valid destination square for the chess piece at is "d2". This move allows for capturing the opponent\'s piece on h6 (h6h5), which is a valid move according to the given chess game and position.\n\nAfter careful consideration, my final answer is "d2".'
    #     )
    # )

    """解析math"""
    print(
        parse_answer(
            dataset="math",
            content="After carefully considering the feedback and analysis provided by the other agents, I agree with the correction made by Agent 2 regarding the determinant of matrix $\\mathbf{M}$. The correct determinant is $\\boxed{-\\frac{25}{9}}$, as calculated by Agent 1 and Agent 2. I apologize for any confusion caused by my previous response."
        )
    )

    """majority vote"""
    for answers in [[None, None, 1], [1, 1, 2], [1, 2, 3], [1, 1, 2, 2], [None, None, 1, 1]]:
        print(_most_frequence_answer(answers))

if __name__ == '__main__':
    args = parse_args()
    args = check_args(args)
    evaluate(args)
    # test()
"""
python evaluate.py --dataset chess --metric acc --repeat 1
python evaluate.py --dataset math --metric acc --repeat 1
"""