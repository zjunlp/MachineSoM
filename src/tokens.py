"""计算token的数量"""
"""
输入:
    repeat: 列表类型，用于指定哪些重复的实验需要被计算
    agent: 智能体个数，默认为3
    turn: 轮次，默认为3
    dataset: 数据集，默认为chess
    experiment_type: 实验类型，默认为main
输出:
    repeat指定的实验的token的平均数量

输入用命令行传入
"""
import argparse
import pickle
import os
from utils import decimal_to_binary
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="计算token的数量")
    """所有的list，如果传入的是-1，就表示全部"""
    parser.add_argument("--repeat", type=str, default='[2, 3, 4, 5]', help="重复次数")
    parser.add_argument("--agent", type=int, default=3, help="agent数量")
    parser.add_argument("--turn", type=int, default=3, help="轮次")
    parser.add_argument("--dataset", type=str, default="chess", help="数据集")
    parser.add_argument("--experiment_type", type=str, default="main")
    parser.add_argument("--role", type=str, default="[-1]")
    parser.add_argument("--strategy", type=str, default="[-1]")
    parser.add_argument("--n_case", type=int, default=50)
    return parser.parse_args()

def load_data(args, repeat, role, strategy, case_id):
    """加载数据"""
    file_name = f"./results/{args.experiment_type}/{args.dataset}/{repeat}/{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy_case_{case_id}_token.pkl"
    if not os.path.exists(file_name):
        """说明是被shutdown的，所以要去找replace文件"""
        files = os.listdir(f"./results/{args.experiment_type}/{args.dataset}/{repeat}/")
        for file in files:
            if f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy}_strategy" in file and f"replace_{case_id}" in file and "token" in file and "shutdown" not in file:
                file_name = f"./results/{args.experiment_type}/{args.dataset}/{repeat}/" + file
                break
    assert os.path.exists(file_name), f"{file_name}"
    consume = 0
    data = pickle.load(open(file_name, "rb"))
    for a in range(args.agent):
        consume += data[a][-1]["total_tokens"]
    return consume

def main(args):
    repeat_list = eval(args.repeat)
    role_list = eval(args.role)
    if role_list[0] == -1:
        role_list = [i for i in range(0, args.agent + 1)]
    strategy_list = eval(args.strategy)
    if strategy_list[0] == -1:
        strategy_list = [decimal_to_binary(_, args.turn) for _ in range(2**args.turn)]
    total_tokens = []

    print(
        "repeat:", repeat_list,
        "\nrole:", role_list,
        "\nstrategy:", strategy_list,
    )

    for repeat in repeat_list:
        for role in role_list:
            for strategy in strategy_list:
                for case_id in range(args.n_case):
                    total_tokens.append(load_data(args, repeat, role, strategy, case_id))

    print("total_tokens:", sum(total_tokens)/len(total_tokens))

if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
# 按照社会
python tokens.py --repeat [1,2,3] --dataset math --role [0]
python tokens.py --repeat [1,2,3] --dataset math --role [1]
python tokens.py --repeat [1,2,3] --dataset math --role [2]
python tokens.py --repeat [1,2,3] --dataset math --role [3]

python tokens.py --repeat [1,2,4] --dataset chess --role [0]
python tokens.py --repeat [1,2,4] --dataset chess --role [1]
python tokens.py --repeat [1,2,4] --dataset chess --role [2]
python tokens.py --repeat [1,2,4] --dataset chess --role [3]

python tokens.py --repeat [2,3,5] --dataset mmlu --role [0]
python tokens.py --repeat [2,3,5] --dataset mmlu --role [1]
python tokens.py --repeat [2,3,5] --dataset mmlu --role [2]
python tokens.py --repeat [2,3,5] --dataset mmlu --role [3]

# 按照策略
python tokens.py --repeat [1,2,3] --dataset math --strategy "['000']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['001']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['010']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['011']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['100']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['101']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['110']"
python tokens.py --repeat [1,2,3] --dataset math --strategy "['111']"

python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['000']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['001']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['010']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['011']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['100']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['101']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['110']"
python tokens.py --repeat [2,3,5] --dataset mmlu --strategy "['111']"

python tokens.py --repeat [1,2,4] --dataset chess --strategy "['000']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['001']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['010']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['011']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['100']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['101']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['110']"
python tokens.py --repeat [1,2,4] --dataset chess --strategy "['111']"

"""