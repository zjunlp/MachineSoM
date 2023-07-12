"""
需要适应不同的数据集
"""
import time

from utils import AgentDialogManagement, decimal_to_binary
from tqdm import tqdm
from api import openai_api
import argparse
from prompt import agent_roles_datasets, agent_characters, interaction_prompt
from dataloader import dataloader
import pickle

class helper:
    dataset = None
    prompt = {
        # 刚创建时用于指定agent的性格
        # Please remember it and don't forget it.
        "create_confident": "Imagine you are {} and {}. Please keep this in mind. If you understand please say ok only.",
        "create_temperate": "You are {} and {}. Please keep this in mind. If you understand please say ok only.",
    }

def debate_start(idx: list, agent_center: AgentDialogManagement, task_info):
    """给每一个agent发送同一个问题，不涉及到答案拼接"""
    # template
    content = interaction_prompt[helper.dataset]["question"].format(*task_info)
    if helper.dataset == "math":
        """{{被解析为->{，因此需要变成两个，且整个只在发送问题的时候会产生"""
        content = content.replace("Put your answer in the form \\boxed{answer}", "Put your answer in the form \\boxed{{answer}}")
    for index in idx:
        assert agent_center.agents[index][-1]["role"] == "assistant"
        agent_center.agents[index].append(
            {"role": "user", "content": content}
        )

def debate_next(idx: list, agent_center: AgentDialogManagement, task_info):
    """需要传入"""
    """直接加载agent_center会出错，因为此时的-1就是user，因为是顺序添加的"""
    memory = []
    for cnt, index in enumerate(idx):
        assert agent_center.agents[index][-1][
                   "role"] == "assistant", f"{agent_center.agents[index][-1]['role']}!=assistant"
        """将其它agent的内容添加到结尾"""
        other_index = idx[0:cnt] + idx[cnt + 1:]
        # template
        content = interaction_prompt[helper.dataset]["debate"][0]
        # content = "These are the solutions to the problem from other agents:"
        for _index in other_index:
            assert agent_center.agents[_index][-1]["role"] == "assistant"
            agent_response = agent_center.agents[_index][-1]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)
            content = content + response
        # template
        content = content + interaction_prompt[helper.dataset]["debate"][1]
        # content = content + "\n\nUsing the reasoning from other agents as addtitional advice and referring to your historical answers, can you give an updated answer? Check your historical answers and the answers from other agents, and confirm your answer starting with \"The answer is $Your Answer$\" at the end."
        # agent_center.agents[index].append(
        #     {"role": "user", "content": content}
        # )
        memory.append({"role": "user", "content": content})
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append(memory[cnt])

def debate_final(idx: list, agent_center: AgentDialogManagement, task_info):
    """需要更新"""
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "debate final"})

def reflection_start(idx: list, agent_center: AgentDialogManagement, task_info):
    """需要更新"""
    for cnt, index in enumerate(idx):
        # template
        agent_center.agents[index].append({
            "role": "user",
            "content": interaction_prompt[helper.dataset]["reflection"]
            # "content": "Can you double check that your answer is correct? Confirm your answer starting with \"The answer is $Your Answer$\" at the end of your response."
        })

def reflection_feedback(idx: list, agent_center: AgentDialogManagement, task_info):
    """需要更新"""
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "reflection feedback"})

def reflection_refine(idx: list, agent_center: AgentDialogManagement, task_info):
    """需要更新"""
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "reflection refine"})

def init(args):
    """对一些参数进行一个初始化"""
    helper.dataset = args.dataset
    helper.prompt["debate"] = {
        "start": debate_start,
        "next": debate_next,
        "final": debate_final
    }
    helper.prompt["reflection"] = {
        "start": reflection_start,
        "feedback": reflection_feedback,
        "refine": reflection_refine,
    }

def _print(message):
    print(f"[{time.ctime()}] {message}")

def parse_args():
    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--role', type=int, default=0)   # [0,1,2,3] 指的是一个社会当中harmony的个数
    parser.add_argument('--dataset', type=str, default="mmlu")  # chess math
    parser.add_argument('--repeat', type=int, default=1)    # 从1开始标记，用于指示是第几次重复实验，且用于保存
    parser.add_argument('--turn', type=int, default=3)      # 几轮
    parser.add_argument('--api_idx', type=int, default=0)   # 从0开始
    parser.add_argument('--api_account', type=str, default=None)    # 用哪个账号下面的api
    parser.add_argument('--experiment_type', type=str, default="main")  # 实验类型
    # ======================================================================
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--agent', type=int, default=3)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()

def args_check(args):
    assert args.role >= 0 and args.role <= 3, "社会参数输入错误"
    assert args.dataset in ["mmlu","math","chess"], "数据集参数输入错误"
    assert args.turn >= 2, "轮数参数输入错误"
    assert args.api_idx >=0, "api参数输入错误"
    print("*"*10, f"  {args.experiment_type}实验设定  ", "*"*10)
    print(f"1. 数据集: {args.dataset}\t第{args.repeat}次重复实验")
    print(f"2. 社会类型: {args.role} Harmony\t对话轮数: {args.turn}轮")
    print(f"3. 智能体个数: {args.agent}\tAPI: {args.api_idx}")
    print(f"4. 使用{args.api_account}用户的API")
    print(f"5. 跑{args.n_case}个样本")
    print("*" * 10, f"{time.ctime()}", "*" * 10)

def create_configs(args):
    _print("创建配置文件......")
    dataset = args.dataset
    turn = args.turn
    """不同数据集中可能扮演不同的角色，比如棋手、数学家等"""
    agent_roles = agent_roles_datasets[dataset]
    agents_configs = {
        "3_harmony": [{"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                      {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                      {"role": agent_roles["expert"], "character": agent_characters["temperate"]}],
        "2_harmony": [{"role": agent_roles["expert"], "character": agent_characters["confident"]},
                      {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                      {"role": agent_roles["expert"], "character": agent_characters["temperate"]}],
        "1_harmony": [{"role": agent_roles["expert"], "character": agent_characters["confident"]},
                      {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                      {"role": agent_roles["expert"], "character": agent_characters["temperate"]}],
        "0_harmony": [{"role": agent_roles["expert"], "character": agent_characters["confident"]},
                      {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                      {"role": agent_roles["expert"], "character": agent_characters["confident"]}],
    }
    """创建交互策略配置文件"""
    _print(f"共有{2**turn}个交互策略")
    rounds_configs = []
    for i in range(0, 2**turn):
        situation = decimal_to_binary(i, turn)
        """除去角色定义，这个就是向每个agent分发问题"""
        rounds_configs.append(
            [{
                "debate": {"idx": [0, 1, 2], "fn": "start"},
                "reflection": {"idx": [], "fn": None},
                "wait": {"idx": [], "fn": ""}
            }]
        )
        """每一位进行遍历"""
        for _ in situation:
            if _ == '1':
                """反思"""
                rounds_configs[-1].append({
                    "debate": {"idx": [], "fn": None},
                    "reflection": {"idx": [0, 1, 2], "fn": "start"},
                    "wait": {"idx": [], "fn": ""}
                })
            elif _ == '0':
                """辩论"""
                rounds_configs[-1].append({
                    "debate": {"idx": [0, 1, 2], "fn": "next"},
                    "reflection": {"idx": [], "fn": None},
                     "wait": {"idx": [], "fn": ""}
                })
            else:
                assert False, "Error!"

    return agents_configs, rounds_configs

def simulate(key, args, agent_config, round_config):
    def _dynamic_agent_roles(agent_config, data_loader: dataloader, args, idx):
        if args.dataset == "mmlu":
            for i in range(len(agent_config)):
                agent_config[i]["role"] = data_loader.database["role"][idx]
        return agent_config

    data_loader = dataloader(name=args.dataset, n_case=args.n_case)
    for case_id in range(args.n_case):
        # 对每个case进行一个模拟
        """agent管理中心"""
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=args.agent,
            default_model=args.model,
            API_KEY=key,
        )
        # 创建角色
        agent_center.generate_agents(agent_config=agent_config)
        # agent_center.generate_agents(agent_config=_dynamic_agent_roles(agent_config, data_loader, args, case_id))
        # 发送并确认
        agent_center.parse_message(
            idx="all",
            memory=agent_center.send_message(
                idx="all"
            )
        )
        print(agent_center.agents)
        # 取出case
        item = data_loader[case_id]
        print("item:", item)
        # 开始对话
        """用于标记对话是否没有超出最大长度"""
        FLAG_NORMAL = True
        """+1是因为要算上question"""
        for round_index in tqdm(range(args.turn+1)):
            """准备好用户的信息"""
            agent_center.prepare_for_message(
                round_config=round_config[round_index],
                task_info=item
            )
            """对非wait的索引进行拼接"""
            idx = []
            idx.extend(round_config[round_index]["debate"]["idx"])
            idx.extend(round_config[round_index]["reflection"]["idx"])
            """发送信息"""
            memory = agent_center.send_message(idx=idx)
            if memory is None:
                """表示超出最大长度，结束当前case"""
                FLAG_NORMAL = False
                break
            agent_center.parse_message(
                idx=idx,
                memory=memory
            )
        # 保存对话记录
        if FLAG_NORMAL:
            agent_center.save(path=f"{args.save_path}_case_{case_id}")
        else:
            agent_center.save(path=f"{args.save_path}_case_{case_id}_shutdown")

def main():
    args = parse_args()
    args_check(args)
    init(args)
    # step-1 创建智能体的配置和交互策略的配置
    agents_configs, rounds_configs = create_configs(args)
    # step-2 根据传入的参数进行配置，包括社会的选择、API的维护
    cur_agent_config = agents_configs[f"{args.role}_harmony"]
    key = openai_api[args.api_account][args.api_idx]
    _print(f"使用的key为{key}")
    # step-3 按照策略进行遍历
    """当前是哪个策略"""
    strategy_labels = [decimal_to_binary(i, args.turn) for i in range(2**args.turn)]
    for idx, cur_round_config in enumerate(rounds_configs):
        args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{args.repeat}/" \
                         f"{args.role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy_labels[idx]}_strategy"
        simulate(key=key, args=args, agent_config=cur_agent_config, round_config=cur_round_config)

def read():
    file_name = ""
    contexts = pickle.load(open(file_name, "rb"))
    print(contexts)

if __name__ == '__main__':
    main()