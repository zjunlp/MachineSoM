import os
import time

from utils import AgentDialogManagement, decimal_to_binary
from tqdm import tqdm
import random
from api import openai_api
import argparse
from prompt import agent_roles_datasets, agent_characters, interaction_prompt
from dataloader import dataloader
import pickle
import numpy as np

class helper:
    dataset = None
    prompt = {
        "create_confident": "Imagine you are {} and {}. Please keep this in mind. If you understand please say ok only.",
        "create_temperate": "You are {} and {}. Please keep this in mind. If you understand please say ok only.",
    }

def debate_start(idx: list, agent_center: AgentDialogManagement, task_info):
    # template
    content = interaction_prompt[helper.dataset]["question"].format(*task_info)
    if helper.dataset == "math":
        content = content.replace("Put your answer in the form \\boxed{answer}", "Put your answer in the form \\boxed{{answer}}")
    for index in idx:
        assert agent_center.agents[index][-1]["role"] == "assistant"
        agent_center.agents[index].append(
            {"role": "user", "content": content}
        )

def debate_next(idx: list, agent_center: AgentDialogManagement, task_info):
    memory = []
    for cnt, index in enumerate(idx):
        assert agent_center.agents[index][-1][
                   "role"] == "assistant", f"{agent_center.agents[index][-1]['role']}!=assistant"
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
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "debate final"})

def reflection_start(idx: list, agent_center: AgentDialogManagement, task_info):
    for cnt, index in enumerate(idx):
        # template
        agent_center.agents[index].append({
            "role": "user",
            "content": interaction_prompt[helper.dataset]["reflection"]
            # "content": "Can you double check that your answer is correct? Confirm your answer starting with \"The answer is $Your Answer$\" at the end of your response."
        })

def reflection_feedback(idx: list, agent_center: AgentDialogManagement, task_info):
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "reflection feedback"})

def reflection_refine(idx: list, agent_center: AgentDialogManagement, task_info):
    for cnt, index in enumerate(idx):
        agent_center.agents[index].append({"role": "user", "content": "reflection refine"})

def init(args):
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
    parser.add_argument('--role', type=int, default=0)                  # [0,1,2,3] refers to the number of harmonies in a society.
    parser.add_argument('--dataset', type=str, default="mmlu")          # chess math
    parser.add_argument('--repeat', type=int, default=1)                # Labeling from 1 was used to indicate which was the first replicate of the experiment and was used to save the
    parser.add_argument('--turn', type=int, default=3)                  # rounds
    parser.add_argument('--api_idx', type=int, default=0)               # start index is 0, which api
    parser.add_argument('--api_account', type=str, default=None)        # which account
    parser.add_argument('--experiment_type', type=str, default="main")  # experiment type.  turn, agent, strategy
    # ======================================================================
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--agent', type=int, default=3)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()

def args_check(args):
    assert args.role >= 0 and args.role <= 3, "role error!"
    assert args.dataset in ["mmlu","math","chess"], "dataset error!"
    assert args.turn >= 2, "turn error!"
    assert args.api_idx >=0, "api index error!"
    print("*"*10, f"  {args.experiment_type} Experimental Type ", "*"*10)
    print(f"1. datasets: {args.dataset}\tRepeat:{args.repeat}")
    print(f"2. society: {args.role} Harmony\tRounds: {args.turn}")
    print(f"3. the number of agents: {args.agent}\tAPI: {args.api_idx}")
    print(f"4. api account:{args.api_account}")
    print(f"5. {args.n_case} cases")
    print("*" * 10, f"{time.ctime()}", "*" * 10)

def create_configs(args):
    dataset = args.dataset
    turn = args.turn
    agent_roles = agent_roles_datasets[dataset]
    if args.agent == 2:
        agents_configs = {
            "2_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]}
            ],
            "1_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
            ],
            "0_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
            ]
        }
    elif args.agent == 3:
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
    elif args.agent == 4:
        agents_configs = {
            "4_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
            ],
            "3_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
            ],
            "2_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
            ],
            "1_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["temperate"]},
            ],
            "0_harmony": [
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
                {"role": agent_roles["expert"], "character": agent_characters["confident"]},
            ]
        }
    else:
        assert False
    rounds_configs = []
    if args.experiment_type != "strategy":
        for i in range(0, 2**turn):
            situation = decimal_to_binary(i, turn)
            rounds_configs.append(
                [{
                    "debate": {"idx": list(range(args.agent)), "fn": "start"},
                    "reflection": {"idx": [], "fn": None},
                    "wait": {"idx": [], "fn": ""}
                }]
            )
            for _ in situation:
                if _ == '1':
                    """debate"""
                    rounds_configs[-1].append({
                        "debate": {"idx": [], "fn": None},
                        "reflection": {"idx": list(range(args.agent)), "fn": "start"},
                        "wait": {"idx": [], "fn": ""}
                    })
                elif _ == '0':
                    """reflection"""
                    rounds_configs[-1].append({
                        "debate": {"idx": list(range(args.agent)), "fn": "next"},
                        "reflection": {"idx": [], "fn": None},
                        "wait": {"idx": [], "fn": ""}
                    })
                else:
                    assert False, "Error!"
    else:
        random.seed(0)
        for i in range(0, 2**turn):
            situation = decimal_to_binary(i, turn)
            rounds_configs.append(
                [{
                    "debate": {"idx": list(range(args.agent)), "fn": "start"},
                    "reflection": {"idx": [], "fn": None},
                    "wait": {"idx": [], "fn": ""}
                }]
            )
            """每一位进行遍历"""
            for _ in situation:
                if _ == '1':
                    # reflect = random.choice(list(range(args.agent)))
                    rounds_configs[-1].append(
                        {
                            "debate": {"idx": [], "fn": None}, 
                            "reflection": {"idx": list(range(args.agent)), "fn": "start"}, 
                            "wait": {"idx": [], "fn": ""}
                        }
                    )
                elif _ == '0':
                    reflect = random.choice(list(range(args.agent)))
                    rounds_configs[-1].append(
                        {
                            "debate": {"idx": list(set(range(args.agent))-set([reflect])), "fn": "next"}, 
                            "reflection": {"idx": [reflect], "fn": "start"}, 
                            "wait": {"idx": [], "fn": ""}}
                    )

    return agents_configs, rounds_configs

def incremental_simulate(key, args, round_config):
    def judge_valid_strategy(s):
        for i in s:
            assert i in ['0', '1']
        return s
    _file_names = os.listdir(f"./results/main/{args.dataset}/{args.repeat}/")
    origin_strategy_length = 0
    for file_name in _file_names:
        for turn in range(args.turn,0,-1):
            if f"{args.role}_harmony_{args.agent}_agents_{turn}_turns_" in file_name:
                origin_strategy_length = len(judge_valid_strategy(file_name.split("_")[6]))
                break
    now_strategy = judge_valid_strategy(args.save_path.split("_")[6])
    now_strategy_length = len(judge_valid_strategy(args.save_path.split("_")[6]))
    assert now_strategy_length > origin_strategy_length

    data_loader = dataloader(name=args.dataset, n_case=args.n_case)
    
    if args.dataset == "mmlu":
        new_case_id = {9:50, 11:51, 13:52, 31:53, 35:54, 49:55}
    for case_id in range(args.n_case):
        prefix = f"./results/main/{args.dataset}/{args.repeat}/{args.role}_harmony_{args.agent}_agents_{origin_strategy_length}_turns_{now_strategy[0:origin_strategy_length]}_strategy"
        if args.dataset == "mmlu" and case_id in new_case_id and args.repeat in [1,2,3]:
            print("replace:", f"{prefix}_case_{new_case_id[case_id]}_replace_{case_id}.pkl")
            history_agent = pickle.load(open(f"{prefix}_case_{new_case_id[case_id]}_replace_{case_id}.pkl","rb"))
            history_token = pickle.load(open(f"{prefix}_case_{new_case_id[case_id]}_replace_{case_id}_token.pkl","rb"))
        else:
            history_agent = pickle.load(open(f"{prefix}_case_{case_id}.pkl","rb"))
            history_token = pickle.load(open(f"{prefix}_case_{case_id}_token.pkl","rb"))
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=args.agent,
            default_model=args.model,
            API_KEY=key
        )
        agent_center.agents = history_agent
        agent_center.tokens = history_token
        item = data_loader[case_id]
        FLAG_NORMAL = True
        for round_index in tqdm(range(args.turn + 1)):
            if round_index <= origin_strategy_length:
                continue
            agent_center.prepare_for_message(
                round_config=round_config[round_index],
                task_info=item
            )
            idx = []
            idx.extend(round_config[round_index]["debate"]["idx"])
            idx.extend(round_config[round_index]["reflection"]["idx"])
            memory = agent_center.send_message(idx=idx)
            if memory is None:
                for _ in range(10):
                    for i in range(args.agent):
                        agent_center.agents[i] = agent_center.agents[i][:-1]
                    agent_center.prepare_for_message(
                        round_config=round_config[round_index],
                        task_info=item
                    )
                    idx = []
                    idx.extend(round_config[round_index]["debate"]["idx"])
                    idx.extend(round_config[round_index]["reflection"]["idx"])
                    memory = agent_center.send_message(idx=idx)
                    if memory is not None:
                        break
                if memory is None:
                    FLAG_NORMAL = False
                    break
            agent_center.parse_message(
                idx=idx,
                memory=memory
            )
        if FLAG_NORMAL:
            assert len(agent_center.agents[0]) == (args.turn + 2) * 2
            agent_center.save(path=f"{args.save_path}_case_{case_id}")
        else:
            agent_center.save(path=f"{args.save_path}_case_{case_id}_shutdown")

def simulate(key, args, agent_config, round_config):
    def _dynamic_agent_roles(agent_config, data_loader: dataloader, args, idx):
        if args.dataset == "mmlu":
            for i in range(len(agent_config)):
                agent_config[i]["role"] = data_loader.database["role"][idx]
        return agent_config

    data_loader = dataloader(name=args.dataset, n_case=args.n_case)
    for case_id in range(args.n_case):
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=args.agent,
            default_model=args.model,
            API_KEY=key,
        )
        agent_center.generate_agents(agent_config=agent_config)
        agent_center.parse_message(
            idx="all",
            memory=agent_center.send_message(
                idx="all"
            )
        )
        item = data_loader[case_id]
        FLAG_NORMAL = True
        for round_index in tqdm(range(args.turn+1)):
            agent_center.prepare_for_message(
                round_config=round_config[round_index],
                task_info=item
            )
            idx = []
            idx.extend(round_config[round_index]["debate"]["idx"])
            idx.extend(round_config[round_index]["reflection"]["idx"])
            memory = agent_center.send_message(idx=idx)
            if memory is None:
                FLAG_NORMAL = False
                break
            agent_center.parse_message(
                idx=idx,
                memory=memory
            )
        if FLAG_NORMAL:
            agent_center.save(path=f"{args.save_path}_case_{case_id}")
        else:
            agent_center.save(path=f"{args.save_path}_case_{case_id}_shutdown")

def main():
    args = parse_args()
    args_check(args)
    init(args)
    # step-1 create config
    agents_configs, rounds_configs = create_configs(args)
    # step-2 prompt and api
    cur_agent_config = agents_configs[f"{args.role}_harmony"]
    key = openai_api[args.api_account][args.api_idx]
    # step-3 run
    strategy_labels = [decimal_to_binary(i, args.turn) for i in range(2**args.turn)]
    for idx, cur_round_config in enumerate(rounds_configs):
        args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{args.repeat}/" \
                         f"{args.role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy_labels[idx]}_strategy"
        if args.experiment_type == "turn":
            incremental_simulate(key=key, args=args, round_config=cur_round_config)
        else:
            simulate(key=key, args=args, agent_config=cur_agent_config, round_config=cur_round_config)

def read():
    file_name = ""
    contexts = pickle.load(open(file_name, "rb"))
    print(contexts)

if __name__ == '__main__':
    main()