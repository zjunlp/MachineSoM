import time

from utils import AgentDialogManagement, decimal_to_binary, isexist
from tqdm import tqdm
from api import openai_api
import argparse
from prompt import agent_roles_datasets, agent_characters, interaction_prompt
from dataloader import dataloader
import pickle
import random

class helper:
    dataset = None
    prompt = {
        # Please remember it and don't forget it.
        "create_confident": "Imagine you are {} and {}. Please keep this in mind. If you understand please say ok only.",
        "create_temperate": "You are {} and {}. Please keep this in mind. If you understand please say ok only.",
    }

def debate_start(idx: list, agent_center: AgentDialogManagement, task_info):
    """Send the same question to every agent without involving answer concatenation."""
    # template
    content = interaction_prompt[helper.dataset]["question"].format(*task_info)
    if helper.dataset == "math":
        """The "{{" is interpreted as "{", so it needs to be doubled, and the whole issue only occurs when sending the question."""
        content = content.replace("Put your answer in the form \\boxed{answer}", "Put your answer in the form \\boxed{{answer}}")
    for index in idx:
        assert agent_center.agents[index][-1]["role"] == "assistant"
        agent_center.agents[index].append(
            {"role": "user", "content": content}
        )

def debate_next(idx: list, agent_center: AgentDialogManagement, task_info):
    """Loading the agent_center directly will result in an error because at this point "-1" is the user, since they are added sequentially."""
    memory = []
    for cnt, index in enumerate(idx):
        assert agent_center.agents[index][-1][
                   "role"] == "assistant", f"{agent_center.agents[index][-1]['role']}!=assistant"
        """Append the content from other agents to the end."""
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
    """Initialize some parameters"""
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
    '''python two_agent_generate.py --role 2 --dataset chess --repeat 1 --turn 3 --api_idx 0 --api_account mike --experiment_type rebuttal --agent 2 --model llama'''
    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--role', type=int, default=0)      # [0,1,2,3] refers to the number of easy-going agents in a society.
    parser.add_argument('--dataset', type=str, default="mmlu")  # chess math
    parser.add_argument('--repeat', type=int, default=1)    # Start labeling from 1, used to indicate the current repeat experiment number; this parameter is also used for saving purposes.
    parser.add_argument('--turn', type=int, default=3)      # collaboration round
    parser.add_argument('--api_idx', type=int, default=0)   # start from 0
    parser.add_argument('--api_account', type=str, default='taobao')    # which account
    parser.add_argument('--experiment_type', type=str, default="main")  # experiment type, also used for saving
    parser.add_argument('--strategy_id', type=str, default="000")       # strategy_id
    parser.add_argument('--case_id', type=str, default=None)            # case_id
    # ======================================================================
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--agent', type=int, default=3)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()

def args_check(args):
    assert args.role >= 0 and args.role <= 3
    assert args.dataset in ["mmlu","math","chess"]
    assert args.turn >= 2
    assert args.api_idx >=0
    args.case_id = eval(args.case_id)
    print("*"*10, f"  setting: {args.experiment_type}  ", "*"*10)
    print(f"1. dataset: {args.dataset}\tRepeat: {args.repeat}")
    print(f"2. society: {args.role} Harmony\tCollaboration Round: {args.turn}")
    print(f"3. number of agents: {args.agent}\tAPI: {args.api_idx}")
    print(f"4. api account: {args.api_account}")
    print(f"5. number of cases: {args.n_case}")
    print(f"6. strategies: {args.strategy_id}")
    print(f"7. cases: {args.case_id}")
    print("*" * 10, f"{time.ctime()}", "*" * 10)

def create_configs(args):
    _print("creating agent configs ......")
    dataset = args.dataset
    turn = args.turn
    """role play"""
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
    rounds_configs = []
    
    situation = args.strategy_id
    rounds_configs.append(
        [{
            "debate": {"idx": [0, 1, 2], "fn": "start"},
            "reflection": {"idx": [], "fn": None},
            "wait": {"idx": [], "fn": ""}
        }]
    )
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

def simulate(key, args, agent_config, round_config):
    def _dynamic_agent_roles(agent_config, data_loader: dataloader, args, idx):
        if args.dataset == "mmlu":
            for i in range(len(agent_config)):
                agent_config[i]["role"] = data_loader.database["role"][idx]
        return agent_config

    data_loader = dataloader(name=args.dataset, n_case=args.n_case)
    for case_id in args.case_id:
        file_save_names = f"{args.save_path}_case_{case_id}.pkl"
        shut_file_save_names = f"{args.save_path}_case_{case_id}_shutdown.pkl"
        if isexist(file_save_names):
            print(f"skip `{file_save_names}` because of existence.")
            continue
        if isexist(shut_file_save_names):
            pass
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=args.agent,
            default_model=args.model,
            API_KEY=key,
            llama_api=key
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
    agents_configs, rounds_configs = create_configs(args)
    cur_agent_config = agents_configs[f"{args.role}_harmony"]
    key = openai_api[args.api_account][args.api_idx]
    strategy_labels = [decimal_to_binary(i, args.turn) for i in range(2**args.turn)]
    strategy_labels = args.strategy_id
    for idx, cur_round_config in enumerate(rounds_configs):
        args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{args.repeat}/" \
                         f"{args.role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy_labels}_strategy"
        simulate(key=key, args=args, agent_config=cur_agent_config, round_config=cur_round_config)

def read():
    file_name = ""
    contexts = pickle.load(open(file_name, "rb"))
    print(contexts)

if __name__ == '__main__':
    main()