import os
import argparse
from tqdm import tqdm
import time
from utils import decimal_to_binary
from dataloader import dataloader
from api import openai_api
from utils import AgentDialogManagement
import pickle
from prompt import agent_roles_datasets, agent_characters, interaction_prompt

save_dir = "./results/main"

class helper:
    dataset = None
    prompt = {
        # Please remember it and don't forget it.
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

def create_configs(args):
    dataset = args.dataset
    turn = args.turn
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
    for i in range(0, 2**turn):
        situation = decimal_to_binary(i, turn)
        rounds_configs.append(
            [{
                "debate": {"idx": [0, 1, 2], "fn": "start"},
                "reflection": {"idx": [], "fn": None},
                "wait": {"idx": [], "fn": ""}
            }]
        )
        for _ in situation:
            if _ == '1':
                rounds_configs[-1].append({
                    "debate": {"idx": [], "fn": None},
                    "reflection": {"idx": [0, 1, 2], "fn": "start"},
                    "wait": {"idx": [], "fn": ""}
                })
            elif _ == '0':
                rounds_configs[-1].append({
                    "debate": {"idx": [0, 1, 2], "fn": "next"},
                    "reflection": {"idx": [], "fn": None},
                     "wait": {"idx": [], "fn": ""}
                })
            else:
                assert False, "Error!"

    return agents_configs, rounds_configs

def simulate(key, args, agent_config, round_config, invalid_case_id, candidate_case, data_loader:dataloader):
    cursor = [0 for _ in range(len(data_loader.database["ratio"]))]
    for case_id in range(len(invalid_case_id)):
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
        print(agent_center.agents)
        group_id = data_loader.parse_group(invalid_case_id[case_id])
        item = candidate_case[group_id][cursor[group_id]]["task_info"]
        cursor[group_id] += 1
        print("item:", item)
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
            agent_center.save(path=f"{args.save_path}_case_{case_id+args.n_case}_replace_{invalid_case_id[case_id]}")
        else:
            agent_center.save(path=f"{args.save_path}_case_{case_id+args.n_case}_replace_{invalid_case_id[case_id]}_shutdown")

def parse_args():
    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--dataset', type=str, default="mmlu") 
    parser.add_argument('--total_role', type=int, default=4)  
    parser.add_argument('--total_repeat', type=int, default=3) 
    parser.add_argument('--turn', type=int, default=3) 
    parser.add_argument('--api_idx', type=int, default=0)  
    parser.add_argument('--api_account', type=str, default=None) 
    parser.add_argument('--experiment_type', type=str, default="main") 
    # ==============================================================
    parser.add_argument('--n_case', type=int, default=50)
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--agent', type=int, default=3)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()

def check_args(args):
    assert args.dataset.lower() in ["mmlu", "math", "chess"]

def find_invalid_case(args):
    def _parse_case_id(file_name:str) -> int:
        items = file_name.split("_")
        return int(items[-2])
    dataset = args.dataset
    repeat = len(os.listdir(f"{save_dir}/{dataset}"))
    invalid_case = []
    for r in range(1, repeat+1):
        dir_name = f"{save_dir}/{dataset}/{r}"
        files_list = os.listdir(dir_name)
        filter_files_list = []
        for item in files_list:
            if "token.pkl" not in item and "shutdown.pkl" in item:
                filter_files_list.append(item)
        for item in filter_files_list:
            invalid_case.append(_parse_case_id(item))
    return sorted(list(set(invalid_case)))

def generate_case(args, invalid_case_id: list, num):
    data_loader = dataloader(name=args.dataset)
    """
    [
        [{"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, ...],
        [{"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, ...],
    ]
    """
    candidate = data_loader.regenerate(invalid_case_id, num=num)
    return candidate, data_loader

def merge_dataset(args, data_loader:dataloader, invalid_case_id, candidate_case):
    database = data_loader.database.copy()
    cursor = [0 for _ in range(len(data_loader.database["ratio"]))]
    for case_id in invalid_case_id:
        group = data_loader.parse_group(case_id)
        database["task_info"][case_id] = \
            candidate_case[group][cursor[group]]["task_info"]
        database["answer"][case_id] = \
            candidate_case[group][cursor[group]]["answer"]
        cursor[group] += 1
    save_name = f"./results/{args.experiment_type}/{args.dataset}_data.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(database, f)

def regenerate(args):
    invalid_case_id = find_invalid_case(args)
    if len(invalid_case_id) == 0:
        data_loader = dataloader(name=args.dataset)
        save_name = f"./results/{args.experiment_type}/{args.dataset}_data.pkl"
        with open(save_name, "wb") as f:
            pickle.dump(data_loader.database, f)
        return
    candidate_case, data_loader = generate_case(args, invalid_case_id, num=3)
    agents_configs, rounds_configs = create_configs(args)
    key = openai_api[args.api_account][args.api_idx]
    strategy_labels = [decimal_to_binary(i, args.turn) for i in range(2 ** args.turn)]
    for r in range(1, args.total_repeat+1):
        for role in range(args.total_role):
            cur_agent_config = agents_configs[f"{role}_harmony"]
            for idx, cur_round_config in enumerate(rounds_configs):
                args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{r}/" \
                                 f"{role}_harmony_{args.agent}_agents_{args.turn}_turns_{strategy_labels[idx]}_strategy"
                simulate(
                    key=key, args=args, agent_config=cur_agent_config, round_config=cur_round_config,
                    invalid_case_id=invalid_case_id, candidate_case=candidate_case, data_loader=data_loader
                )
    merge_dataset(
        args, data_loader, invalid_case_id, candidate_case
    )

def test(args):
    print(
        generate_case(args, find_invalid_case(args))
    )

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    init(args)
    regenerate(args)

"""
nohup python -u regenerate.py --dataset math --api_idx 0 --api_account gpttest1> math_regenerate.txt 2>&1 & 
nohup python -u regenerate.py --dataset mmlu --api_idx 0 --api_account gpttest2> mmlu_regenerate.txt 2>&1 &
"""
