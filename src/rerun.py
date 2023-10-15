import argparse
import pickle
from utils import decimal_to_binary, AgentDialogManagement
from prompt import interaction_prompt, agent_characters, agent_roles_datasets
from tqdm import tqdm
import time
from api import openai_api
import os

class helper:
    dataset = None
    prompt = {
        # Please remember it and don't forget it.
        "create_confident": "Imagine you are {} and {}. Please keep this in mind. If you understand please say ok only.",
        "create_temperate": "You are {} and {}. Please keep this in mind. If you understand please say ok only.",
    }

def _print(message):
    print(f"[{time.ctime()}] {message}")

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

def simulate(key, args, agent_config, round_config, case_id:int, database:list):
    rerun_cnt = 0
    while True:
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=args.agent,
            default_model=args.model,
            API_KEY=key,
        )
        agent_center.generate_agents(agent_config=agent_config)
        # agent_center.generate_agents(agent_config=_dynamic_agent_roles(agent_config, data_loader, args, case_id))
        agent_center.parse_message(
            idx="all",
            memory=agent_center.send_message(
                idx="all"
            )
        )
        print(agent_center.agents)
        item = database[case_id]
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
            agent_center.save(path=f"{args.save_path}_case_{case_id}")
            break
        else:
            rerun_cnt += 1
            if rerun_cnt > 5:
                assert False, "too long!"
            continue
            # agent_center.save(path=f"{args.save_path}_case_{case_id}_shutdown")

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
                """reflection"""
                rounds_configs[-1].append({
                    "debate": {"idx": [], "fn": None},
                    "reflection": {"idx": [0, 1, 2], "fn": "start"},
                    "wait": {"idx": [], "fn": ""}
                })
            elif _ == '0':
                """debate"""
                rounds_configs[-1].append({
                    "debate": {"idx": [0, 1, 2], "fn": "next"},
                    "reflection": {"idx": [], "fn": None},
                     "wait": {"idx": [], "fn": ""}
                })
            else:
                assert False, "Error!"

    return agents_configs, rounds_configs

def parse_args():
    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument("--dataset", type=str, default="mmlu", help="which datasets，[chess, mmlu, math]")
    parser.add_argument("--repeat", type=int, default=1, help="which repeat time")
    parser.add_argument("--case_id", type=int, default=0, help="which case, from 0")
    parser.add_argument("--strategy", type=str, default="000", help="strategy")
    parser.add_argument("--turn", type=int, default=3, help="round")
    parser.add_argument("--role", type=int, default=0, help="society")
    parser.add_argument("--agent", type=int, default=3, help="the number of agents")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    # ==============================
    parser.add_argument("--api_idx", type=int, default=0, help="api index")
    parser.add_argument("--api_account", type=str, default=None, help="api account")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--experiment_type", type=str, default="main")
    return parser.parse_args()

def check_args(args):
    assert args.dataset.lower() in ["chess", "mmlu", "math"], "dataset must be in [chess, mmlu, math]"
    assert args.repeat >= 1, "repeat must be >= 1"
    assert args.case_id >= 0, "case_id must be >= 0"
    assert args.turn >= 2, "turn must be >= 2"
    strategies = [decimal_to_binary(i, length=args.turn) for i in range(2 ** args.turn)]
    assert args.strategy in strategies, f"strategy must be in {strategies}"
    assert args.role in [0, 1, 2, 3], "role must be in [0, 1, 2, 3]"
    assert args.api_idx >= 0, "api_idx must be >= 0"
    assert args.api_account is not None, "api_account must be not None"
    return args

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

def get_dataset(args):
    data_path = {
        "mmlu": "./eval_data/mmlu.pkl",
        "chess": "./eval_data/chess.pkl",
        "math": "./eval_data/math.pkl"
    }
    """
    {"task_info": [], "answer": []}
    """
    return pickle.load(open(data_path[args.dataset], "rb"))

def binary_to_decimal(binary):
    return int(binary, 2)

def main(args):
    agents_configs, rounds_configs = create_configs(args)
    cur_agent_config = agents_configs[f"{args.role}_harmony"]
    cur_round_config = rounds_configs[binary_to_decimal(args.strategy)]
    key = openai_api[args.api_account][args.api_idx]
    database = get_dataset(args)
    args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{args.repeat}/" \
                     f"{args.role}_harmony_{args.agent}_agents_{args.turn}_turns_{args.strategy}_strategy"
    assert os.path.exists(f"{args.save_path}_case_{args.case_id}_shutdown.pkl")
    simulate(
        key=key, args=args, agent_config=cur_agent_config, round_config=cur_round_config,
        case_id=args.case_id, database=database["task_info"]
    )

if __name__ == '__main__':
    args = parse_args()
    args = check_args(args)
    init(args)
    main(args)

"""
python rerun.py --turn 3 --agent 3 --api_idx 0 --api_account gpt.test.idx.6 --experiment_type main \
	--dataset mmlu --repeat 4 --case_id 29 --strategy 000 --role 3
	
python rerun.py --turn 3 --agent 3 --api_idx 0 --api_account gpt.test.idx.6 --experiment_type main \
	--dataset mmlu --repeat 5 --case_id 28 --strategy 000 --role 1
"""