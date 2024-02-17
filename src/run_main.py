import os
import time
from utils import AgentDialogManagement, isexist
from tqdm import tqdm
from api import openai_api
import argparse
from prompt import agent_roles_datasets, agent_characters, interaction_prompt
from dataloader import dataloader
import pickle
from copy import deepcopy

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
    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--role', type=int, default=0)      # [0,1,2,3] refers to the number of easy-going agents in a society.
    parser.add_argument('--dataset', type=str, default="mmlu")  # chess math
    parser.add_argument('--repeat', type=int, default=1)    # Start labeling from 1, used to indicate the current repeat experiment number; this parameter is also used for saving purposes.
    parser.add_argument('--turn', type=int, default=3)      # collaboration round
    parser.add_argument('--api_idx', type=int, default=0)   # start from 0
    parser.add_argument('--api_account', type=str, default='taobao')    # which account
    parser.add_argument('--experiment_type', type=str, default="main")  # experiment type
    parser.add_argument('--strategies', type=str, default="['000', '001', '010', '011', '100', '101', '110', '111']")       # strategy_id
    parser.add_argument('--case_id', type=str, default=None)                # case_id
    parser.add_argument('--search_experiment_type', type=str, default=None) # This mainly targets the initial loading, determining where to search, corresponding to self.__init_state
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
    if args.search_experiment_type is None:
        args.search_experiment_type = args.experiment_type
    args.strategies = eval(args.strategies)
    if args.case_id is None:
        args.case_id = [i for i in range(args.n_case)]
    else:
        args.case_id = eval(args.case_id)
    assert isinstance(args.case_id, list)
    assert isinstance(args.strategies, list)
    print("*"*10, f"  setting: {args.experiment_type}  ", "*"*10)
    print(f"1. dataset: {args.dataset}\trepeat: {args.repeat}")
    print(f"2. society: {args.role} Easy-going agents\tCollaboration Round: {args.turn}")
    print(f"3. number of agents: {args.agent}\tAPI: {args.api_idx}")
    print(f"4. api account: {args.api_account}")
    print(f"5. number of cases: {args.n_case}")
    print(f"6. strategies: {args.strategies}")
    print(f"7. cases: {args.case_id}")
    print(f"8. model: {args.model}")
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
    """create collaboration configs"""
    _print(f"number of strategies: {2**turn}")
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
            print(f"file `{file_save_names}` exist, skip!")
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
        # create agent
        agent_center.generate_agents(agent_config=agent_config)
        # agent_center.generate_agents(agent_config=_dynamic_agent_roles(agent_config, data_loader, args, case_id))
        # send message and parse it
        agent_center.parse_message(
            idx="all",
            memory=agent_center.send_message(
                idx="all"
            )
        )
        item = data_loader[case_id]
        """Used to mark whether the conversation has not exceeded the maximum length"""
        FLAG_NORMAL = True
        """The "+1" is because the question needs to be counted as well"""
        for round_index in tqdm(range(args.turn+1)):
            """prepare the user message"""
            agent_center.prepare_for_message(
                round_config=round_config[round_index],
                task_info=item
            )
            """Concatenate indices that are not 'wait'"""
            idx = []
            idx.extend(round_config[round_index]["debate"]["idx"])
            idx.extend(round_config[round_index]["reflection"]["idx"])
            """发送信息"""
            memory = agent_center.send_message(idx=idx)
            if memory is None:
                """Indicates that the maximum length has been exceeded, thus terminating the current case"""
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

class TreeNode:
    def __init__(self, value=None, path=None):
        self.value:str = value
        self.left: TreeNode = None
        self.right: TreeNode = None
        # The identification of existence is divided into two steps. 
        # The first step is to directly traverse the directory to see if there is a cache. 
        # If there is, it can be directly loaded. The second step is to run the process.
        self.state:bool = False

        self.agent_center: AgentDialogManagement = None     # agent center
        self.path: str = path                               # Record the path from the parent node to the current node
        self.father: TreeNode = None
        self.depth: int = None                              # start from 0
        self.is_root: bool = False
        self.should_run: bool = True                        # If it's too long, then do not run it first.

class StrategyTree:
    def __init_state(self, case_id: int):
        """The main function of this function is to traverse each node and check if there is a corresponding state. 
        If there is, it can be directly loaded, and then the process is complete."""
        # search_experiment_type
        
        path = f'results/{self.args.search_experiment_type}/{self.args.dataset}/{self.args.repeat}/'
        file_name_template = f"{self.args.role}_harmony_{self.args.agent}_agents_{self.args.turn}_turns_"  # {strategy_labels}_strategy
        file_name_template = f"{self.args.role}_harmony_{self.args.agent}_agents_"
        file_name_last = "{}_strategy_case_{}"

        strategy_in_tree:list = self.__get_all_paths()
        print('hello',strategy_in_tree)
        print(self.visualize())
        print(path)
        # print(file_name_template)
        # assert False

        # Based on the current tree structure, traverse each one.
        # Another situation is when the length is not sufficient. In this case, you need to open that pickle file and then capture a segment.
        # The situation that needs to be dealt with here is, for example, if I ran everything except for 101, then when running 101, it should load 100, but there might be a situation where 1 from 110 gets loaded by mistake.
        # But actually, this is not a big issue because even if it is overwritten, it remains unchanged; it might just consume a bit more memory. This can be used as a method for checking.

        for file in os.listdir(path):
            if file_name_template in file and f'case_{case_id}.pkl' in file and 'token' not in file:
                # The social, dataset, number of repetitions, and case ID just match, and it is not a token file; this is the strategy on the file.
                # cur_strategy = file.split(file_name_template)[1].split('_')[0]
                cur_strategy = file.split(file_name_template)[1].split('_')[2]
                # Check for validity
                self.__check_strategy([cur_strategy])   
                # Whether they are equal, or whether there is a prefix
                round = -1
                # a=input(f"2. {cur_strategy}")
                if cur_strategy in strategy_in_tree:  
                    round = len(cur_strategy)
                else:
                    # If not, then it is necessary to see if the prefix exists.
                    for i in range(len(cur_strategy)-1, 0, -1):
                        if cur_strategy[0:i] in strategy_in_tree:
                            round = i
                            break
                if round >= 1:
                    print(f"命中{path}{file}")
                    self.__set_value(
                        # file_name=f'{path}{file_name_template}'+file_name_last.format(cur_strategy, case_id),
                        file_name=f'{path}{file[:-4]}',
                        strategy=cur_strategy[0:round],    
                        round=round
                    )

    def __find_node_by_path(self, path:str) -> TreeNode:
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if path == node.path:
                return node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def __set_value(self, file_name:str, strategy:str, round:int=None) -> None:
        # First, it is necessary to find the node with the path of 'strategy', and then create each one, traversing upward
        assert round >= 1
        target_node: TreeNode = self.__find_node_by_path(path=strategy)
        agent_center: AgentDialogManagement = self.__pickle2management(file_name)
        target_node.agent_center = self.__cut_agent_center(agent_center=agent_center, round=round, copy=True)
        target_node.state = True
        delta = 1
        while target_node.father != None:
            target_node = target_node.father
            target_node.state = True
            _agent_center: AgentDialogManagement = self.__cut_agent_center(agent_center=agent_center, round=round-delta, copy=True)
            if target_node.agent_center is not None:
                # A simple verification can be performed.
                assert _agent_center.agents == target_node.agent_center.agents, f"{_agent_center.agents}\n\n{target_node.agent_center.agents}"
                assert _agent_center.tokens == target_node.agent_center.tokens
            # else:
            target_node.agent_center = _agent_center
            assert round-delta >= 0
            delta += 1
      
    def __cut_agent_center(self, agent_center: AgentDialogManagement, round:int, copy=True) -> AgentDialogManagement:
        # Trim the contents within agent_center.
        # Round refers to how many items are to be retained.
        if copy:
            agent_center = deepcopy(agent_center)
        for agent_id in range(self.args.agent):
            # Ensure that the current last one is the agent's response
            assert agent_center.agents[agent_id][-1]['role'] == 'assistant'
            assert len(agent_center.agents[agent_id]) >= round*2+4, f"{len(agent_center.agents[agent_id])} < {round*2+4}"
            agent_center.agents[agent_id] = agent_center.agents[agent_id][0:round*2+4]
            agent_center.tokens[agent_id] = agent_center.tokens[agent_id][0:round+2]
        return agent_center

    def __pickle2management(self, file_name:str) -> AgentDialogManagement:
        # The `idx` is mainly used for slicing.
        # The main function of this file is to read in a pkl file and then convert it into an AgentDialogManagement.
        """
        agent_center.agents:
            [
                [{'role':'', 'content':''}],   # agent 1
                [],
                ...
                # There are as many lists as there are agents.
            ]
        agent_center.tokens:
            [
                [,,,],  # The number here is half of that mentioned above
                [],
                ...
                # There are as many lists as there are agents.
            ]
        """
        history_agent = pickle.load(open(f'{file_name}.pkl',"rb"))
        history_token = pickle.load(open(f'{file_name}_token.pkl',"rb"))
        agent_center = AgentDialogManagement(
            prompt=helper.prompt,
            num_agents=self.args.agent,
            default_model=self.args.model,
            API_KEY=self.api_key,
            llama_api=self.api_key
        )
        agent_center.agents = history_agent
        agent_center.tokens = history_token
        assert len(history_agent) == self.args.agent
        return agent_center
    
    def __check_strategy(self, strategies: list):
        for strategy in strategies:
            assert isinstance(strategy, str)
            for thinking_pattern in strategy:
                assert thinking_pattern in ['0', '1']

    def __init__(self, strategies:list, args, api_key):
        self.__check_strategy(strategies)
        self.strategies = strategies
        self.construct_tree()
        self.args = args
        self.api_key:str = api_key

    def construct_tree(self):
        self.root = TreeNode("Root")
        self.root.is_root = True
        self.root.depth = 0
        for strategy in self.strategies:
            self.insert(strategy)
        print('construct done!')
        print(self.visualize())

    def insert(self, path):
        node = self.root
        for idx, char in enumerate(path):
            if char == '0':
                if not node.left:
                    node.left = TreeNode('0')  
                node.left.father = node
                node = node.left
            elif char == '1':
                if not node.right:
                    node.right = TreeNode('1') 
                node.right.father = node
                node = node.right
            node.path = path[0:idx+1]
            node.depth = idx

    def get_depth(self, node):
        if not node:
            return 0
        else:
            return max(self.get_depth(node.left), self.get_depth(node.right)) + 1

    def fill_tree(self, node, depth, pos, tree_lines, width):
        if node:
            left_width = width // 2
            right_width = width - left_width - 1
            tree_lines[depth][pos] = str(node.value)

            if node.left:
                tree_lines[depth + 1][pos - left_width // 2] = '/'
                self.fill_tree(node.left, depth + 2, pos - left_width // 2, tree_lines, left_width)

            if node.right:
                tree_lines[depth + 1][pos + right_width // 2] = '\\'
                self.fill_tree(node.right, depth + 2, pos + right_width // 2, tree_lines, right_width)

    def visualize(self):
        root = self.root
        depth = self.get_depth(root)
        width = 2 ** (depth - 1) * 3
        tree_lines = [[' ' for _ in range(width)] for _ in range(2 * depth - 1)]

        self.fill_tree(root, 0, width // 2, tree_lines, width)

        return '\n'.join([''.join(row) for row in tree_lines])

    def level_order_traversal(self):
        """ Function to perform level order traversal (breadth-first) of the tree. """
        root = self.root
        if not root:
            return []

        queue = [root]
        result = []

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.pop(0)
                current_level.append(node.value)
                print(node.path, end=" ")
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            print()
            result.append(current_level)

        return result
    
    def __get_all_paths(self) -> list:
        # Obtain the path for each node.
        queue = [self.root]
        paths = []
        while queue:
            node = queue.pop(0)
            if not node.is_root:
                paths.append(node.path)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        print(self.visualize())
        return paths

    def __set_should_run(self, start_node: TreeNode, value: bool):
        # Set the should_run attribute of the current node and its child nodes to False, 
        # because the input exceeds the maximum input length of the model.
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            node.should_run = value
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def run(self, case_id:int, data_loader: dataloader, agent_config: list):
        """
        Run a particular case, a particular society, all strategies for a specific repeat experiment.
        :param: case_id
            [0,50)
        :param: data_loader
            used for load case
        :param: agent_config
            Primarily for the first execution, it is necessary to create roles
        """
        # The code has been run here, and before this, the state of all nodes has already been updated. Just load those that are missing.
        # Traverse by hierarchy; if the current state is False, copy the state from the parent node, then run a strategy, and after that proceed to the next one.
        # The execution order here follows a hierarchical traversal.
        self.visualize()
        self.__init_state(case_id=case_id)
        three_type_of_round_config = {
            'start': {
                "debate": {"idx": list(range(self.args.agent)), "fn": "start"},
                "reflection": {"idx": [], "fn": None},
                "wait": {"idx": [], "fn": ""}
            },
            'debate': {
                "debate": {"idx": list(range(self.args.agent)), "fn": "next"},
                "reflection": {"idx": [], "fn": None},
                "wait": {"idx": [], "fn": ""}
            },
            'reflection': {
                "debate": {"idx": [], "fn": None},
                "reflection": {"idx": list(range(self.args.agent)), "fn": "start"},
                "wait": {"idx": [], "fn": ""}
            }
        }
        
        item = data_loader[case_id]

        queue = [self.root]
        # pbar = tqdm(total=)
        while queue:
            node = queue.pop(0)
            RUN = False
            # Currently, it is the root node and has not been run before, which is equivalent to being brand new. 
            # Therefore, role-playing and the first response will be conducted on this node.
            if node.is_root and node.state == False and node.should_run == True:
                assert node.agent_center is None
                """1-role play"""
                # create AgentDialogManagement
                node.agent_center = AgentDialogManagement(
                    prompt=helper.prompt,
                    num_agents=self.args.agent,
                    default_model=self.args.model,
                    llama_api=self.api_key,
                    API_KEY=self.api_key
                )
                # create role
                node.agent_center.generate_agents(agent_config=agent_config)
                # execute role play
                node.agent_center.parse_message(
                    idx="all",
                    memory=node.agent_center.send_message(
                        idx="all"
                    )
                )
                """2-Answer the first question for the first time"""
                cur_round_config = three_type_of_round_config["start"]
                RUN = True
            elif node.is_root == False and node.state == False and node.should_run == True:
                # Currently, it is not a root node, and there is no state for the current node, and the current node is not excessively long.
                # Step 1. initialize agent_center
                assert (node.father.state == True and node.father.agent_center), "Please ensure that the parent node has finished running"
                node.agent_center = deepcopy(node.father.agent_center)
                
                # Step 2. set round_config
                if node.value == '0':
                    # debate
                    cur_round_config = three_type_of_round_config['debate']
                elif node.value == '1':
                    # reflection
                    cur_round_config = three_type_of_round_config['reflection']
                else:
                    assert False
                RUN = True
            if RUN:
                # start !
                """user input"""
                node.agent_center.prepare_for_message(
                    round_config=cur_round_config,
                    task_info=item
                )
                idx = []
                idx.extend(cur_round_config["debate"]["idx"])
                idx.extend(cur_round_config["reflection"]["idx"])
                """send message"""
                memory = node.agent_center.send_message(idx=idx)
                if memory is None:
                    """exceed the max input length"""
                    # Since the current node is too long, it will not be run. Moreover, the should_run attribute of its child nodes should also be set to False.
                    node.should_run = False  
                    # Set the child nodes as non-runnable.
                    self.__set_should_run(node, value=False)    
                else:
                    node.agent_center.parse_message(
                        idx=idx,
                        memory=memory
                    )
                    node.state = True
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if node.state and node.path in self.strategies and node.should_run:
                # The current node has finished running, is not excessively long, 
                # and the corresponding strategy for the current node is in the strategies list.
                print(f"saving {self.args.save_path}{node.path}_strategy_case_{case_id}...")
                node.agent_center.save(
                    path=f"{self.args.save_path}{node.path}_strategy_case_{case_id}"
                )

def generate_agents_config(args) -> dict:
    dataset = args.dataset
    n_agent = args.agent
    agent_roles = agent_roles_datasets[dataset]
    role_dicts = {
        "e": {"role": agent_roles["expert"], "character": agent_characters["temperate"]},   # easygoing
        "o": {"role": agent_roles["expert"], "character": agent_characters["confident"]}    # overconfident
    }
    agents_configs = {}
    for i in range(n_agent+1):
        agents_configs[f'{i}_harmony'] = [role_dicts['e'] for j in range(i)]
        agents_configs[f'{i}_harmony'].extend([role_dicts['o'] for j in range(n_agent-i)])
    return agents_configs

def main():
    args = parse_args()
    args_check(args)
    init(args)
    # Step-1 create agent conifg
    agents_configs = generate_agents_config(args)

    # Step-2 set the society and api according to agents_config and args
    agent_config = agents_configs[f'{args.role}_harmony']
    key = openai_api[args.api_account][args.api_idx]
    data_loader = dataloader(name=args.dataset, n_case=args.n_case)

    # Step-3 run
    for case_id in args.case_id:
        simulator = StrategyTree(
            strategies=args.strategies,
            args=args,
            api_key=key
        )
        args.save_path = f"./results/{args.experiment_type}/{args.dataset}/{args.repeat}/" \
                         f"{args.role}_harmony_{args.agent}_agents_{args.turn}_turns_"  # {strategy_labels}_strategy
        simulator.run(case_id=case_id, data_loader=data_loader, agent_config=agent_config)

if __name__ == '__main__':
    main()

