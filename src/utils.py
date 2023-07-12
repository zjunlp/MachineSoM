import time
import openai
import pickle

SIMULATE_OPENAI = False

def simulate_openai():
    return {
     'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
     'object': 'chat.completion',
     'created': 1677649420,
     'model': 'gpt-3.5-turbo',
     'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
     'choices': [
       {
        'message': {
          'role': 'assistant',
          'content': 'This is a test'},
        'finish_reason': 'stop',
        'index': 0
       }
      ]
    }

def decimal_to_binary(decimal, length):
    """length: 长度"""
    if decimal == 0:
        start = "0"
        for i in range(length - 1):
            start += "0"
        return start
    binary = ''
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal = decimal // 2
    return binary.zfill(length)

class AgentDialogManagement:
    def __init__(
        self,
        prompt: dict,
        num_agents: int,
        default_model: str,
        API_KEY:str,
        ORGNIZATION: str=None,
        RETRY_TIME: int=20,
        SYSTEM_PROMPT: str=None,
        # SYSTEM_PROMPT: str=f"You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2021-09 Current date: {time.strftime('%Y-%m-%d')}"
    ):
        self.prompt = prompt
        self.num_agents = num_agents
        self.default_model = default_model
        self.RETRY_TIME = RETRY_TIME
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        print(SYSTEM_PROMPT)

        self.agents = [
            [] for _ in range(num_agents)
        ]
        self.tokens = [ # 统计token数
            [] for _ in range(num_agents)
        ]
        openai.api_key = API_KEY
        if ORGNIZATION is not None:
            openai.organization = ORGNIZATION

    def _print_log(self, message):
        print(f"[{time.ctime()}] {message}")

    def _check_idx(self, idx):
        if isinstance(idx, list):
            assert len(idx) >= 1
        if isinstance(idx, int):
            assert idx >= 0 and idx < self.num_agents
            idx = [idx]
        if isinstance(idx, str):
            assert idx.lower() == "all"
            idx = list(range(self.num_agents))
        return idx

    def generate_agents(self, agent_config: list):
        """初始化生成智能体"""
        assert len(agent_config) == self.num_agents
        for idx in range(self.num_agents):
            role, character = agent_config[idx]["role"], agent_config[idx]["character"]
            if self.SYSTEM_PROMPT is not None:
                self.agents[idx].append(
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    }
                )
            self.agents[idx].append(
                {
                    "role":"user",
                    "content": self.prompt["create_confident"].format(role, character) if "confident" in character else self.prompt["create_temperate"].format(role, character)
                }
            )

    def send_message(self, idx, model:str=None):
        """将内容发送给openai，并传回来。idx为agent的索引。返回的是"""
        idx:list = self._check_idx(idx)
        if model is None:
            model = self.default_model
        cur_cnt = 0
        memory = []
        while cur_cnt < len(idx):
            try:
                index = idx[cur_cnt]
                assert self.agents[index][-1]["role"] == "user"
                # self._print_log(self.agents[index])
                if not SIMULATE_OPENAI:
                    completion = openai.ChatCompletion.create(
                        model=model,
                        messages=self.agents[index],
                        n=1
                    )
                else:
                    completion = simulate_openai()
                memory.append(completion)
                cur_cnt += 1
            except Exception as e:
                self._print_log(e)
                if "maximum context length is 4097 tokens" in str(e):
                    self._print_log("超过最大长度！跳过！")
                    return None
                self._print_log(f"创建失败，等待{self.RETRY_TIME}秒，正在重新尝试...")
                time.sleep(self.RETRY_TIME)
        return memory

    def parse_message(self, idx, memory: list):
        """将回答的内容添加到agent"""
        """memory的值就是send_message的值"""
        idx:list = self._check_idx(idx)
        """检查是否匹配"""
        assert len(idx) == len(memory)
        for cnt, index in enumerate(idx):
            assert self.agents[index][-1]["role"] == "user"
            content = memory[cnt]["choices"][0]["message"]["content"]
            self.agents[index].append(
                {"role": "assistant", "content": content}
            )
            self.tokens[index].append(
                memory[cnt]['usage']
            )
            # print("parse:", self.agents[index])

    def _prepare_debate(self, idx, fn, task_info):
        """准备debate的内容"""
        """在之前已经check过了"""
        if fn is not None:
            self.prompt["debate"][fn](idx, self, task_info)

    def _prepare_reflection(self, idx, fn, task_info):
        """准备reflection的内容"""
        if fn is not None:
            self.prompt["reflection"][fn](idx, self, task_info)

    def _prepare_wait(self, idx, fn, task_info):
        """准备wait"""
        """do nothing"""
        return

    def prepare_for_message(
        self,
        round_config: dict,
        task_info=None          # 可以不用
    ):
        """搜集为下面的轮次进行准备"""
        """直接将prompt添加"""
        self._prepare_debate(round_config["debate"]["idx"], round_config["debate"]["fn"], task_info)
        self._prepare_reflection(round_config["reflection"]["idx"], round_config["reflection"]["fn"], task_info)
        self._prepare_wait(round_config["wait"]["idx"], round_config["wait"]["fn"], task_info)

    def save(self, path):
        """保存"""
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.agents, f)
        with open(f"{path}_token.pkl", "wb") as f:
            pickle.dump(self.tokens, f)

