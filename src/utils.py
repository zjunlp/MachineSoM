import time
import openai
import pickle
import replicate
import os
from copy import deepcopy
from http import HTTPStatus
import dashscope
import requests
import numpy as np

SIMULATE_OPENAI = False

class AnyscaleMixtral:
    def __init__(
        self,
        api_key:str
    ):
        self.api_base = "https://api.endpoints.anyscale.com/v1"
        self.token = api_key
        self.url = f"{self.api_base}/chat/completions"
        self.s = requests.Session()
        print(api_key)

    def create(self, messages: list, temperature: float=0.7, max_new_tokens:int=1024) -> dict:
        print(messages)
        body = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": messages,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
        }
        with self.s.post(self.url, headers={"Authorization": f"Bearer {self.token}"}, json=body) as resp:
            answer = resp.json()
            return answer

class AnyscaleLlaMA13:
    def __init__(
        self,
        api_key:str
    ):
        self.api_base = "https://api.endpoints.anyscale.com/v1"
        self.token = api_key
        self.url = f"{self.api_base}/chat/completions"
        self.s = requests.Session()
        print(api_key)
        print("meta-llama/Llama-2-13b-chat-hf")

    def create(self, messages: list, temperature: float=0.7, max_new_tokens:int=1024) -> dict:
        print(messages)
        body = {
            "model": "meta-llama/Llama-2-13b-chat-hf",
            "messages": messages,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
        }
        with self.s.post(self.url, headers={"Authorization": f"Bearer {self.token}"}, json=body) as resp:
            answer = resp.json()
            return answer

class AnyscaleLlaMA70:
    def __init__(
        self,
        api_key:str
    ):
        self.api_base = "https://api.endpoints.anyscale.com/v1"
        self.token = api_key
        self.url = f"{self.api_base}/chat/completions"
        self.s = requests.Session()
        print(api_key)
        print('meta-llama/Llama-2-70b-chat-hf')

    def create(self, messages: list, temperature: float=0.7, max_new_tokens:int=1024) -> dict:
        print(messages)
        body = {
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "messages": messages,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
        }
        with self.s.post(self.url, headers={"Authorization": f"Bearer {self.token}"}, json=body) as resp:
            answer = resp.json()
            return answer


class ReplicateMixtral:
    def __init__(
        self,
        api_key="",
        openai_format:bool=True,
    ):
        self.openai_format = openai_format
        print('mixtral api:', api_key)
        os.environ['REPLICATE_API_TOKEN'] = api_key

    def convert_openai_format(self, output:str):
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
                        'content': output
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        }

    def messages2str(self, messages: list):
        # [{'role': 'assistant', 'content':''}, {'role': 'user', 'content':''}, ...]
        template = {
            'user': " [INST] {} [/INST]",
            'assistant': " {}</s>"
        }
        dialog_prompt = "<s>"
        for item in messages:
            role, content = item['role'], item['content']
            if role in ['user', 'assistant']:
                dialog_prompt += template[role].format(content)
        return dialog_prompt

    def create(self, messages: list, temperature: float=0.7, max_new_tokens: int=1024):
        stream_output = replicate.run(
            # "mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e",
            "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 0.95,
                "prompt": self.messages2str(messages),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "prompt_template": "{prompt}"
            },
            stream=False
        )
        output = ""
        for i in stream_output:
            output += i
        if self.openai_format:
            print(output)
            return self.convert_openai_format(output)
        else:
            return output

class DashscopeQwen:
    def __init__(
        self,
        api_key="",
        openai_format:bool=True,
        model_name="qwen-max-1201",
        seed=2023
    ):
        dashscope.api_key = api_key
        self.model_name: str = model_name
        self.openai_format = openai_format
        self.seed = seed
        print(f"model: {self.model_name}")

    def _qwen_format(self):
        output = {
            "status_code": 200,
            "request_id": "05dc83af-7185-9e14-9b0b-4466de159d6a",
            "code": "",
            "message": "",
            "output": {
                "text": None,
                "finish_reason": None,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "首先，准备两个鸡蛋，一个西红柿，适量的盐、糖、料酒和生抽。将鸡蛋打入碗中，搅拌均匀，西红柿切块。锅中加油，油热后加入鸡蛋液，炒至金黄色，盛出备用。锅中加油，油热后加入西红柿块，翻炒均匀，加入适量的盐、糖、料酒和生抽，炒至西红柿软烂，加入炒好的鸡蛋，翻炒均匀即可。"
                        }
                    }
                ]
            },
            "usage": {
                "input_tokens": 12,
                "output_tokens": 98,
                "total_tokens": 110
            }
        }

    def convert_openai_format(self, response):
        return {
            'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
            'object': 'chat.completion',
            'created': 1677649420,
            'model': 'gpt-3.5-turbo',
            'usage': {'prompt_tokens': response.usage['input_tokens'], 'completion_tokens': response.usage['output_tokens'], 'total_tokens': response.usage['total_tokens']},
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': response.output.choices[0]['message']['content']
                    },
                    'finish_reason': response.output.choices[0]["finish_reason"],
                    'index': 0
                }
            ]
        }

    def create(self, messages: list, temperature: float=0.75, max_new_tokens: int=1000):
        response = dashscope.Generation.call(
            self.model_name,
            messages=messages,
            # set the random seed, optional, default to 1234 if not set
            seed=self.seed,
            result_format='message',  # set the result to be "message" format.
            temperature=temperature,
        )
        if self.openai_format:
            return self.convert_openai_format(response)
        else:
            return response

class ReplicateLlaMA:
    model_id = None
    def __init__(
        self,
        api_key:str,
        system_prompt="You are a helpful, respectful and honest assistant.",
        openai_format:bool=True
    ):
        print('mike_use_api:', api_key)
        os.environ['REPLICATE_API_TOKEN'] = api_key
        self.template = {
            'system': """<s>[INST] <<SYS>>
{}
<</SYS>>""",
            'user': """{} [/INST] """,
            'assistant': """{} </s><s>[INST] """,
            'concat': """{}\n\n{}"""
        }
        self.system_prompt = system_prompt
        self.openai_format = openai_format

    def messages2str(self, messages: list):
        # [{'role': 'assistant', 'content':''}, {'role': 'user', 'content':''}, ...]
        system_prompt = ""
        dialog_prompt = ""
        for item in messages:
            role, content = item['role'], item['content']
            # if role == 'system':
            #     system_prompt = self.template[role].format(content)
            if role in ['user', 'assistant']:
                dialog_prompt += self.template[role].format(content)

        # if len(system_prompt) == 0:
        #     system_prompt = self.template['system'].format(self.system_prompt)
        # return self.template['concat'].format(system_prompt, dialog_prompt).strip()
        return dialog_prompt[0:-9]
    
    def convert_openai_format(self, output:str):
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
                        'content': output
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        }

    def create(self, messages: list, temperature: float=0.75, max_new_tokens: int=1000):
        stream_output = replicate.run(
            ReplicateLlaMA.model_id,
            # "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            # 'meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1',
            # 'meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d',
            input={
                "debug": False,
                "top_k": 40,
                "top_p": 0.95,
                "prompt": self.messages2str(messages),
                "temperature": temperature,
                "system_prompt": self.system_prompt,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": -1,
                "prompt_template": "{prompt}"
            },
            stream=False
        )
        output = ""
        for i in stream_output:
            output += i
        if self.openai_format:
            print(output)
            return self.convert_openai_format(output)
        else:
            return output


MODEL_MAPPING = {
    'AnyscaleLlaMA13'.lower(): AnyscaleLlaMA13,
    'AnyscaleLlaMA70'.lower(): AnyscaleLlaMA70,
    'AnyscaleMixtral'.lower(): AnyscaleMixtral,
    'DashscopeQwen'.lower(): DashscopeQwen,
    'ReplicateMixtral'.lower(): ReplicateMixtral,
    'ReplicateLlaMA13'.lower(): ReplicateLlaMA,
    'ReplicateLlaMA70'.lower(): ReplicateLlaMA
}

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
          'content': f'This is a test. {time.ctime()}'},
        'finish_reason': 'stop',
        'index': 0
       }
      ]
    }

def decimal_to_binary(decimal, length):
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

def isexist(file_name):
    return os.path.exists(file_name)

def reduce_memory(dialog:list, idx:int, mode:str='', cut_rate=0.5, model:str='llama', save_behind:bool=True):
    def restore_dialog(content:str):
        flag = '\n\n One agent response: ```'
        assert flag in content
        _split = content.split(flag)
        new_split = [_split[0]]
        ans_idx = []
        for i in range(1, len(_split)):
            new_split.append(flag)
            assert '```' in _split[i]
            start = _split[i].rfind('```')
            ans_idx.append(len(new_split))
            new_split.append(_split[i][0:start])
            new_split.append(_split[i][start:])
        return new_split, ans_idx

    assert mode.lower() in ['remove', 'summary', 'cut']
    assert cut_rate < 1 and cut_rate >0
    copy_dialog = deepcopy(dialog)
    content:str = dialog[idx]['content']
    if mode.lower() == 'remove':
        dialog.pop(idx)
    elif mode.lower() == 'summary':
        pass
        # if model == 'llama':
        #     model = llama(
        #         system_prompt='You are a helpful, respectful and honest assistant. You are very skilled at condensing given text without changing the original meaning of the sentences. Please maintain the original narrative person.', 
        #         openai_format=False
        #     )
            
        #     split_content, agent_idx = restore_dialog(content=content)
        #     for i in agent_idx:
        #         messages = [
        #             {'role': 'user', 'content': f'Here are the sentences that need to be abbreviated. \n```{split_content[i]}\n```'},
        #         ]
        #         print('org:', split_content[i])
        #         split_content[i] = model.create(messages)
        #         if '"' in split_content[i]:
        #             split_content[i] = split_content[i][split_content[i].find('"')-1:]
        #         print('out:', split_content[i])
        #     print(f'''summary: {len(dialog[idx]['content'].split(' '))} -> {len(''.join(split_content).split(' '))}''')
        #     dialog[idx]['content'] = ''.join(split_content)
    elif mode.lower() == 'cut':
        try:
            # 就是将其进行分割吧
            results, template = split_debate(content)
            assert len(results)>=3
            for i in range(1, len(results)-1):
                _split = results[i].split(' ')
                length = int(len(_split)*cut_rate)
                if save_behind:
                    results[i] = ' '.join(_split[length:])
                else:
                    results[i] = ' '.join(_split[0:len(_split)-length])
            dialog[idx]['content'] = template.format(*results)
            # a = input(f"<MIKE> {dialog[idx]['content']} </MIKE>")
        except:
            _split = content.split(' ')
            length = int(len(_split)*cut_rate)
            if save_behind:
                print('origin:', content, end="<split>")
                dialog[idx]['content'] = ' '.join(_split[length:])
                print('after:', dialog[idx]['content'])
            else:
                print('origin:', content, end="<split>")
                dialog[idx]['content'] = ' '.join(_split[0:len(_split)-length])
                print('after:', dialog[idx]['content'])
        
    else:
        assert False
    
    return dialog, copy_dialog

def split_debate(content: str) -> tuple:
    flag = "\n\n One agent response: ```"
    flag_end = "```"
    assert flag in content

    parts = content.split(flag)
    for i in range(len(parts)):
        if flag_end in parts[i]:
            parts[i] = parts[i].split(flag_end)
    results = []
    template = "{}"+flag
    prefix = ""
    for i in range(len(parts)):
        if isinstance(parts[i], list):
            assert len(parts[i]) == 2
            results.append(prefix+parts[i][0])
            prefix = parts[i][1]
            if i!=len(parts)-1:
                template += ("{}"+flag_end+flag)
        elif isinstance(parts[i], str):
            results.append(parts[i])
        else:
            assert False
    results.append(prefix)
    template += ("{}" + flag_end+"{}")
    
    assert template.format(*results)==content
    return results, template

class AgentDialogManagement:
    def __init__(
        self,
        prompt: dict,
        num_agents: int,
        default_model: str,
        API_KEY:str,
        ORGNIZATION: str=None,
        RETRY_TIME: int=180,
        SYSTEM_PROMPT: str=None,
        # SYSTEM_PROMPT: str=f"You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2021-09 Current date: {time.strftime('%Y-%m-%d')}"
    ):
        self.prompt = prompt
        self.num_agents = num_agents
        self.default_model = default_model
        self.RETRY_TIME = RETRY_TIME - np.random.randint(int(RETRY_TIME*0.8))
        print(f"Wait for {self.RETRY_TIME} seconds if timeout occurs")
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        print(SYSTEM_PROMPT)

        self.agents = [
            [] for _ in range(num_agents)
        ]
        self.tokens = [ 
            [] for _ in range(num_agents)
        ]
        if default_model.lower() in MODEL_MAPPING:
            self.model = MODEL_MAPPING[default_model](api_key=API_KEY)
            print(f'Backbone: {default_model}')
            # "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            # 'meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1',
            if default_model.lower().startswith('replicate') and 'llama70' in default_model.lower():
                print(f'Backbone: ReplicateLlaMA70')
                ReplicateLlaMA.model_id = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
            elif default_model.lower().startswith('replicate') and 'llama13' in default_model.lower():
                print(f'Backbone: ReplicateLlaMA13')
                ReplicateLlaMA.model_id = 'meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1'
        else:
            print(f"Backbone: OpenAI- {default_model}")
            self.model = None
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

    def send_message(self, idx, model:str=None, temperature:float=0.75, max_new_tokens:int=1000):
        idx:list = self._check_idx(idx)
        if model is None:
            model = self.default_model
        cur_cnt = 0
        memory = []
        summary_log = None
        RETRY_CNT = 0
        while cur_cnt < len(idx):
            if RETRY_CNT >= 20:
                self._print_log("exceed the max count of retrying")
                return None
            try:
                index = idx[cur_cnt]       
                if summary_log is None:
                    summary_log = {'agent': cur_cnt, 'summary?':False, 'idx': 3, 'copy': None}
                assert self.agents[index][-1]["role"] == "user"
                # self._print_log(self.agents[index])
                if not SIMULATE_OPENAI:
                    if self.model is not None:
                        print('send!')
                        completion = self.model.create(
                            messages=self.agents[index], 
                            temperature=temperature, 
                            max_new_tokens=max_new_tokens
                        )
                        if 'error' in completion:
                            self._print_log(completion['error'])
                            content = 'Anyscale Mixtral Error!'
                            if 'Input too long'.lower() in completion['error']['message'].lower():
                                content = "maximum context length is 4097 tokens"
                            raise ConnectionResetError(content)
                    else:
                        completion = openai.ChatCompletion.create(
                            model=model,
                            messages=self.agents[index],
                            n=1
                        )
                else:
                    completion = simulate_openai()
                memory.append(completion)
                cur_cnt += 1
                if summary_log['summary?']:
                    self.agents[index] = summary_log['copy']
                summary_log = None
            except Exception as e:
                self._print_log(e)
                if "maximum context length is 4097 tokens" in str(e) or "Your input is too long.".lower() in str(e).lower():
                    
                    # if self.default_model.lower() in ['llama', 'anyscalellama13', 'anyscalellama70']:
                    #     self._print_log("summary ...")
                    #     # if summary_log['idx'] > 3:
                    #     #     self._print_log("exceed, skip!")
                    #     #     return None
                    #     # print(summary_log['idx'])
                    #     # for _ in range(summary_log['idx'], 3, -1):
                    #     # a=input(f"{summary_log['idx']},{len(self.agents[index])}")
                    #     for _ in range(summary_log['idx'], len(self.agents[index])): 
                    #         # print('_:', _)
                    #         if 'One agent response: ```' in self.agents[index][_]['content'] or self.agents[index][_]['role'] == 'assistant':
                    #         # if self.agents[index][_]['role'] == 'assistant':
                    #             # b = input('in:')
                    #             new_dialog, orig_dialog = reduce_memory(self.agents[index], idx=_, mode='cut', cut_rate=0.5, save_behind=True)
                    #             if summary_log['copy'] is None:
                    #                 summary_log['copy'] = orig_dialog
                    #                 self.agents[index] = new_dialog
                    #                 # summary_log['idx'] = _ - 1
                    #             summary_log['idx'] = _ + 1
                    #             summary_log['summary?'] = True
                    #             break
                    #         else:
                    #             # b = input('on:')
                    #             print(self.agents[index][_])
                    #     RETRY_CNT += 1
                    #     continue
                    return None
                self._print_log(f"waiting for {self.RETRY_TIME} second...")
                time.sleep(self.RETRY_TIME)
        return memory

    def parse_message(self, idx, memory: list):
        idx:list = self._check_idx(idx)
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

    def _prepare_debate(self, idx, fn, task_info):
        if fn is not None:
            self.prompt["debate"][fn](idx, self, task_info)

    def _prepare_reflection(self, idx, fn, task_info):
        if fn is not None:
            self.prompt["reflection"][fn](idx, self, task_info)

    def _prepare_wait(self, idx, fn, task_info):
        return

    def prepare_for_message(
        self,
        round_config: dict,
        task_info=None         
    ):
        self._prepare_debate(round_config["debate"]["idx"], round_config["debate"]["fn"], task_info)
        self._prepare_reflection(round_config["reflection"]["idx"], round_config["reflection"]["fn"], task_info)
        self._prepare_wait(round_config["wait"]["idx"], round_config["wait"]["fn"], task_info)

    def save(self, path):
        print(f"[utils.py] [AgentDialogManagement] saving {path}.pkl ...")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.agents, f)
        print(f"[utils.py] [AgentDialogManagement] saving {path}_token.pkl ...")
        with open(f"{path}_token.pkl", "wb") as f:
            pickle.dump(self.tokens, f)

if __name__ == '__main__':
    exit()