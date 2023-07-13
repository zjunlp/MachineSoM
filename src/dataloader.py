from glob import glob
import numpy as np
import random
import pandas as pd
import json
import pickle
import tiktoken

class dataloader:
    FILE_PATH = {
        "math": "./data/math/filter.pk",
        "chess": "./data/chess/chess.json",
        "mmlu": "./data/mmlu/data/test/high_school_*.csv"
    }
    def __init__(self, name:str, n_case:int=50):
        assert name.lower() in ["math","chess","mmlu"], f"dataset {name} is not a valid name."
        self.dataset = name.lower()
        self.n_case = n_case
        self.name = name
        self.mapping = {
            "math": self._load_math,
            "mmlu": self._load_mmlu,
            "chess": self._load_chess
        }
        self.database:dict = self.mapping[name]()
        self.mode = "question"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def set_mode(self, mode:str):
        assert mode in ["all", "question", "answer"], f"mode {mode} is not valid."
        self.mode = mode

    def parse_group(self, case_id: int):
        group = 0
        if case_id >= 0 and case_id < self.database["ratio"][1]:
            return group
        group = 1
        while group + 1 < len(self.database["ratio"]) and not (
                case_id >= sum(self.database["ratio"][:group]) and case_id < sum(self.database["ratio"][:group + 1])):
            group += 1
        if case_id >= sum(self.database["ratio"][:group]) and case_id < sum(self.database["ratio"][:group + 1]):
            return group
        else:
            assert False

    def regenerate(self, invalid_case_id:list, num: int=2):
        return:
            {
                "t1":{"task_info", "answer", "ratio", "item_size"}，
                "t2":{"task_info", "answer", "ratio", "item_size"}，
                ......
            }
        """

        def _sort_dict_by_value(dictionary):
            sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
            sorted_keys = [item[0] for item in sorted_dict]
            return sorted_keys

        def _sort_by_tokens(db, idx, return_value=True):
            value = {
                "task_info": [],
                "answer": []
            }
            pair = {}
            if self.dataset == "mmlu":
                for ix in idx:
                    question = db.iloc[ix, 0]
                    a = db.iloc[ix, 1]
                    b = db.iloc[ix, 2]
                    c = db.iloc[ix, 3]
                    d = db.iloc[ix, 4]
                    pair[ix] = 0
                    for _ in (question, a, b, c, d):
                        pair[ix] += len(self.tokenizer.encode(_))
                sorted_index = _sort_dict_by_value(pair)
                if return_value:
                    """
                    {"task_info":[], "answer":[]}
                    """
                    for ix in sorted_index:
                        question = db.iloc[ix, 0]
                        a = db.iloc[ix, 1]
                        b = db.iloc[ix, 2]
                        c = db.iloc[ix, 3]
                        d = db.iloc[ix, 4]
                        answer = db.iloc[ix, 5]
                        value["task_info"].append((question, a, b, c, d,))
                        value["answer"].append(answer)
                    return value
                return sorted_index
            elif self.dataset == "math":
                for ix in idx:
                    pair[ix] = len(self.tokenizer.encode(db[ix]["problem"]))
                sorted_index = _sort_dict_by_value(pair)
                if return_value:
                    for ix in sorted_index:
                        value["task_info"].append((db[ix]["problem"],))
                        value["answer"].append((db[ix]["answer"],))

                    return value
            else:
                assert False

        def _math_reshape(d, types):
            new = {}
            for level in range(3,6):
                new[f"Level {level}"] = []
                for t in types:
                    new[f"Level {level}"].extend(d[f"Level {level}"][t])
            return new

        def generate_candidate():
            self._set_seed(seed=0)
            candidate_id = {}
            """
            {
                "t1": {"task_info":[], "answer": []},
                "t2": {"task_info":[], "answer": []},
                ...
            }
            """
            if self.dataset == "mmlu":
                files_name = glob(dataloader.FILE_PATH["mmlu"])
                for idx in range(len(self.database["ratio"])):
                    db = pd.read_csv(files_name[idx])
                    new_index = list(set(range(len(db)))-set(self.database["sampled_index"][idx]))
                    value = _sort_by_tokens(db, new_index, return_value=True)
                    candidate_id[f"t{idx}"] = value
                return candidate_id
            elif self.dataset == "math":
                types = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
                with open(dataloader.FILE_PATH["math"], "rb") as f:
                    db = pickle.load(f)
                db = _math_reshape(db, types)
                for idx, level in enumerate(range(3, 6)):
                    cur_db = db[f"Level {level}"]
                    new_index = list(set(range(len(cur_db)))-set(self.database["sampled_index"][idx]))
                    value = _sort_by_tokens(cur_db, new_index, return_value=True)
                    candidate_id[f"t{idx}"] = value
                return candidate_id
            else:
                assert False

        """
        return：
        [
            [{"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, ...],
            [{"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, {"task_info": (,), "answer": ""}, ...],
        ]
        """
        return_case = [[] for i in range(len(self.database["ratio"]))]
        if self.database["ratio"] is None:      # chess
            assert self.dataset == "chess"
        else:                                   # math, mmlu
            assert self.dataset in ["math", "mmlu"]
            """
            candidate：
            {
                "t0": {"task_info":[], "answer": []},
                "t1": {"task_info":[], "answer": []},
                ...
            }
            """
            candidate = generate_candidate()
            pre_group = self.parse_group(invalid_case_id[0])
            cur_cache = []  
            for case_id in invalid_case_id:
                group = self.parse_group(case_id)
                if pre_group == group:
                    for _ in range(num):
                        cur_cache.append(
                            {
                                "task_info": candidate[f"t{group}"]["task_info"][len(cur_cache)],
                                "answer": candidate[f"t{group}"]["answer"][len(cur_cache)]
                            }
                        )
                else:
                    # return_case.append(cur_cache.copy())
                    return_case[pre_group] = cur_cache.copy()
                    cur_cache = []
                    for _ in range(num):
                        cur_cache.append(
                            {
                                "task_info": candidate[f"t{group}"]["task_info"][len(cur_cache)],
                                "answer": candidate[f"t{group}"]["answer"][len(cur_cache)]
                            }
                        )
                pre_group = group
            # return_case.append(cur_cache)
            return_case[pre_group] = cur_cache.copy()
            return return_case

    def _load_chess(self):
        self._set_seed(seed=0)
        db = json.load(open(dataloader.FILE_PATH["chess"]))["examples"]
        """
        db = {
            ...,
            "examples":[
                {"input": "", "target": ["", "", ...]},
                {"input": "", "target": ["", "", ...]}
                ...
            ]
        }
        """
        database = []
        answer = []
        sampled_idx = random.sample(list(range(len(db))), self.n_case)
        for idx in sampled_idx:
            database.append((db[idx]["input"],))  
            answer.append(db[idx]["target"])
        return {
            "task_info": database,
            "answer": answer,
            "ratio": None,
            "item_size": 1
        }

    def _load_mmlu(self):
        def parse_question_answer(df, ix):
            question = df.iloc[ix, 0]
            a = df.iloc[ix, 1]
            b = df.iloc[ix, 2]
            c = df.iloc[ix, 3]
            d = df.iloc[ix, 4]
            answer = df.iloc[ix, 5]
            return (question, a, b, c, d, answer)

        def parse_role(filename:str):
            if "high_school_statistics" in filename:
                return "an expert in statistics"
            elif "high_school_mathematics_test" in filename:
                return "an expert in mathematics"
            elif "high_school_computer_science_test" in filename:
                return "an expert in computer science"
            elif "high_school_biology_test" in filename:
                return "an expert in biology"
            elif "high_school_chemistry_test" in filename:
                return "an expert in chemistry"
            elif "high_school_physics_test" in filename:
                return "an expert in physics"
            else:
                assert False

        self._set_seed(seed=0)
        files_name = glob(dataloader.FILE_PATH["mmlu"])
        ratio = [8, 8, 8, 8, 9, 9]
        assert len(files_name) == len(ratio)
        assert sum(ratio) == self.n_case
        database = []
        answer = []
        role = []
        sampled_indexes = []
        for idx in range(len(ratio)):
            db = pd.read_csv(files_name[idx])
            sampled_idx = random.sample(list(range(len(db))), ratio[idx])
            sampled_indexes.append(sampled_idx)
            for i in sampled_idx:
                pair = parse_question_answer(db, i)
                database.append(pair[:-1])
                answer.append(pair[-1])
                role.append(parse_role(files_name[idx]))
        return {
            "task_info": database,
            "answer": answer,
            "ratio": ratio,
            "item_size": len(database[-1]),
            "role": role,
            "sampled_index": sampled_indexes
        }

    def _load_math(self):
        types = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        def reshape(d):
            new = {}
            for level in range(3,6):
                new[f"Level {level}"] = []
                for t in types:
                    new[f"Level {level}"].extend(d[f"Level {level}"][t])
            return new
        self._set_seed(seed=0)
        with open(dataloader.FILE_PATH["math"], "rb") as f:
            db = pickle.load(f)
        db = reshape(db)
        # print(db)
        """
        {
            "Level 1":{
                "algebra":[
                    {"file name":"1.json","problem":"", "level": "", "type": "", "solution": "",},
                    {"file name":"1.json","problem":"", "level": "", "type": "", "solution": "",},
                    {"file name":"1.json","problem":"", "level": "", "type": "", "solution": "",},
                    ......
                ],
                "prealgebra":[],
                ...
            },
            "Level 2":{
            
            }
        }
        """
        ratio = [22, 22, 6]
        database = []
        answer = []
        sampled_indexes = []
        for idx, level in enumerate(range(3,6)):
            sampled_idx = random.sample(list(range(len(db[f"Level {level}"]))), ratio[idx])
            sampled_indexes.append(sampled_idx)
            for i in sampled_idx:
                database.append((db[f"Level {level}"][i]["problem"],))
                answer.append(db[f"Level {level}"][i]["answer"])
        return {
            "task_info": database,
            "answer": answer,
            "ratio": ratio,
            "item_size": 1,
            "sampled_index": sampled_indexes
        }

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def __len__(self):
        assert self.n_case == len(self.database["task_info"])
        return self.n_case

    def __getitem__(self, idx):
        if self.mode == "question":
            return self.database["task_info"][idx]
        elif self.mode == "answer":
            return self.database["answer"][idx]
        elif self.mode == "all":
            return {
                "task_info": self.database["task_info"][idx],
                "answer": self.database["answer"][idx],
                "ratio": self.database["ratio"],
                "item_size": self.database["item_size"]
            }

