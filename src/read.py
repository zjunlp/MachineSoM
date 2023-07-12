import pickle
from regenerate import find_invalid_case

class args:
    dataset = "math"

def read(file_name):
    contexts = pickle.load(open(file_name, "rb"))
    for _ in contexts[0]:
        print(_)

def read_data(dataset="math"):
    """加载答案文件"""
    file_name = f"./results/main/{dataset}_data.pkl"
    database = pickle.load(open(file_name, "rb"))
    """
    {
        "task_info": [(), (), ()],
        "answer": [ , , ,],             # chess是个json
        "ratio": ratio,
        # 一个item有多少个元素
        "item_size": len(database[-1]),
        "role": role,
        "sampled_index": sampled_indexes
    }
    """
    for i in range(50):
        print(database["answer"][i], (database["task_info"][i],))
    # for i in database["answer"]:
    #     print(i)
    
    # return database

read(file_name="./results/main/math/1/0_harmony_3_agents_3_turns_111_strategy_case_19.pkl")
# print(find_invalid_case(args))
# read_data()
# read(file_name="./results/main/mmlu/1/0_harmony_3_agents_3_turns_000_strategy_case_26_token.pkl")