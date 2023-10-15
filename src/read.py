import pickle
from regenerate import find_invalid_case

class args:
    dataset = "math"

def read(file_name):
    contexts = pickle.load(open(file_name, "rb"))
    for _ in contexts[0]:
        print(_)

def read_dataset(file_name):
    contexts = pickle.load(open(file_name, "rb"))
    print(contexts.keys())
    print(contexts["role"])

def read_data(dataset="math"):
    file_name = f"./results/main/{dataset}_data.pkl"
    database = pickle.load(open(file_name, "rb"))
    """
    {
        "task_info": [(), (), ()],
        "answer": [ , , ,],             # chess json
        "ratio": ratio,
        # # element in item
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

read(file_name="./results/main/math/2/0_harmony_3_agents_3_turns_000_strategy_case_19.pkl")
# read_dataset(file_name="./eval_data/mmlu.pkl")
# print(find_invalid_case(args))
# read_data()
# read(file_name="./results/main/mmlu/1/0_harmony_3_agents_3_turns_000_strategy_case_26_token.pkl")