import pickle
from regenerate import find_invalid_case

def read(file_name):
    contexts = pickle.load(open(file_name, "rb"))
    for _ in contexts[0]:
        print(_)

read(file_name="./results/main/math/1/0_harmony_3_agents_3_turns_111_strategy_case_19.pkl")
