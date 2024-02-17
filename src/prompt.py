
agent_roles_datasets = {
    "mmlu":{
        "expert": "an expert in biology, chemistry, computer science, mathematics, physics"
    },
    "chess":{
        "expert": "an expert skilled in playing chess"
    },
    "math":{
        "expert": "an expert skilled in solving mathematical problems"
    },
}

agent_characters = {
    "temperate": "are objective and unbiased, and you can be persuaded if other agent's answers make sense",
    "confident": "are confident in your answer and often persuades other agents to believe in you"
}


interaction_prompt = {
    "mmlu":{
        "question": "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ],
        "reflection": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response.",
    },
    "math":{
        # "question": "Here is a math problem written in LaTeX:{}\nPlease carefully consider it and confirm your answer by starting with \"The answer is $Your Answer$\" at the end. Let's think step by step.",
        "question": "Here is a math problem written in LaTeX:{}\nPlease carefully consider it and explain your reasoning. Put your answer in the form \\boxed{{answer}}, at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents:",
            "\n\nUsing the reasoning from other agents as additional information and referring to your historical answers, can you give an updated answer? Put your answer in the form \\boxed{{answer}}, at the end of your response."
            # "\n\nUsing the solutions from other agents as additional information, can you provide your answer to the math problem? Put your answer in the form \\boxed{{answer}}, at the end of your response."
            # "\n\nUsing the reasoning from other agents as addtitional advice and referring to your historical answers, can you give an updated answer? Check your historical answers and the answers from other agents. Put your answer in the form \\boxed{{answer}}, at the end of your response."
        ],
        "reflection": "Can you double check that your answer is correct? Please reiterate your answer, with your answer in the form \\boxed{{answer}}, at the end of your response.",
    },
    "chess":{
        "question": "Given the chess game \"{}\", give one valid destination square for the chess piece at \"{}\". Give a one line explanation of why your destination square is a valid move. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]. ",
        "debate": [
            "Here are destination square suggestions from other agents:",
            "\n\nCan you double check that your destination square is a valid move? Check the valid move justifications from other agents and your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."
        ],
        "reflection": "Can you double check that your destination square is a valid move? Check the valid move justifications from your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8].",
    },
    "csqa":{
        "question": "",
        "debate": [
            "",
            ""
        ],
        "reflection": ""
    }
}
