from glob import glob
import pandas as pd
import json
import time
import random
import openai

# START: COPIED FROM <https://github.com/composable-models/llm_multiagent_debate.git>

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo-0301",
                  messages=answer_context,
                  n=1)
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

# END: COPIED FROM <https://github.com/composable-models/llm_multiagent_debate.git>

if __name__ == "__main__":
    agents = 3
    rounds = 2

    tasks = glob("/data/vision/billf/scratch/yilundu/llm_iterative_debate/mmlu/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    for i in range(100):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)

        response_dict[question] = (agent_contexts, answer)

    json.dump(response_dict, open("mmlu_{}_{}.json".format(agents, rounds), "w"))
