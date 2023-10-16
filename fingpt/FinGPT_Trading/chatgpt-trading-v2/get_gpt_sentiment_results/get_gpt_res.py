
import pandas as pd
from tqdm import tqdm
import openai
import pickle
import time
from token_ import OPEN_AI_TOKEN

openai.api_key = OPEN_AI_TOKEN
df = pd.read_csv("df.csv",index_col= 0)
df.shape

sentences = df.text.unique()
len(sentences)

bar = tqdm(total=df.shape[0])

res_dict = {}

def get_gpt_res(sentences):
    sentences = [f"Decide whether a sentence's sentiment is positive, neutral, or negative.\n\nSentence: \"{i}\"\nSentiment: " for i in sentences]
    global bar
    bar.update(20)
    try:
        response = openai.Completion.create(
                model = "text-curie-001",
                prompt = sentences,
                temperature=0,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0
                )
        response = [response["choices"][i]["text"] for i in range(len(sentences))]
        return response
    except Exception as e:
        time.sleep(10)
        return "error"
def save_dict(dic,evo):
    with open(f"res/evo.pkl","wb") as f:
        pickle.dump(dic,f)


evo = 0
while len(sentences)>0:
    remain = len(sentences)
    print(f"{evo}: {remain}")
    evo += 1
    save_dict(res_dict,evo)
    to_predict = sentences[:20]
    res = get_gpt_res(to_predict)
    if res == "error":
        pass
    else:
        for i in range(len(to_predict)):
            res_dict[to_predict[i]] = res[i]
        if len(sentences)>20:
            sentences = sentences[20:]
        else:
            break