import openai
import time
import json
import argparse
import csv


# START: COPIED FROM <https://github.com/RUCAIBox/HaluEval.git

openai.api_key = 'sk-'


def get_qa_res(knowledge, question, answer, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Knowledge#: " + knowledge +
                                        "\n#Question#: " + question +
                                        "\n#Right Answer#: " + answer +
                                        "\n#Hallucinated Answer#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Question#: " + question +
                           "\n#Right Answer#: " + answer +
                           "\n#Hallucinated Answer#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")
             
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    
    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def get_dialogue_res(knowledge, dialog, response, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Knowledge#: " + knowledge +
                                        "\n#Dialogue History#: " + dialog +
                                        "\n#True Response#: " + response +
                                        "\n#Hallucinated Response#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated response for the following dialogue. You can use any method you have learned that is suitable for the given dialogue history." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Dialogue History#: " + dialog +
                           "\n#True Response#: " + response +
                           "\n#Hallucinated Response#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def get_summarization_res(text, summary, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Document#: " + text +
                                        "\n#Right Summary#: " + summary +
                                        "\n#Hallucinated Summary#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated summary for the following document. You can use any method you have learned that is suitable for the given document. #Hallucinated Summary# must not be longer than #Right Summary#." +
                           "\n\n#Document#: " + text +
                           "\n#Right Summary#: " + summary +
                           "\n#Hallucinated Summary#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def generate_qa_dataset(seed_data, instruction, output_path):
    with open(seed_data, 'r', encoding="utf-8") as f:
        text = json.load(f)

        for i in range(10000):
            question = text[i]['question']
            answer = text[i]['answer']
            supporting_facts = text[i]['supporting_facts']
            context = text[i]['context']
            knowledge = ""
            for fact in supporting_facts:
                for para in context:
                    if para[0] == fact[0]:
                        if fact[1] < len(para[1]):
                            knowledge = knowledge + para[1][fact[1]]
            ans = get_qa_res(knowledge, question, answer, instruction)
            data = {"knowledge": knowledge, "question": question, "right_answer": answer, "hallucinated_answer": ans}
            dump_jsonl(data, output_path, append=True)
            print(" sample {} completed!".format(i))


def generate_dialogue_dataset(seed_data, instruction, output_path):
    SENDER = {"user": "[Human]", "assistant": "[Assistant]"}
    with open(seed_data, 'r', encoding="utf-8") as f:
        i = 0
        data = csv.DictReader(f)
        for r in data:
            if i >= 10000:
                break
            r = eval(r['Messages'])
            dialog = ""
            knowledge = ""
            response = ""
            k = 0
            d = 0
            for message in r:
                if "message" in message:
                    if k > 1 and message['sender'] == "assistant":
                        response = message['message']
                        break
                    if d > 3 and message['sender'] == "assistant":
                        response = message['message']
                        break
                    else:
                        dialog = dialog + (SENDER[message['sender']] + ": " + message['message']) + " "
                        d = d + 1

                if "metadata" in message:
                    if "path" in message['metadata']:
                        knowledge = knowledge + message['metadata']['path'][2]
                    k = k + 1

            if knowledge == "" or dialog == "" or response == "":
                continue
            res = get_dialogue_res(knowledge, dialog, response, instruction)
            data = {"knowledge": knowledge, "dialogue_history": dialog, "right_response": response, "hallucinated_response": res}
            dump_jsonl(data, output_path, append=True)
            i = i + 1
            print("sample {} completed!".format(i))


def generate_summarization_dataset(seed_data, instruction, output_path):
    with open(seed_data, 'r', encoding="utf-8") as f:
        data = f.readlines()
        text = [json.loads(d) for d in data]

        for i in range(10000):
            document = text[i]["document"]
            summary = text[i]["summary"]
            sum = get_summarization_res(document, summary, instruction)
            data = {"document": document, "right_summary": summary, "hallucinated_summary": sum}
            dump_jsonl(data, output_path, append=True)
            print("sample {} completed!".format(i))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

# END: COPIED FROM < https://github.com/RUCAIBox/HaluEval.git >


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--seed_data", default="hotpot_train_v1.1.json", help="the original dataset file")
    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--strategy",default="one-turn", help="one-turn or multi-turn")
    args = parser.parse_args()

    seed_data = args.seed_data

    if args.strategy == "one-turn":
        instruction_file = "{}/{}_{}_instruction.txt".format(args.task, args.task, args.strategy)
        f = open(instruction_file, 'r', encoding="utf-8")
        instruction = f.read()
    elif args.strategy == "multi-turn":
        instruction_file = "{}/{}_{}_instruction.json".format(args.task, args.task, args.strategy)
        with open(instruction_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        instruction = [json.loads(line) for line in lines]
    else:
        raise ValueError("The strategy must be one-turn or multi-turn!")

    output_path = "{}/{}_{}_data.json".format(args.task, args.task, args.strategy)

    if args.task == "qa":
        generate_qa_dataset(seed_data, instruction, output_path)
    elif args.task == "dialogue":
        generate_dialogue_dataset(seed_data, instruction, output_path)
    elif args.task == "summarization":
        generate_summarization_dataset(seed_data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
