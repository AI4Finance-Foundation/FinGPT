import random
import openai
import time
import json
import argparse

# START: COPIED FROM <https://github.com/RUCAIBox/HaluEval.git>

openai.api_key = 'sk-'

def get_qa_res(knowledge, question, answer1, answer2, instruction):

    message = [
        {"role": "system", "content":"You are an answer judge. You MUST select an answer from the provided two answers. The answer you provided is \"The best answer is Answer 1.\" or \"The best answer is Answer 2.\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Knowledge#: " + knowledge +
                                    "\n#Question#: " + question +
                                    "\n#Answer 1#: " + answer1 +
                                    "\n#Answer 2#: " + answer2 +
                                    "\n#Your Choice#: "} 
    ]

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=256
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


def get_dialogue_res(knowledge, dialog, response1, response2, instruction):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST select an response from the provided two responses. Your choice MUST be \"The best response is Response 1.\" or \"The best response is Response 2.\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Knowledge#: " + knowledge +
                                    "\n#Dialogue History#: " + dialog +
                                    "\n#Response 1#: " + response1 +
                                    "\n#Response 2#: " + response2 +
                                    "\n#Your Choice#: "}
    ]

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=256
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


def get_summarization_res(document, summary1, summary2, instruction):
    message = [
        {"role": "system", "content": "You are a summary judge. You have to select a summary from the provided two summaris. The answer you provided must be \"The best summary is Summary 1.\" or \"The best summary is Summary 2.\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary 1#: " + summary1 +
                                    "\n#Summary 2#: " + summary2 +
                                    "\n#Your Choice#: "}
    ]

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=256
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


def filtering_qa_dataset(file1, file2, instruction, output_path):
    with open(file1, 'r', encoding="utf-8") as f:
        data1 = []
        for line in f:
            data1.append(json.loads(line))

    with open(file2, 'r', encoding="utf-8") as f:
        data2 = []
        for line in f:
            data2.append(json.loads(line))

        assert(len(data1) == len(data2))
        for i in range(len(data1)):
            knowledge = data1[i]["knowledge"]
            question = data1[i]["question"]
            answer1 = data1[i]["hallucinated_answer"]
            answer2 = data2[i]["hallucinated_answer"]
            right_ans = data1[i]["right_answer"]

            if answer1 == answer2:
                if random.random() > 0.5:
                    ans = "The best answer is Answer 2."
                else:
                    ans = "The best answer is Answer 1."
            else:
                ans = get_qa_res(knowledge, question, answer1, answer2, instruction)
                k = 0
                while("The best answer is Answer 1" not in ans and "The best answer is Answer 2" not in ans) or ("The best answer is Answer 1" in ans and "The best answer is Answer 2" in ans):
                    assert(k<5)
                    ans = get_qa_res(knowledge, question, answer1, answer2, instruction)
                    k = k+1

            if ("1" in ans and "2" in ans) or ("1" not in ans and "2" not in ans):
                answer = {"knowledge": knowledge, "question": question, "right_answer": right_ans, "hallucinated_answer": "failed!"}
            elif "1" in ans:
                answer = {"knowledge": knowledge, "question": question, "right_answer": right_ans, "hallucinated_answer": answer1}
            elif "2" in ans:
                answer = {"knowledge": knowledge, "question": question, "right_answer": right_ans, "hallucinated_answer": answer2}
            else:
                answer = None
            assert(answer is not None)

            dump_jsonl(answer, output_path, append=True)
            print("sample {} completed!".format(i))


def filtering_dialogue_dataset(file1, file2, instruction, output_path):
    with open(file1, 'r', encoding="utf-8") as f:
        data1 = []
        for line in f:
            data1.append(json.loads(line))

    with open(file2, 'r', encoding="utf-8") as f:
        data2 = []
        for line in f:
            data2.append(json.loads(line))

        assert (len(data1) == len(data2))

        for i in range(len(data1)):
            knowledge = data1[i]["knowledge"]
            dialog = data1[i]["dialogue_history"]
            right_response = data1[i]["right_response"]
            response1 = data1[i]["hallucinated_response"]
            response2 = data2[i]["hallucinated_response"]

            if response1 == response2:
                if random.random() > 0.5:
                    res = "The best response is Response 2."
                else:
                    res = "The best response is Response 1."

            else:
                res = get_dialogue_res(knowledge, dialog, response1, response2, instruction)
                k = 0
                while (("The best response is Response 1" not in res and "The best response is Response 2" not in res) or (
                               "The best response is Response 1" in res and "The best response is Response 2" in res)):
                    assert (k < 5)
                    res = get_dialogue_res(knowledge, dialog, response1, response2, instruction)
                    k = k + 1

            if ("1" in res and "2" in res) or ("1" not in res and "2" not in res):
                answer = {"knowledge": knowledge, "dialogue_history": dialog, "right_response": right_response,
                          "hallucinated_response": "failed!"}
                print(i + 1, "failed!")
            elif "1" in res:
                answer = {"knowledge": knowledge, "dialogue_history": dialog, "right_response": right_response,
                          "hallucinated_response": response1}
            elif "2" in res:
                answer = {"knowledge": knowledge, "dialogue_history": dialog, "right_response": right_response,
                          "hallucinated_response": response2}
            else:
                answer = None
            assert (answer is not None)

            dump_jsonl(answer, output_path, append=True)
            print("sample {} completed!".format(i))


def filtering_summarization_dataset(file1, file2, instruction, output_path):
    with open(file1, 'r', encoding="utf-8") as f:
        data1 = []
        for line in f:
            data1.append(json.loads(line))

    with open(file2, 'r', encoding="utf-8") as f:
        data2 = []
        for line in f:
            data2.append(json.loads(line))

        assert (len(data1) == len(data2))

        for i in range(len(data1)):
            document = data1[i]["document"]
            right_summary = data1[i]["right_summary"]
            summary1 = data1[i]["hallucinated_summary"]
            summary2 = data2[i]["hallucinated_summary"]

            if summary1 == summary2:
                if random.random() > 0.5:
                    res = "The best summary is Summary 2."
                else:
                    res = "The best summary is Summary 1."
            else:
                res = get_summarization_res(document, summary1, summary2, instruction)
                k = 0
                while ("The best summary is Summary 1" not in res and "The best summary is Summary 2" not in res) or (
                        "The best summary is Summary 1" in res and "The best summary is Summary 2" in res):
                    assert (k < 5)
                    res = get_summarization_res(document, summary1, summary2, instruction)
                    k = k + 1

            if ("1" in res and "2" in res) or ("1" not in res and "2" not in res):
                answer = {"document": document, "right_summary": right_summary, "hallucinated_summary": "failed!"}
            elif "1" in res:
                if res != "The best summary is Summary 1":
                    res = "The best summary is Summary 1"
                answer = {"document": document, "right_summary": right_summary, "hallucinated_summary": summary1}
            elif "2" in res:
                if res != "The best summary is Summary 2":
                    res = "The best summary is Summary 2"
                answer = {"document": document, "right_summary": right_summary, "hallucinated_summary": summary2}
            else:
                answer = None
            assert (answer is not None)

            dump_jsonl(answer, output_path, append=True)
            print("sample {} completed!".format(i))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


# END: COPIED FROM <  https://github.com/RUCAIBox/HaluEval.git  >

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    args = parser.parse_args()

    instruction_file = "{}/{}_filtering_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    output_path = "../data/{}_data.json".format(args.task, args.task)

    sample_1 = "{}/{}_one-turn_data.json".format(args.task, args.task)
    sample_2 = "{}/{}_multi-turn_data.json".format(args.task, args.task)

    if args.task == "qa":
        filtering_qa_dataset(sample_1, sample_2, instruction, output_path)
    elif args.task == "dialogue":
        filtering_dialogue_dataset(sample_1, sample_2, instruction, output_path)
    elif args.task == "summarization":
        filtering_summarization_dataset(sample_1, sample_2, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
