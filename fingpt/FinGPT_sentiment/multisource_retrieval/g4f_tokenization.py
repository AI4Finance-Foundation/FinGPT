import csv
import os
import g4f as g4f
from g4f.Provider import (
    Ails, # "https://api.caipacity.com/v1/chat/completions" 404
    You, # website works; but code incomplete
    Bing, # slow to load
    Yqcloud,
    Theb, # too many historical messages in current chat
    Aichat,
    Bard,
    Vercel,
    Forefront,
    Lockchat,
    Liaobots,
    H2o,
    ChatgptLogin,
    DeepAi,
    GetGpt
)

number_of_messages = 10

csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fiqa', 'test.csv')
messages = []
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        if i >= number_of_messages:  # Limit the number of processed rows to 10
            break
        if row:
            words = ["Given", "financial", "information: "]
            words.extend(row[0].split())  # Split the string into words and extend the list
            words.extend([". ", "Extract", "all", "keywords", "from", "it"])
            messages.append(words)
print("Messages: ", messages)

# Perform completion for each message using the specified provider
provider = Vercel
responses = []
for i in range(number_of_messages):
    print(f"Asking {provider} this: {messages[i]}")
    response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
        {"role": "user", "content": messages[i]}
    ], provider=provider)
    responses.append(response)
    print(f"{provider} answers with this: {responses[i]}")


# normal response
# response = g4f.ChatCompletion.create(model=g4f.Model.gpt_4, messages=[
#                                      {"role": "user", "content": "hi"}]) # alterative model setting
#
# print(response)
#
#
# # Set with provider
# response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.DeepAi, messages=[
#                                      {"role": "user", "content": "Hello world"}], stream=True)
#
# for message in response:
#     print(message)