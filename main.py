import pandas as pd
import requests
import json
import itertools

prompt_history = []


def get_data():
    data = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
    data.columns = data.columns.str.replace('Column1.', '', regex=False)
    return data

def filter_data(data):
    # remove unnecessary columns
    data = data.iloc[:,:-5]
    data = data.drop('ScoutingReportTemplateId', axis=1)
    data = data.drop('ScoutingReportTemplate', axis=1)
    data = data.drop('EventEndDate', axis=1)
    data = data.drop('FilePartition', axis=1)
    data = data.drop('Age', axis=1)
    data = data.drop('ScoutingReportId', axis=1)
    data = data.drop('ChangedAt',axis=1)

    #filter players with less than 5 matches
    minimum_match_amount = 5
    id_counts = data['PlayerId'].value_counts()
    data = data[data['PlayerId'].isin(id_counts[id_counts >= minimum_match_amount].index)]

    return data

def print_word_count(data, min_words, max_words, filename):
    filter_words = data['word_count'] = data['Comment'].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    filtered_data = data[data['word_count'] > min_words]
    filtered_data = filtered_data[filtered_data['word_count'] <= max_words]
    # Group by 'ScoutID' and take up to 3 entries per group
    filtered_data = filtered_data.groupby('ScoutId').head(3)

    # Save to a text file
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, row in filtered_data.iterrows():
            f.write(f"-------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            f.write(f"Line: {idx}, ScoutID: {row['ScoutId']}, PlayerID: {row['PlayerId']}, ExactPosition: {row['ExactPosition']} \nComment: {row['Comment']} \nWord Count: {row['word_count']}\n\n")

def send_prompt(current_prompt: dict):
    global prompt_history
    # Define the API URL
    url = "http://127.0.0.1:1234/v1/chat/completions"
    message_list = list(itertools.chain(*[[current_prompt], prompt_history]))
    print("MESSAGE LIST: ", message_list)

    # Define the request payload
    payload = {
        "model": "llama-3.2-1b-instruct",
        "messages": message_list,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": True
    }

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request
    try:
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as response:
            response.raise_for_status()  # Raise an HTTPError for bad responses
            print("Streamed Content:")
            result = []
            for line in response.iter_lines():
                if line:  # Ignore keep-alive new lines or empty lines
                    try:
                        # Remove "data: " prefix and parse JSON
                        line_data = json.loads(line.decode("utf-8").lstrip("data: "))
                        # Extract the content field if available
                        choices = line_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                result.append(content)  # Collect content
                    except json.JSONDecodeError:
                        print(f"Non-JSON line received: {line.decode('utf-8')}")

            # Join all collected content into a single string and print it
            full_content = "".join(result)
            print(full_content)
    except requests.exceptions.RequestException as e:
        print("Error:", e)

    prompt_history = message_list
    print("PROMPT_HISTORY: ",prompt_history)


def main():
    data = get_data()
    data = filter_data(data)
    #print_word_count(data,45, 90 ,"filtered_words.txt")
    send_prompt({"role": "user", "content": "Introduce yourself."})
    send_prompt({"role": "user", "content": "When were you developed?"})


if __name__ == '__main__':
    main()