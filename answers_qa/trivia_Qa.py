import json
import random
# from datasets import load_dataset

# print('We are in the qa brach')
# # Load the dataset
# ds = load_dataset("mandarjoshi/trivia_qa", "rc")

# # Extract the validation dataset (you can use 'test' or any split as needed)
# validation_data = ds['validation']

# # Select only the 'question' and 'answer' columns
# questions_and_answers = validation_data.select_columns(['question', 'answer'])

# # Convert the dataset to a list of dictionaries
# qa_list = [{'question': item['question'], 'answer': item['answer']} for item in questions_and_answers]

# # Save to a JSON file
output_file = "dataset/trivia.json"
# with open(output_file, "w") as f:
#     json.dump(qa_list, f, indent=4)

# print(f"Saved questions and answers to {output_file}")

with open(output_file,"r") as f:
    data = json.load(f)
    
random_sample = random.sample(data, min(500, len(data)))
    
for element in random_sample[:5]:
    print(element['question'])
    print(element['answer']['aliases'])