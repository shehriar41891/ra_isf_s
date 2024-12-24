import json
import random

# Path to your JSON file
file_path = "./dataset/hotpot_qa/extracted_questions_answers.json"

# Read the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Randomly sample 100 items
sampled_data = random.sample(data, 500)

# for key,value in sampled_data[:5]:
#     print('key:',key,'value:',value)
    
for element in sampled_data[:5]:
    print('question:',element['question'],'answer:',element['answer'])

import os
file_path = r"D:\rs-isf\ra-isf\ra-isf\Lib\site-packages\opentelemetry_exporter_otlp_proto_http-1.28.2.dist-info\entry_points.txt"
print("Exists:", os.path.exists(file_path))
print("Readable:", os.access(file_path, os.R_OK))
