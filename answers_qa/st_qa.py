import json

output_file = 'dataset/StrategyQA/StrategyQA.json'
with open(output_file,'r') as f:
    data = json.load(f)
    

# Extract the 'examples' part
examples_data = data.get("examples", [])

# Print the examples
for example in examples_data[:5]:
    print(example['target'].split(".")[0])  # Pretty-print each example
    print(example['input'])