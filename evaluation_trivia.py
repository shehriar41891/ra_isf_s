import json
import re
from difflib import SequenceMatcher

# Function to Normalize Text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(formemrs|phd|dr|prof|sir|mrs|mr|ms|jr|ii|iii|iv|v|m\.d\.|b\.a\.)", "", text)
    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Function to Compute Similarity Score
def similarity_score(predicted: str, target: str) -> float:
    normalized_predicted = normalize_text(predicted)
    normalized_target = normalize_text(target)
    return SequenceMatcher(None, normalized_predicted, normalized_target).ratio()

# Function to Check Match Against Ground Truths
def match_against_ground_truths(predicted: str, ground_truths: list, threshold: float = 0.8) -> int:
    for gt in ground_truths:
        if similarity_score(predicted, gt) >= threshold:
            return 1
    return 0

# Calculate Modified EM Score
def calculate_modified_em_score(predictions, ground_truths, threshold=0.8):
    matches = 0
    total = len(ground_truths)
    
    for pred, gts in zip(predictions, ground_truths):
        matches += match_against_ground_truths(pred, gts, threshold)
    
    em_score = (matches / total) * 100
    return em_score

# Load JSON file
file_path = "./answer_qa_llama/output_trivia.json"  # Replace with your file path
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Extract Predictions and Ground Truths
predictions = [item['answer'] for item in data]
ground_truths = [item['expected_answer'] for item in data]

# Calculate Modified EM Score
threshold = 1 # Set threshold for similarity
em_score = calculate_modified_em_score(predictions, ground_truths, threshold)

# Print the Result
print('Score on Trivia Qa')
print(f"Exact Match Score: {em_score:.2f}%")
