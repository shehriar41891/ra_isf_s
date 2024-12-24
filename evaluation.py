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

# Function to Check Match with Threshold
def match_with_threshold(predicted: str, target: str, threshold: float = 0.8) -> int:
    score = similarity_score(predicted, target)
    return 1 if score >= threshold else 0

# Calculate Modified EM Score
def calculate_modified_em_score(predictions, ground_truths, threshold=0.8):
    matches = 0
    for pred, gt in zip(predictions, ground_truths):
        if match_with_threshold(pred, gt, threshold) == 1:
            matches += 1
    em_score = (matches / len(ground_truths)) * 100
    return em_score

# Load JSON file
file_path = "./answers_qa/output_st.json"  # Replace with your file path
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Extract Predictions and Ground Truths
predictions = [item['answer'] for item in data]
print(predictions)
ground_truths = [item['expected_answer'] for item in data]
print(ground_truths)
# Calculate Modified EM Score
threshold = 1  # Set threshold for similarity
em_score = calculate_modified_em_score(predictions, ground_truths, threshold)

# Print the Result
print(f"Exact Match Score with Threshold {threshold * 100:.0f}%: {em_score:.2f}%")
