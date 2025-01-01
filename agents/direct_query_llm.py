import os
import replicate
from dotenv import load_dotenv

load_dotenv()
replicate_token = os.getenv('REPLICATE_TOKEN')

os.environ['REPLICATE_API_TOKEN'] = replicate_token

# Pre-prompt to enforce concise responses
pre_prompt = (
    "You are an assistant restricted to providing answers in only one word or a short phrase. "
    "Strictly do not include any context, explanations, or extra words. "
    "Only provide the direct answer without any other content."
)
prompt_input = "Who is the captain of Pakistan cricket team?"

# Generate LLM response
output = replicate.run(
    'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',  # LLM model
    input={
        "prompt": f"{pre_prompt}\n\nQuestion: {prompt_input}\nAnswer: ",
        "temperature": 0.0,  # Reduce randomness
        "top_p": 1.0,       # Prioritize highest probability tokens
        "max_length": 20,   # Limit response length
        "repetition_penalty": 1.2,  # Discourage repetitive responses
    }
)

# Extract concise output from the response
def extract_direct_answer(full_response):
    # Simple logic to filter the direct answer
    for line in full_response:
        line = line.strip()
        if line and "Assistant:" not in line and "Sure" not in line:
            return line.split(".")[0]  # Return the first sentence or phrase
    return "No valid response."

# Combine the output and process it
final_response = extract_direct_answer("".join(output))

# Print the final answer
print(final_response)
