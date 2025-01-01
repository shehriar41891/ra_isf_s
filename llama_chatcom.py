import os
import replicate
from dotenv import load_dotenv

def generate_short_response(question):
    """
    Generate a concise response to a question using the Replicate LLM model.

    Parameters:
        question (str): The question to ask the LLM model.

    Returns:
        str: The cleaned, concise answer provided by the LLM model.
    """
    # Load environment variables
    load_dotenv()
    replicate_token = os.getenv('REPLICATE_TOKEN')

    if not replicate_token:
        raise ValueError("REPLICATE_TOKEN is not set in environment variables.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_token

    # Pre-prompt to instruct the model
    pre_prompt = (
        "You are an assistant restricted to providing answers in only one word or a short phrase. "
        "Do not include any additional context, explanations, or extra words. "
        "Your response must directly answer the question. "
        "Do not even include the supporting word or clauses. "
        "If question asked is; What is capital of Pakistan? Your answer should just be 'Islamabad' with no "
        "supporting details or words."
    )

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{pre_prompt}\n\nQuestion: {question}\nAnswer: ",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_length": 10,
            "repetition_penalty": 1.2,
        }
    )

    # Combine the output into a single response
    final_response = "".join(output).strip()

    # Clean the response
    cleaned_response = final_response.replace("Sure, I'd be happy to help!", "").replace("Here's my answer:","").strip()

    return cleaned_response


def generate_short_response_with_context(question, context):
    """
    Generate a concise response to a question using the Replicate LLM model, considering additional context.

    Parameters:
        question (str): The question to ask the LLM model.
        context (str): Additional context to provide to the model for answering the question.

    Returns:
        str: The cleaned, concise answer provided by the LLM model.
    """
    # Load environment variables
    load_dotenv()
    replicate_token = os.getenv('REPLICATE_TOKEN')

    if not replicate_token:
        raise ValueError("REPLICATE_TOKEN is not set in environment variables.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_token

    # Pre-prompt to instruct the model
    pre_prompt = (
        "You are an assistant restricted to providing answers in only one word or a short phrase. "
        "Do not include any additional context, explanations, or extra words. "
        "Your response must directly answer the question. "
        "Do not even include the supporting word or clauses. "
        "If question asked is; What is capital of Pakistan? Your answer should just be 'Islamabad' with no "
        "supporting details or words."
    )

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{pre_prompt}\n\nContext: {context}\n\nQuestion: {question}\nAnswer: ",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_length": 10,
            "repetition_penalty": 1.2,
        }
    )

    # Combine the output into a single response
    final_response = "".join(output).strip()

    # Clean the response
    cleaned_response = final_response.replace("Sure, I'd be happy to help!", "").replace("Here's my answer:","").strip()

    return cleaned_response


def evaluate_query_with_replicate(query):
    """
    Evaluate the given query using the Replicate LLM model to determine if it is within the knowledge domain.

    Parameters:
        query (str): The query to evaluate.

    Returns:
        str: 'know' if the query is within the knowledge domain, otherwise 'not know'.
    """
    # Load environment variables
    load_dotenv()
    replicate_token = os.getenv('REPLICATE_TOKEN')

    if not replicate_token:
        raise ValueError("REPLICATE_TOKEN is not set in environment variables.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_token

    # Pre-prompt to instruct the model
    pre_prompt = (
        "You are an evaluator. For the given query, determine if you have knowledge about it. "
        "If you have any idea or reasonable understanding about the topic, answer 'know'. "
        "If the topic is outside your knowledge domain, answer 'not know'. "
        "Answer only with 'know' or 'not know', without any explanations or additional context."
    )

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{pre_prompt}\n\nQuery: {query}\nAnswer: ",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_length": 10,
            "repetition_penalty": 1.2,
        }
    )

    # Combine the output into a single response
    final_response = "".join(output).strip()

    # Clean the response
    cleaned_response = final_response.replace("Sure, I'd be happy to help!", "").replace('Query Knowledge Evaluation:','').strip()

    return cleaned_response


# Example usage
if __name__ == "__main__":
    question = "in which year vivo launch its first phone in india?"
    answer = generate_short_response(question)
    print(answer)