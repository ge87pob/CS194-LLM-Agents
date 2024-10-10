import math
from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import json

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # TODO
    # This function takes in a restaurant name and returns the reviews for that restaurant. 
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call. 
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    reviews = []
    restaurant_name_lower = restaurant_name.lower()
    with open('restaurant-data.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_restaurant_name = line.split('.')[0].strip()
            if line_restaurant_name.lower() == restaurant_name_lower:
                review = line[len(line_restaurant_name)+1:].strip()
                reviews.append(review)
    return {restaurant_name: reviews}

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service. 
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.04}
    # NOTE: be sure to round the score to 3 decimal places.
    N = len(food_scores)
    assert N == len(customer_service_scores), "Length of food_scores and customer_service_scores must be the same."
    total = 0
    for i in range(N):
        val = math.sqrt(food_scores[i]**2 * customer_service_scores[i])
        total += val
    overall_score = total * (1 / (N * math.sqrt(125))) * 10
    overall_score = round(overall_score, 3)
    overall_score = f"{overall_score:.3f}"
    return {restaurant_name: overall_score}

def get_unique_restaurant_names(file_path: str) -> List[str]:
    """Extracts unique restaurant names from the restaurant-data.txt file."""
    restaurant_names = set()  # Using a set to store unique names
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            restaurant_name = line.split('.')[0].strip()  # Extracting the restaurant name before the period
            restaurant_names.add(restaurant_name)
    return list(restaurant_names)

def create_entrypoint_agent(llm_config, restaurant_names: List[str]):
    # Convert restaurant_names list to a string for the LLM
    restaurant_names_str = ', '.join(restaurant_names)
    
    entrypoint_agent_system_message = f"""
    You are an AI assistant that helps users find the overall score of a restaurant based on reviews.
    Your task is to identify the restaurant name from the user's query by matching it with one of the known restaurant names: {restaurant_names_str}.
    Then, you will call the function to fetch reviews and calculate the overall score.
    """
    
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config,
        code_execution_config={"enabled": True, "use_docker": False},  # Enable code execution without Docker
        human_input_mode="NEVER",  # Never ask for human input
    )
    
    # Register the function with LLM for recognition
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant."
    )(fetch_restaurant_data)
    
    return entrypoint_agent

def create_review_analysis_agent(llm_config):
    review_analysis_agent_system_message = """
    You are an assistant that analyzes a list of restaurant reviews to extract food and customer service scores. 
    Each review corresponds to exactly one food score and one customer service score. 
    You must return the same number of scores as there are reviews.

    For each review:
    - Extract the keyword(s) corresponding to the quality of the food.
    - Extract the keyword(s) corresponding to the quality of the customer service.
    - Assign a food score and a customer service score between 1 and 5 based on the keywords.

    Keywords and their corresponding scores:
    - Score 1: awful, horrible, disgusting
    - Score 2: bad, unpleasant, offensive
    - Score 3: average, uninspiring, forgettable
    - Score 4: good, enjoyable, satisfying
    - Score 5: awesome, incredible, amazing

    Ensure the following:
    - Each review has exactly two scores: one food score and one customer service score.
    - The number of food and customer service scores must exactly match the number of reviews.
    - Return a dictionary with two keys: 'food_scores' and 'customer_service_scores', each containing a list of integers.
    """
    review_analysis_agent = ConversableAgent(
        "review_analysis_agent",
        system_message=review_analysis_agent_system_message,
        llm_config=llm_config
    )
    return review_analysis_agent


def print_chat(conversation_result):
    if 'messages' in conversation_result and len(conversation_result['messages']) > 0:
        print("Chat between agents:")
        for message in conversation_result['messages']:
            print(f"{message['role']}: {message['content']}")
        print("-" * 40)  # Separator for readability


def execute_tool_calls(agent, reply):
    if 'tool_calls' in reply and reply['tool_calls']:
        for tool_call in reply['tool_calls']:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            
            # Execute the corresponding function
            if function_name == "fetch_restaurant_data":
                restaurant_name = function_args["restaurant_name"]
                result = fetch_restaurant_data(restaurant_name)
                print(f"Fetched data for {restaurant_name}: {result}")
                # Return the function result to the agent
                return {"function_result": result}
            elif function_name == "calculate_overall_score":
                restaurant_name = function_args["restaurant_name"]
                food_scores = function_args["food_scores"]
                customer_service_scores = function_args["customer_service_scores"]
                result = calculate_overall_score(restaurant_name, food_scores, customer_service_scores)
                print(f"Calculated overall score for {restaurant_name}: {result}")
                return {"function_result": result}
            else:
                print(f"Unknown function call: {function_name}")
                return {}
    return {}

# Do not modify the signature of the "main" function.
def main(user_query: str):
    # Example LLM config for the agents
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    
    restaurant_data_file = "restaurant-data.txt"
    restaurant_names = get_unique_restaurant_names(restaurant_data_file)

    # Create Agents
    entrypoint_agent = create_entrypoint_agent(llm_config, restaurant_names)
    review_analysis_agent = create_review_analysis_agent(llm_config)
    
    # Step 1: Entrypoint Agent processes the user query
    user_message = {"content": user_query, "role": "user"}
    entrypoint_reply = entrypoint_agent.generate_reply(messages=[user_message])
    print_chat({'messages': [{'role': 'assistant', 'content': entrypoint_reply}]})
    
    # Extract restaurant name from the agent's response (handling the dictionary directly)
    if 'tool_calls' in entrypoint_reply and len(entrypoint_reply['tool_calls']) > 0:
        tool_call = entrypoint_reply['tool_calls'][0]
        if tool_call['function']['name'] == 'fetch_restaurant_data':
            arguments = json.loads(tool_call['function']['arguments'])
            restaurant_name = arguments.get('restaurant_name')
            if not restaurant_name:
                print("Failed to extract restaurant name from the agent's response.")
                return
        else:
            print("Function call is not related to restaurant data fetching.")
            return
    else:
        print("No tool calls found in the response.")
        return
    
    # Step 2: Manually call fetch_restaurant_data
    restaurant_reviews = fetch_restaurant_data(restaurant_name)
    print(f"Fetched Reviews: {restaurant_reviews}")
    
    # Step 3: Send reviews to Review Analysis Agent
    reviews_content = restaurant_reviews[restaurant_name]
    analysis_message = {"content": f"Analyze the following reviews: {reviews_content}", "role": "user"}
    analysis_reply = review_analysis_agent.generate_reply(messages=[analysis_message])
    print_chat({'messages': [{'role': 'assistant', 'content': analysis_reply}]})
    
    try:
        # Remove triple backticks and `python` label if present, then evaluate the string
        clean_reply = analysis_reply.replace("```python", "").replace("```json", "").replace("```", "").strip()
        scores = eval(clean_reply)
    
        food_scores = scores.get('food_scores', [])
        customer_service_scores = scores.get('customer_service_scores', [])
    except Exception as e:
        print(f"Failed to parse analysis scores: {e}")
        return
    
    print(f"Extracted Scores - Food: {food_scores}, Customer Service: {customer_service_scores}")
    
    # Step 4: Manually call calculate_overall_score
    overall_score = calculate_overall_score(restaurant_name, food_scores, customer_service_scores)
    
    # Final Output
    print(overall_score[restaurant_name])


    
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])


