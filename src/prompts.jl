# Prompts for fact extraction, memory updates, and procedural memory

using Dates

function _today_str()
    Dates.format(Dates.now(), "yyyy-mm-dd")
end

const MEMORY_ANSWER_PROMPT = """
You are an expert at answering questions based on the provided memories. Your task is to provide accurate and concise answers to the questions by leveraging the information given in the memories.

Guidelines:
- Extract relevant information from the memories based on the question.
- If no relevant information is found, make sure you don't say no information is found. Instead, accept the question and provide a general response.
- Ensure that the answers are clear, concise, and directly address the question.

Here are the details of the task:
"""

function user_memory_extraction_prompt()
    """You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input data.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Here are some few shot examples:

User: Hi.
Assistant: Hello! I enjoy assisting you. How can I help today?
Output: {"facts" : []}

User: There are branches in trees.
Assistant: That's an interesting observation. I love discussing nature.
Output: {"facts" : []}

User: Hi, I am looking for a restaurant in San Francisco.
Assistant: Sure, I can help with that. Any particular cuisine you're interested in?
Output: {"facts" : ["Looking for a restaurant in San Francisco"]}

User: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Assistant: Sounds like a productive meeting. I'm always eager to hear about new projects.
Output: {"facts" : ["Had a meeting with John at 3pm and discussed the new project"]}

User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John! My name is Alex and I admire software engineering. How can I help?
Output: {"facts" : ["Name is John", "Is a Software engineer"]}

User: Me favourite movies are Inception and Interstellar. What are yours?
Assistant: Great choices! Both are fantastic movies. I enjoy them too. Mine are The Dark Knight and The Shawshank Redemption.
Output: {"facts" : ["Favourite movies are Inception and Interstellar"]}

Return the facts and preferences in a JSON format as shown above.

Remember the following:
# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.
- Today's date is $(_today_str()).
- Do not return anything from the custom few shot example prompts provided above.
- If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
- Create the facts based on the user messages only. Do not pick anything from the assistant or system messages.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts" and corresponding value will be a list of strings.
- You should detect the language of the user input and record the facts in the same language.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above.
"""
end

function agent_memory_extraction_prompt()
    """You are an Assistant Information Organizer, specialized in accurately storing facts, preferences, and characteristics about the AI assistant from conversations.
Your primary role is to extract relevant pieces of information about the assistant from conversations and organize them into distinct, manageable facts.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES. DO NOT INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Assistant's Preferences: Keep track of likes, dislikes, and specific preferences the assistant mentions.
2. Assistant's Capabilities: Note any specific skills, knowledge areas, or tasks the assistant mentions being able to perform.
3. Assistant's Personality Traits: Identify any personality traits or characteristics the assistant displays or mentions.
4. Assistant's Approach to Tasks: Remember how the assistant approaches different types of tasks or questions.
5. Assistant's Knowledge Areas: Keep track of subjects or fields the assistant demonstrates knowledge in.
6. Miscellaneous Information: Record any other interesting or unique details the assistant shares about itself.

Return the facts in a JSON format: {"facts": ["fact1", "fact2", ...]}

Remember:
- Today's date is $(_today_str()).
- Extract facts ONLY from the assistant's messages.
- If nothing relevant is found, return {"facts": []}.
- Detect the language and record facts in the same language.

Following is a conversation between the user and the assistant. Extract relevant facts about the assistant.
"""
end

const DEFAULT_UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) ADD a new memory, (2) UPDATE an existing memory, (3) DELETE an existing memory, and (4) NONE if no changes are needed.

Guidelines:
- **ADD**: If the retrieved facts contain a new piece of information not present in the existing memory, you add it.
- **UPDATE**: If the retrieved facts contain information that is already present in the existing memory but with changes or additional details, you update it.
- **DELETE**: If the retrieved facts indicate that certain information in the existing memory is no longer accurate or relevant, you delete it.
- **NONE**: If the retrieved facts do not contain any new or updated information compared to the existing memory, you do nothing.

You will return the output in the following JSON format:
{
    "memory" : [
        {
            "id" : "existing_memory_id_or_new",
            "text" : "The fact text",
            "event" : "ADD" or "UPDATE" or "DELETE" or "NONE",
            "old_memory" : "The old memory text (only for UPDATE)"
        }
    ]
}
"""

const PROCEDURAL_MEMORY_SYSTEM_PROMPT = """You are a memory recording system. Your task is to record complete execution history of agent conversations.

For each conversation, create a detailed numbered list of steps capturing:
1. Every action taken by the agent
2. Every tool call and its parameters
3. Every response or output received
4. The exact sequence and order of operations
5. Any decisions or branches in the execution flow

Record everything exactly as it happened, preserving:
- The full detail of each step
- The exact inputs and outputs
- The order of operations
- Any errors or retries

Format as a numbered step-by-step execution record.
"""

"""
    get_fact_retrieval_messages(messages; is_agent_memory=false)

Build the system + user prompt pair for fact extraction from a conversation.
Returns `(system_prompt, user_content)`.
"""
function get_fact_retrieval_messages(messages::AbstractString; is_agent_memory::Bool=false)
    system_prompt = is_agent_memory ? agent_memory_extraction_prompt() : user_memory_extraction_prompt()
    return (system_prompt, messages)
end

"""
    get_update_memory_messages(existing_memories, new_facts; custom_prompt=nothing)

Build the prompt messages for the memory update decision LLM call.
Returns a vector of `Dict("role" => ..., "content" => ...)`.
"""
function get_update_memory_messages(existing_memories::AbstractString, new_facts::AbstractString;
                                    custom_prompt::Union{Nothing, String}=nothing)
    prompt = something(custom_prompt, DEFAULT_UPDATE_MEMORY_PROMPT)
    user_content = """Existing Memories:
$(existing_memories)

New Retrieved Facts:
$(new_facts)

Based on the above, determine what operations (ADD/UPDATE/DELETE/NONE) need to be performed. Return the result in the specified JSON format."""

    return [
        Dict("role" => "system", "content" => prompt),
        Dict("role" => "user", "content" => user_content),
    ]
end
