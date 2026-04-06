# Abstract LLM interface

"""
    AbstractLLM

Abstract base type for language model providers.
"""
abstract type AbstractLLM end

"""
    LLMResponse

Standardized response from an LLM call.
"""
Base.@kwdef struct LLMResponse
    content::Union{Nothing, String} = nothing
    tool_calls::Vector{Dict{String, Any}} = Dict{String, Any}[]
    role::String = "assistant"
end

"""
    generate_response(llm, messages; response_format=nothing, tools=nothing, tool_choice="auto")

Generate a response from the LLM. Must be implemented by all providers.

# Arguments
- `messages`: Vector of message dicts with "role" and "content" keys
- `response_format`: Optional format constraint (e.g., "json")
- `tools`: Optional vector of tool definitions for function calling
- `tool_choice`: Tool calling mode ("auto", "required", "none")

# Returns
An `LLMResponse` with content and optional tool_calls.
"""
function generate_response end
