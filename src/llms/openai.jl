# OpenAI LLM provider

"""
    OpenAILLM <: AbstractLLM

OpenAI-compatible LLM provider using the Chat Completions API.
"""
Base.@kwdef mutable struct OpenAILLM <: AbstractLLM
    model::String = "gpt-4.1-nano-2025-04-14"
    api_key::Any = get(ENV, "OPENAI_API_KEY", "")
    base_url::String = get(ENV, "OPENAI_API_BASE", "https://api.openai.com/v1")
    temperature::Float64 = 0.1
    max_tokens::Int = 2000
    top_p::Float64 = 1.0
end

function OpenAILLM(config::Dict{String, Any})
    OpenAILLM(
        model = get(config, "model", "gpt-4.1-nano-2025-04-14"),
        api_key = get(config, "api_key", get(ENV, "OPENAI_API_KEY", "")),
        base_url = get(config, "base_url", get(ENV, "OPENAI_API_BASE", "https://api.openai.com/v1")),
        temperature = get(config, "temperature", 0.1),
        max_tokens = get(config, "max_tokens", 2000),
        top_p = get(config, "top_p", 1.0),
    )
end

function _openai_headers(llm::OpenAILLM)
    return Dict(
        "Authorization" => "Bearer $(_resolve_bearer(llm.api_key))",
        "Content-Type" => "application/json",
    )
end

function generate_response(llm::OpenAILLM, messages::Vector;
                           response_format=nothing,
                           tools=nothing,
                           tool_choice="auto")
    url = "$(llm.base_url)/chat/completions"

    body = Dict{String, Any}(
        "model" => llm.model,
        "messages" => messages,
        "temperature" => llm.temperature,
        "max_tokens" => llm.max_tokens,
        "top_p" => llm.top_p,
    )

    if response_format == "json"
        body["response_format"] = Dict("type" => "json_object")
    end

    if tools !== nothing && !isempty(tools)
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    end

    resp = HTTP.post(url,
        _openai_headers(llm),
        JSON3.write(body);
        status_exception=false,
    )

    if resp.status != 200
        throw(Mem0ProviderError("openai", "API request failed ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    choice = data["choices"][1]
    msg = choice["message"]

    content = get(msg, "content", nothing)
    tool_calls_raw = get(msg, "tool_calls", nothing)

    parsed_tools = Dict{String, Any}[]
    if tool_calls_raw !== nothing
        for tc in tool_calls_raw
            fn = tc["function"]
            args = fn["arguments"]
            parsed_args = if args isa AbstractString
                try JSON3.read(args, Dict{String, Any}) catch; Dict{String, Any}("raw" => args) end
            else
                Dict{String, Any}(args)
            end
            push!(parsed_tools, Dict{String, Any}(
                "name" => fn["name"],
                "arguments" => parsed_args,
            ))
        end
    end

    return LLMResponse(content=content, tool_calls=parsed_tools)
end
