# Ollama LLM provider

"""
    OllamaLLM <: AbstractLLM

Ollama-compatible LLM provider using the local Ollama API.
"""
Base.@kwdef mutable struct OllamaLLM <: AbstractLLM
    model::String = "llama3.1"
    base_url::String = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
    temperature::Float64 = 0.1
    max_tokens::Int = 2000
    top_p::Float64 = 1.0
end

function OllamaLLM(config::Dict{String, Any})
    OllamaLLM(
        model = get(config, "model", "llama3.1"),
        base_url = get(config, "base_url", get(ENV, "OLLAMA_HOST", "http://localhost:11434")),
        temperature = get(config, "temperature", 0.1),
        max_tokens = get(config, "max_tokens", 2000),
        top_p = get(config, "top_p", 1.0),
    )
end

function generate_response(llm::OllamaLLM, messages::Vector;
                           response_format=nothing,
                           tools=nothing,
                           tool_choice="auto")
    url = "$(llm.base_url)/api/chat"

    body = Dict{String, Any}(
        "model" => llm.model,
        "messages" => messages,
        "stream" => false,
        "options" => Dict{String, Any}(
            "temperature" => llm.temperature,
            "num_predict" => llm.max_tokens,
            "top_p" => llm.top_p,
        ),
    )

    if response_format == "json"
        body["format"] = "json"
        # Inject JSON instruction if not present in messages
        _inject_json_instruction!(body["messages"])
    end

    if tools !== nothing && !isempty(tools)
        body["tools"] = tools
    end

    resp = HTTP.post(url,
        ["Content-Type" => "application/json"],
        JSON3.write(body);
        status_exception=false,
    )

    if resp.status != 200
        throw(Mem0ProviderError("ollama", "API request failed ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    msg = data["message"]
    content = get(msg, "content", nothing)
    tool_calls_raw = get(msg, "tool_calls", nothing)

    parsed_tools = Dict{String, Any}[]
    if tool_calls_raw !== nothing
        for tc in tool_calls_raw
            fn = get(tc, "function", tc)
            args = get(fn, "arguments", Dict{String, Any}())
            parsed_args = if args isa AbstractString
                try JSON3.read(args, Dict{String, Any}) catch; Dict{String, Any}("raw" => args) end
            else
                Dict{String, Any}(args)
            end
            push!(parsed_tools, Dict{String, Any}(
                "name" => get(fn, "name", "unknown"),
                "arguments" => parsed_args,
            ))
        end
    end

    return LLMResponse(content=content, tool_calls=parsed_tools)
end

function _inject_json_instruction!(messages::Vector)
    has_json = any(messages) do m
        c = get(m, "content", "")
        c isa AbstractString && occursin("json", lowercase(c))
    end
    if !has_json && !isempty(messages)
        last_msg = messages[end]
        old_content = get(last_msg, "content", "")
        messages[end] = Dict{String, Any}(
            "role" => get(last_msg, "role", "user"),
            "content" => old_content * "\n\nPlease respond in valid JSON format."
        )
    end
end
