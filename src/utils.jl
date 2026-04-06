# Utility functions for Mem0.jl

using SHA
using JSON3

"""
    parse_messages(messages)

Convert a vector of message dicts to a formatted string with role prefixes.
"""
function parse_messages(messages::Vector)::String
    lines = String[]
    for msg in messages
        role = get(msg, "role", get(msg, :role, "user"))
        content = get(msg, "content", get(msg, :content, ""))
        push!(lines, "$(titlecase(string(role))): $content")
    end
    return join(lines, "\n")
end

parse_messages(msg::AbstractString) = msg

"""
    normalize_facts(raw_facts)

Normalize LLM output of facts into a clean vector of strings.
Handles strings, dicts with "fact"/"text" keys, and other formats.
"""
function normalize_facts(raw_facts)::Vector{String}
    result = String[]
    for fact in raw_facts
        if fact isa AbstractString
            s = strip(fact)
            !isempty(s) && push!(result, s)
        elseif fact isa AbstractDict
            for key in ("fact", "text", "memory", "content")
                if haskey(fact, key)
                    s = strip(string(fact[key]))
                    !isempty(s) && push!(result, s)
                    break
                end
            end
        else
            s = strip(string(fact))
            !isempty(s) && push!(result, s)
        end
    end
    return result
end

"""
    remove_code_blocks(content)

Remove markdown code blocks and <think> tags from content.
"""
function remove_code_blocks(content::AbstractString)::String
    s = content
    # Remove ```json ... ``` blocks
    s = replace(s, r"```(?:json)?\s*\n?"s => "")
    s = replace(s, r"```\s*"s => "")
    # Remove <think>...</think> blocks
    s = replace(s, r"<think>.*?</think>"s => "")
    return strip(s)
end

"""
    extract_json(text)

Extract JSON from text that may contain code blocks or other wrapping.
Returns a parsed Dict/Vector or `nothing` if no valid JSON found.
"""
function extract_json(text::AbstractString)
    cleaned = remove_code_blocks(text)
    # Try direct parse first
    try
        return JSON3.read(cleaned, Dict{String, Any})
    catch
    end
    # Try to find JSON in braces
    m = match(r"\{[^{}]*\}"s, cleaned)
    if m !== nothing
        try
            return JSON3.read(m.match, Dict{String, Any})
        catch
        end
    end
    # Try to find JSON array
    m = match(r"\[.*\]"s, cleaned)
    if m !== nothing
        try
            return JSON3.read(m.match)
        catch
        end
    end
    return nothing
end

"""
    memory_hash(text)

Generate an MD5 hash string for memory deduplication.
"""
function memory_hash(text::AbstractString)::String
    return bytes2hex(sha256(text))
end

"""
    format_entities(entities)

Format entity relationships as `source -- relationship -- destination` lines.
"""
function format_entities(entities::Vector)::String
    lines = String[]
    for e in entities
        src = get(e, "source", get(e, :source, ""))
        rel = get(e, "relationship", get(e, :relationship, ""))
        dst = get(e, "destination", get(e, :destination, ""))
        push!(lines, "$src -- $rel -- $dst")
    end
    return join(lines, "\n")
end

"""
    normalize_entity(name)

Normalize an entity name: lowercase, replace spaces with underscores.
"""
function normalize_entity(name::AbstractString)::String
    return replace(lowercase(strip(name)), " " => "_")
end

"""
    build_filters_and_metadata(; user_id=nothing, agent_id=nothing, run_id=nothing,
                                  input_metadata=nothing, input_filters=nothing)

Build metadata template and query filters from session IDs.
Returns `(metadata::Dict, filters::Dict)`.
"""
function build_filters_and_metadata(;
    user_id::Union{Nothing, String}=nothing,
    agent_id::Union{Nothing, String}=nothing,
    run_id::Union{Nothing, String}=nothing,
    input_metadata::Union{Nothing, Dict}=nothing,
    input_filters::Union{Nothing, Dict}=nothing,
)
    if user_id === nothing && agent_id === nothing && run_id === nothing
        throw(Mem0ValidationError(
            "At least one of user_id, agent_id, or run_id must be provided.",
            "MISSING_FILTER",
            "Pass user_id=\"...\", agent_id=\"...\", or run_id=\"...\"."
        ))
    end

    metadata = Dict{String, Any}()
    filters = Dict{String, Any}()

    if user_id !== nothing
        metadata["user_id"] = user_id
        filters["user_id"] = user_id
    end
    if agent_id !== nothing
        metadata["agent_id"] = agent_id
        filters["agent_id"] = agent_id
    end
    if run_id !== nothing
        metadata["run_id"] = run_id
        filters["run_id"] = run_id
    end

    # Merge input metadata
    if input_metadata !== nothing
        merge!(metadata, input_metadata)
    end

    # Merge input filters
    if input_filters !== nothing
        merge!(filters, input_filters)
    end

    return (metadata, filters)
end

"""
    cosine_similarity(a, b)

Compute cosine similarity between two vectors.
"""
function cosine_similarity(a::AbstractVector, b::AbstractVector)::Float64
    na = norm(a)
    nb = norm(b)
    (na == 0.0 || nb == 0.0) && return 0.0
    return dot(a, b) / (na * nb)
end

using LinearAlgebra: norm, dot

"""
    now_iso()

Return the current time as an ISO 8601 string.
"""
function now_iso()::String
    return Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
end
