# Core types for Mem0.jl

using Dates
using UUIDs

"""Memory type classification."""
@enum MemoryType begin
    SEMANTIC_MEMORY
    EPISODIC_MEMORY
    PROCEDURAL_MEMORY
end

const MEMORY_TYPE_STRINGS = Dict{MemoryType, String}(
    SEMANTIC_MEMORY => "semantic_memory",
    EPISODIC_MEMORY => "episodic_memory",
    PROCEDURAL_MEMORY => "procedural_memory",
)

function memory_type_from_string(s::AbstractString)::MemoryType
    for (mt, str) in MEMORY_TYPE_STRINGS
        str == s && return mt
    end
    throw(Mem0ValidationError("Unknown memory type: $s"))
end

"""
    MemoryItem

A single memory record returned from search/get operations.
"""
Base.@kwdef mutable struct MemoryItem
    id::String = string(uuid4())
    memory::String = ""
    hash::String = ""
    metadata::Dict{String, Any} = Dict{String, Any}()
    score::Union{Nothing, Float64} = nothing
    created_at::Union{Nothing, String} = nothing
    updated_at::Union{Nothing, String} = nothing
    user_id::Union{Nothing, String} = nothing
    agent_id::Union{Nothing, String} = nothing
    run_id::Union{Nothing, String} = nothing
    actor_id::Union{Nothing, String} = nothing
    role::Union{Nothing, String} = nothing
end

"""
    MemoryResult

Result from add/search/get_all operations.
"""
Base.@kwdef struct MemoryResult
    results::Vector{Dict{String, Any}} = Dict{String, Any}[]
    relations::Union{Nothing, Vector{Dict{String, Any}}} = nothing
end

# --- Configuration types ---

"""LLM provider configuration."""
Base.@kwdef mutable struct LlmConfig
    provider::String = "openai"
    config::Dict{String, Any} = Dict{String, Any}(
        "model" => "gpt-4.1-nano-2025-04-14",
        "temperature" => 0.1,
        "max_tokens" => 2000,
    )
end

"""Embedding provider configuration."""
Base.@kwdef mutable struct EmbedderConfig
    provider::String = "openai"
    config::Dict{String, Any} = Dict{String, Any}(
        "model" => "text-embedding-3-small",
        "embedding_dims" => 1536,
    )
end

"""Vector store configuration."""
Base.@kwdef mutable struct VectorStoreConfig
    provider::String = "in_memory"
    config::Dict{String, Any} = Dict{String, Any}(
        "collection_name" => "mem0",
        "embedding_model_dims" => 1536,
    )
end

"""Graph store configuration."""
Base.@kwdef mutable struct GraphStoreConfig
    provider::String = "in_memory"
    config::Union{Nothing, Dict{String, Any}} = nothing
    llm::Union{Nothing, LlmConfig} = nothing
    custom_prompt::Union{Nothing, String} = nothing
    threshold::Float64 = 0.7
end

"""Reranker configuration."""
Base.@kwdef mutable struct RerankerConfig
    provider::String = "llm"
    config::Union{Nothing, Dict{String, Any}} = nothing
end

"""
    MemoryConfig

Top-level configuration for the Memory system.
"""
Base.@kwdef mutable struct MemoryConfig
    llm::LlmConfig = LlmConfig()
    embedder::EmbedderConfig = EmbedderConfig()
    vector_store::VectorStoreConfig = VectorStoreConfig()
    graph_store::GraphStoreConfig = GraphStoreConfig()
    reranker::Union{Nothing, RerankerConfig} = nothing
    history_db_path::String = ":memory:"
    version::String = "v1.1"
    custom_fact_extraction_prompt::Union{Nothing, String} = nothing
    custom_update_memory_prompt::Union{Nothing, String} = nothing
end

# --- Message helpers ---

"""A chat message with role and content."""
Base.@kwdef struct ChatMessage
    role::String = "user"
    content::String = ""
end
