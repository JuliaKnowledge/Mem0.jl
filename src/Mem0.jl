"""
    Mem0

Julia port of mem0 — a self-improving memory layer for AI agents.

Provides LLM-based fact extraction, semantic search, optional graph memory,
and change history tracking. Supports OpenAI and Ollama as LLM/embedding
providers with an extensible factory pattern.

# Quick start
```julia
using Mem0

config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "test", "embedding_model_dims" => 768)),
)
mem = Memory(config=config)

add(mem, "I love programming in Julia"; user_id="alice")
results = search(mem, "What programming languages?"; user_id="alice")
```
"""
module Mem0

# Standard library
using Dates
using UUIDs
using LinearAlgebra: norm, dot, normalize

# External dependencies
using HTTP
using JSON3
using SQLite

# Core files
include("exceptions.jl")
include("types.jl")
include("prompts.jl")
include("utils.jl")
include("auth.jl")
include("storage.jl")

# Provider abstractions
include("llms/base.jl")
include("llms/openai.jl")
include("llms/ollama.jl")

include("embeddings/base.jl")
include("embeddings/openai.jl")
include("embeddings/ollama.jl")

include("vector_stores/base.jl")
include("vector_stores/in_memory.jl")

include("graphs/base.jl")
include("graphs/tools.jl")
include("graphs/memory_graph.jl")
include("graphs/neo4j_graph.jl")

# Factory and main
include("factory.jl")
include("memory.jl")

# Exports — types
export MemoryConfig, LlmConfig, EmbedderConfig, VectorStoreConfig, GraphStoreConfig, RerankerConfig
export MemoryItem, MemoryResult, ChatMessage, MemoryType
export SEMANTIC_MEMORY, EPISODIC_MEMORY, PROCEDURAL_MEMORY

# Exports — abstract types
export AbstractLLM, AbstractEmbedder, AbstractVectorStore, AbstractGraphStore

# Exports — concrete providers
export OpenAILLM, OllamaLLM
export OpenAIEmbedding, OllamaEmbedding
export InMemoryVectorStore
export InMemoryGraphStore
export Neo4jGraphStore

# Exports — core types
export VectorRecord, LLMResponse, HistoryManager

# Exports — Memory API
export add, update, delete, get_memory, get_all, history

# Exports — also export list_records and remove_code_blocks
export list_records, remove_code_blocks

# Exports — utilities
export generate_response, embed
export parse_messages, normalize_facts, extract_json, memory_hash
export build_filters_and_metadata, cosine_similarity, normalize_entity, now_iso

# Exports — factory
export create_llm, create_embedder, create_vector_store, create_graph_store
export register_llm_provider!, register_embedder_provider!, register_vector_store_provider!

# Exports — storage
export add_history!, get_history, reset_history!

# Exports — graph
export add_to_graph!, search_graph, delete_from_graph!, delete_all_graph!, get_all_graph

# Exports — exceptions
export Mem0Error, Mem0ValidationError, Mem0ProviderError

end # module Mem0
