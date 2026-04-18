# Mem0.jl

*A self-improving memory layer for AI agents in Julia.*

Mem0.jl is a Julia port of [mem0](https://github.com/mem0ai/mem0) — providing LLM-based fact extraction, semantic vector search, optional graph memory, and full change history tracking. It enables AI agents and applications to remember user preferences, past interactions, and contextual knowledge across sessions.

## Features

- **LLM-powered fact extraction** — automatically distills conversations into discrete, searchable facts
- **Semantic vector search** — find relevant memories using cosine similarity over embeddings
- **Graph memory** — optional entity-relationship extraction for structured knowledge (in-memory or Neo4j)
- **History tracking** — SQLite-backed audit trail of every add, update, and delete operation
- **Intelligent deduplication** — LLM-driven decisions to add, update, or delete memories based on new information
- **Multi-tenant** — filter memories by `user_id`, `agent_id`, or `run_id`
- **Multiple providers** — supports OpenAI and Ollama for both LLM and embedding backends
- **Extensible factory pattern** — register custom LLM, embedding, and vector store providers

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sdwfrost/Mem0.jl.git")
```

## Quick Start

```julia
using Mem0

# Configure with Ollama (local, no API key needed)
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "demo", "embedding_model_dims" => 768)),
)
mem = Memory(config=config)

# Add memories from conversations
add(mem, "I love programming in Julia and Python"; user_id="alice")
add(mem, "My favorite editor is VS Code"; user_id="alice")

# Search memories by semantic similarity
results = search(mem, "What programming languages does she like?"; user_id="alice")
for r in results["results"]
    println(r["memory"], " (score: ", round(r["score"]; digits=3), ")")
end

# Retrieve all memories for a user
all_memories = get_all(mem; user_id="alice")
```

## Documentation Overview

### [Guide](@ref getting_started)

- [Getting Started](@ref getting_started) — installation, basic concepts, and a complete walkthrough
- [Configuration](@ref configuration) — all configuration options explained
- [Providers](@ref providers_guide) — LLM, embedding, vector store, and graph store providers
- [Graph Memory](@ref graph_memory) — entity-relationship extraction and graph-based search

### [API Reference](@ref memory_api)

- [Memory](@ref memory_api) — the main `Memory` type and its public methods
- [Types](@ref types_api) — configuration structs, data types, and enums
- [Providers](@ref providers_api) — abstract interfaces and concrete provider implementations
- [Utilities](@ref utilities_api) — helper functions for parsing, hashing, filtering, and more

## Module reference

```@docs
Mem0
```
