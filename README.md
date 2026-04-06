# Mem0.jl

Julia port of [mem0](https://github.com/mem0ai/mem0) — a self-improving memory layer for AI agents.

## Features

- **LLM-based fact extraction**: Automatically extracts atomic facts from conversations
- **Semantic search**: Find relevant memories using vector similarity
- **Memory deduplication**: LLM-driven ADD/UPDATE/DELETE decisions against existing memories
- **Graph memory**: Optional entity-relationship graph for structured knowledge
- **History tracking**: SQLite-backed audit trail for all memory operations
- **Multi-tenant**: Scope memories by `user_id`, `agent_id`, and `run_id`

## Providers

| Component | Providers |
|-----------|-----------|
| LLM | OpenAI, Ollama |
| Embeddings | OpenAI, Ollama |
| Vector Store | In-memory (cosine similarity) |
| Graph Store | In-memory (adjacency lists) |

## Quick Start

```julia
using Mem0

# Configure with Ollama (fully local, no API keys)
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict(
        "model" => "nomic-embed-text",
        "embedding_dims" => 768,
    )),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict(
        "collection_name" => "my_memories",
        "embedding_model_dims" => 768,
    )),
)

mem = Mem0.Memory(config=config)

# Add memories from a conversation
add(mem, [
    Dict("role" => "user", "content" => "Hi, I'm Alice. I love Julia programming."),
    Dict("role" => "assistant", "content" => "Nice to meet you, Alice!"),
]; user_id="alice")

# Search memories
results = Mem0.search(mem, "What programming languages does Alice like?"; user_id="alice")
for r in results["results"]
    println(r["memory"], " (score: ", round(r["score"], digits=3), ")")
end

# Get all memories
all = get_all(mem; user_id="alice")

# View change history
h = history(mem, results["results"][1]["id"])
```

## Configuration

### OpenAI (cloud)

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="openai", config=Dict(
        "model" => "gpt-4.1-nano-2025-04-14",
        "api_key" => ENV["OPENAI_API_KEY"],
    )),
    embedder = EmbedderConfig(provider="openai", config=Dict(
        "model" => "text-embedding-3-small",
        "embedding_dims" => 1536,
        "api_key" => ENV["OPENAI_API_KEY"],
    )),
)
```

### With Graph Memory

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "mem", "embedding_model_dims" => 768)),
    graph_store = GraphStoreConfig(provider="in_memory", config=Dict{String,Any}(), threshold=0.7),
)

mem = Mem0.Memory(config=config)
result = add(mem, "Alice works at Microsoft"; user_id="alice")
# result["relations"] contains extracted entity relationships
```

## API

| Function | Description |
|----------|-------------|
| `add(mem, messages; user_id, agent_id, run_id, infer)` | Extract and store memories |
| `Mem0.search(mem, query; user_id, limit, threshold)` | Semantic memory search |
| `update(mem, memory_id, new_text)` | Update a memory |
| `delete(mem, memory_id)` | Delete a memory |
| `get_memory(mem, memory_id)` | Retrieve a single memory |
| `get_all(mem; user_id)` | List all memories |
| `history(mem, memory_id)` | View change history |
| `Mem0.reset!(mem)` | Clear all memories |

## Extending

Register custom providers:

```julia
register_llm_provider!("my_llm", MyLLMType)
register_embedder_provider!("my_embedder", MyEmbedderType)
register_vector_store_provider!("my_store", MyStoreType)
```

Custom providers must implement `generate_response(llm, messages; ...)` for LLMs,
`embed(embedder, text; ...)` for embedders, and the vector store interface methods.

## License

MIT
