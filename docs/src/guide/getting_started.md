# [Getting Started](@id getting_started)

This guide walks you through installing Mem0.jl, understanding its core concepts, and building your first memory-powered application.

## Installation

Mem0.jl is installed directly from its Git repository:

```julia
using Pkg
Pkg.add(url="https://github.com/sdwfrost/Mem0.jl.git")
```

### Prerequisites

- **Julia 1.9+**
- An LLM provider — either:
  - [Ollama](https://ollama.com/) running locally (recommended for getting started)
  - An [OpenAI API key](https://platform.openai.com/api-keys)

If using Ollama, pull the required models:

```bash
ollama pull qwen2.5
ollama pull nomic-embed-text
```

## Core Concepts

Mem0.jl provides a **self-improving memory layer** for AI agents. Here's how the pieces fit together:

### Memory

The [`Memory`](@ref) type is the central orchestrator. It coordinates:

1. **LLM-based fact extraction** — an LLM reads conversations and extracts discrete facts
2. **Embedding generation** — facts are converted to vector representations
3. **Vector storage** — embeddings are stored for fast similarity search
4. **Deduplication** — the LLM decides whether new facts should be added, merged with existing memories, or ignored
5. **History tracking** — every operation is logged in SQLite

### Providers

Mem0.jl uses a provider abstraction for each component:

| Component | Abstract Type | Available Providers |
|-----------|--------------|-------------------|
| LLM | [`AbstractLLM`](@ref) | [`OllamaLLM`](@ref), [`OpenAILLM`](@ref) |
| Embeddings | [`AbstractEmbedder`](@ref) | [`OllamaEmbedding`](@ref), [`OpenAIEmbedding`](@ref) |
| Vector Store | [`AbstractVectorStore`](@ref) | [`InMemoryVectorStore`](@ref) |
| Graph Store | [`AbstractGraphStore`](@ref) | [`InMemoryGraphStore`](@ref), [`Neo4jGraphStore`](@ref) |

### Configuration

All providers are configured through [`MemoryConfig`](@ref), which holds sub-configs for each component. See [Configuration](@ref configuration) for full details.

## Complete Walkthrough

### Step 1: Create a Memory Instance

```julia
using Mem0

config = MemoryConfig(
    llm = LlmConfig(
        provider = "ollama",
        config = Dict("model" => "qwen2.5", "temperature" => 0.1),
    ),
    embedder = EmbedderConfig(
        provider = "ollama",
        config = Dict("model" => "nomic-embed-text", "embedding_dims" => 768),
    ),
    vector_store = VectorStoreConfig(
        provider = "in_memory",
        config = Dict("collection_name" => "my_memories", "embedding_model_dims" => 768),
    ),
)

mem = Memory(config=config)
```

### Step 2: Add Memories

Pass conversation text or message arrays to [`add`](@ref). The LLM extracts facts automatically:

```julia
# Add from a plain string
add(mem, "My name is Alice and I work at Acme Corp as a data scientist."; user_id="alice")

# Add from a conversation (vector of message dicts)
messages = [
    Dict("role" => "user", "content" => "I'm planning a trip to Japan next month."),
    Dict("role" => "assistant", "content" => "That sounds exciting! Have you been before?"),
    Dict("role" => "user", "content" => "No, it's my first time. I love sushi and want to visit Kyoto."),
]
add(mem, messages; user_id="alice")
```

The LLM will extract facts like:
- "Name is Alice"
- "Works at Acme Corp as a data scientist"
- "Planning a trip to Japan next month"
- "Loves sushi"
- "Wants to visit Kyoto"

### Step 3: Search Memories

Use [`search`](@ref) to find relevant memories by semantic similarity:

```julia
results = search(mem, "What does Alice do for work?"; user_id="alice")

for r in results["results"]
    println("Memory: ", r["memory"])
    println("Score:  ", round(r["score"]; digits=3))
    println()
end
```

### Step 4: Retrieve All Memories

Use [`get_all`](@ref) to list every memory for a user:

```julia
all = get_all(mem; user_id="alice")
for r in all["results"]
    println("- ", r["memory"])
end
```

### Step 5: Update and Delete

You can update or delete individual memories by ID:

```julia
# Get a specific memory by ID
memory_id = results["results"][1]["id"]
mem_data = get_memory(mem, memory_id)
println("Current: ", mem_data["memory"])

# Update it
update(mem, memory_id, "Works at Acme Corp as a senior data scientist")

# Delete it
delete(mem, memory_id)
```

### Step 6: View History

Every operation is tracked. Use [`history`](@ref) to see the audit trail:

```julia
h = history(mem, memory_id)
for entry in h
    println(entry["event"], ": ", something(entry["new_memory"], "(deleted)"))
end
```

### Step 7: Reset

To clear all memories and history:

```julia
reset!(mem)
```

## Skipping LLM Inference

If you want to store messages directly without LLM-based fact extraction, pass `infer=false`:

```julia
add(mem, "Raw message to store verbatim"; user_id="alice", infer=false)
```

## Next Steps

- [Configuration](@ref configuration) — customize providers, models, and prompts
- [Providers](@ref providers_guide) — learn about each provider and how to create custom ones
- [Graph Memory](@ref graph_memory) — enable entity-relationship extraction
- [API Reference](@ref memory_api) — full function signatures and docstrings
