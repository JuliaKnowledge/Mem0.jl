# [Configuration](@id configuration)

Mem0.jl is configured through a hierarchy of config structs. The top-level [`MemoryConfig`](@ref) holds sub-configurations for each component.

## MemoryConfig

[`MemoryConfig`](@ref) is the single entry point for all configuration:

```julia
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
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | [`LlmConfig`](@ref) | OpenAI gpt-4.1-nano | LLM provider for fact extraction and memory decisions |
| `embedder` | [`EmbedderConfig`](@ref) | OpenAI text-embedding-3-small | Embedding provider for vectorizing memories |
| `vector_store` | [`VectorStoreConfig`](@ref) | In-memory store | Backend for vector storage and similarity search |
| `graph_store` | [`GraphStoreConfig`](@ref) | In-memory (disabled) | Optional graph memory for entity-relationship storage |
| `reranker` | [`RerankerConfig`](@ref) or `nothing` | `nothing` | Optional reranker configuration |
| `history_db_path` | `String` | `":memory:"` | SQLite database path for history (`":memory:"` for in-memory) |
| `version` | `String` | `"v1.1"` | Configuration version string |
| `custom_fact_extraction_prompt` | `String` or `nothing` | `nothing` | Override the default fact extraction system prompt |
| `custom_update_memory_prompt` | `String` or `nothing` | `nothing` | Override the default memory update decision prompt |

## LlmConfig

Configures the language model used for fact extraction and memory management decisions:

```julia
Base.@kwdef mutable struct LlmConfig
    provider::String = "openai"
    config::Dict{String, Any} = Dict{String, Any}(
        "model" => "gpt-4.1-nano-2025-04-14",
        "temperature" => 0.1,
        "max_tokens" => 2000,
    )
end
```

### Provider-specific config keys

**OpenAI** (`provider = "openai"`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"model"` | String | `"gpt-4.1-nano-2025-04-14"` | Model name |
| `"api_key"` | String | `ENV["OPENAI_API_KEY"]` | API key |
| `"base_url"` | String | `"https://api.openai.com/v1"` | API base URL |
| `"temperature"` | Float64 | `0.1` | Sampling temperature |
| `"max_tokens"` | Int | `2000` | Maximum tokens in response |
| `"top_p"` | Float64 | `1.0` | Nucleus sampling parameter |

**Ollama** (`provider = "ollama"`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"model"` | String | `"llama3.1"` | Model name |
| `"base_url"` | String | `ENV["OLLAMA_HOST"]` or `"http://localhost:11434"` | Ollama API URL |
| `"temperature"` | Float64 | `0.1` | Sampling temperature |
| `"max_tokens"` | Int | `2000` | Maximum tokens (`num_predict` in Ollama) |
| `"top_p"` | Float64 | `1.0` | Nucleus sampling parameter |

## EmbedderConfig

Configures the embedding model used for vectorizing memories and queries:

```julia
Base.@kwdef mutable struct EmbedderConfig
    provider::String = "openai"
    config::Dict{String, Any} = Dict{String, Any}(
        "model" => "text-embedding-3-small",
        "embedding_dims" => 1536,
    )
end
```

### Provider-specific config keys

**OpenAI** (`provider = "openai"`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"model"` | String | `"text-embedding-3-small"` | Embedding model name |
| `"api_key"` | String | `ENV["OPENAI_API_KEY"]` | API key |
| `"base_url"` | String | `"https://api.openai.com/v1"` | API base URL |
| `"embedding_dims"` | Int | `1536` | Output embedding dimensions |

**Ollama** (`provider = "ollama"`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"model"` | String | `"nomic-embed-text"` | Embedding model name |
| `"base_url"` | String | `ENV["OLLAMA_HOST"]` or `"http://localhost:11434"` | Ollama API URL |
| `"embedding_dims"` | Int | `768` | Output embedding dimensions |

!!! warning "Matching dimensions"
    The `embedding_dims` in `EmbedderConfig` must match the `embedding_model_dims` in `VectorStoreConfig`. A mismatch will cause incorrect similarity scores.

## VectorStoreConfig

Configures the vector storage backend:

```julia
Base.@kwdef mutable struct VectorStoreConfig
    provider::String = "in_memory"
    config::Dict{String, Any} = Dict{String, Any}(
        "collection_name" => "mem0",
        "embedding_model_dims" => 1536,
    )
end
```

**In-memory** (`provider = "in_memory"`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"collection_name"` | String | `"mem0"` | Name for the collection |
| `"embedding_model_dims"` | Int | `1536` | Expected embedding dimensions |

## GraphStoreConfig

Configures the optional graph memory backend. Graph memory is **disabled by default** — set `config` to a non-`nothing` dict to enable it:

```julia
Base.@kwdef mutable struct GraphStoreConfig
    provider::String = "in_memory"
    config::Union{Nothing, Dict{String, Any}} = nothing
    llm::Union{Nothing, LlmConfig} = nothing
    custom_prompt::Union{Nothing, String} = nothing
    threshold::Float64 = 0.7
end
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | String | `"in_memory"` | Graph backend: `"in_memory"` or `"neo4j"` |
| `config` | Dict or `nothing` | `nothing` | Provider config; `nothing` disables graph memory |
| `llm` | [`LlmConfig`](@ref) or `nothing` | `nothing` | Override LLM for graph extraction (falls back to main LLM) |
| `custom_prompt` | String or `nothing` | `nothing` | Custom prompt for relationship extraction |
| `threshold` | Float64 | `0.7` | Cosine similarity threshold for entity matching |

See [Graph Memory](@ref graph_memory) for detailed setup.

## RerankerConfig

Configuration for an optional result reranker:

```julia
Base.@kwdef mutable struct RerankerConfig
    provider::String = "llm"
    config::Union{Nothing, Dict{String, Any}} = nothing
end
```

## Configuration Examples

### Minimal Ollama Setup

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "test", "embedding_model_dims" => 768)),
)
```

### OpenAI with Persistent History

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="openai", config=Dict(
        "model" => "gpt-4.1-nano-2025-04-14",
        "api_key" => ENV["OPENAI_API_KEY"],
    )),
    embedder = EmbedderConfig(provider="openai", config=Dict(
        "model" => "text-embedding-3-small",
        "embedding_dims" => 1536,
    )),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict(
        "collection_name" => "production",
        "embedding_model_dims" => 1536,
    )),
    history_db_path = "memory_history.sqlite",
)
```

### With Graph Memory Enabled

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "test", "embedding_model_dims" => 768)),
    graph_store = GraphStoreConfig(
        provider = "in_memory",
        config = Dict{String, Any}(),  # non-nothing enables graph memory
        threshold = 0.7,
    ),
)
```

### Custom Fact Extraction Prompt

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "test", "embedding_model_dims" => 768)),
    custom_fact_extraction_prompt = """
    You are a medical information organizer. Extract only medically relevant facts
    from conversations: symptoms, diagnoses, medications, allergies, and vital signs.
    Return as JSON: {"facts": ["fact1", "fact2", ...]}
    """,
)
```

## Environment Variables

Mem0.jl reads the following environment variables as defaults:

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | [`OpenAILLM`](@ref), [`OpenAIEmbedding`](@ref) | OpenAI API authentication key |
| `OPENAI_API_BASE` | [`OpenAILLM`](@ref), [`OpenAIEmbedding`](@ref) | Custom OpenAI-compatible API base URL |
| `OLLAMA_HOST` | [`OllamaLLM`](@ref), [`OllamaEmbedding`](@ref) | Ollama server URL (default: `http://localhost:11434`) |

## Custom Provider Registration

You can register custom providers at runtime using the factory registration functions:

```julia
using Mem0

# Define a custom LLM type
struct MyCustomLLM <: AbstractLLM
    model::String
end
MyCustomLLM(config::Dict{String, Any}) = MyCustomLLM(get(config, "model", "default"))

# Implement the required interface
function Mem0.generate_response(llm::MyCustomLLM, messages::Vector; kwargs...)
    # ... your implementation ...
end

# Register it
register_llm_provider!("my_custom", MyCustomLLM)

# Now use it in config
config = MemoryConfig(
    llm = LlmConfig(provider="my_custom", config=Dict("model" => "my-model")),
    # ...
)
```

See [Providers](@ref providers_guide) for full details on implementing custom providers.
