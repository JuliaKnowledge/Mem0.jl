# [Providers](@id providers_guide)

Mem0.jl uses a provider abstraction to decouple the memory system from specific backend implementations. Each component — LLM, embedder, vector store, and graph store — has an abstract base type and one or more concrete providers.

## LLM Providers

LLM providers generate text responses for fact extraction, memory update decisions, and entity/relationship extraction. All providers implement the [`generate_response`](@ref) method.

### OllamaLLM

[`OllamaLLM`](@ref) connects to a locally running [Ollama](https://ollama.com/) server. No API key required — ideal for local development and privacy-sensitive deployments.

```julia
config = LlmConfig(
    provider = "ollama",
    config = Dict(
        "model" => "qwen2.5",           # any Ollama model
        "base_url" => "http://localhost:11434",
        "temperature" => 0.1,
        "max_tokens" => 2000,
        "top_p" => 1.0,
    ),
)
```

Default model: `llama3.1`. The `base_url` defaults to `ENV["OLLAMA_HOST"]` or `http://localhost:11434`.

!!! tip
    For best results with fact extraction, use a model with strong JSON output support such as `qwen2.5`, `llama3.1`, or `mistral`.

### OpenAILLM

[`OpenAILLM`](@ref) uses the OpenAI Chat Completions API. Compatible with any OpenAI-compatible endpoint (Azure OpenAI, vLLM, LiteLLM, etc.) by setting a custom `base_url`.

```julia
config = LlmConfig(
    provider = "openai",
    config = Dict(
        "model" => "gpt-4.1-nano-2025-04-14",
        "api_key" => ENV["OPENAI_API_KEY"],
        "base_url" => "https://api.openai.com/v1",
        "temperature" => 0.1,
        "max_tokens" => 2000,
        "top_p" => 1.0,
    ),
)
```

Default model: `gpt-4.1-nano-2025-04-14`. The `api_key` defaults to `ENV["OPENAI_API_KEY"]`.

## Embedding Providers

Embedding providers convert text into dense vector representations for similarity search. All providers implement the [`embed`](@ref) method.

### OllamaEmbedding

[`OllamaEmbedding`](@ref) uses Ollama's embedding API with local models:

```julia
config = EmbedderConfig(
    provider = "ollama",
    config = Dict(
        "model" => "nomic-embed-text",   # 768-dimensional embeddings
        "base_url" => "http://localhost:11434",
        "embedding_dims" => 768,
    ),
)
```

Default model: `nomic-embed-text` (768 dimensions).

### OpenAIEmbedding

[`OpenAIEmbedding`](@ref) uses the OpenAI Embeddings API:

```julia
config = EmbedderConfig(
    provider = "openai",
    config = Dict(
        "model" => "text-embedding-3-small",
        "api_key" => ENV["OPENAI_API_KEY"],
        "embedding_dims" => 1536,
    ),
)
```

Default model: `text-embedding-3-small` (1536 dimensions). The `dimensions` parameter is automatically sent for `text-embedding-3-*` models.

## Vector Store Providers

Vector stores hold embeddings and support similarity search with metadata filtering. They implement `insert!`, `search`, `get`, `delete!`, `update!`, `list_records`, and `reset!`.

### InMemoryVectorStore

[`InMemoryVectorStore`](@ref) is a simple in-memory store using brute-force cosine similarity. Suitable for development, testing, and small-scale deployments.

```julia
config = VectorStoreConfig(
    provider = "in_memory",
    config = Dict(
        "collection_name" => "my_memories",
        "embedding_model_dims" => 768,
    ),
)
```

!!! warning
    Data in `InMemoryVectorStore` is lost when the Julia process exits. For persistence, consider implementing a custom vector store provider backed by a database.

## Graph Store Providers

Graph stores maintain entity-relationship networks extracted from conversations. They implement [`add_to_graph!`](@ref), [`search_graph`](@ref), [`delete_from_graph!`](@ref), [`delete_all_graph!`](@ref), and [`get_all_graph`](@ref).

### InMemoryGraphStore

[`InMemoryGraphStore`](@ref) stores nodes and edges in dictionaries with adjacency-list semantics. Supports entity embeddings for semantic matching, soft deletion, and multi-tenant filtering.

```julia
config = MemoryConfig(
    # ... llm and embedder config ...
    graph_store = GraphStoreConfig(
        provider = "in_memory",
        config = Dict{String, Any}(),   # non-nothing enables graph
        threshold = 0.7,
    ),
)
```

### Neo4jGraphStore

[`Neo4jGraphStore`](@ref) persists the knowledge graph in a Neo4j database via the HTTP Transactional Cypher API. Uses `MERGE` for upsert semantics and stores entity embeddings as node properties.

```julia
config = MemoryConfig(
    # ... llm and embedder config ...
    graph_store = GraphStoreConfig(
        provider = "neo4j",
        config = Dict{String, Any}(
            "url" => "http://localhost:7474",
            "username" => "neo4j",
            "password" => "your-password",
            "database" => "neo4j",
        ),
        threshold = 0.7,
    ),
)
```

The `url` field accepts `http://`, `https://`, `bolt://`, and `neo4j://` URLs. For bolt/neo4j URLs, the host is extracted and HTTP port 7474 is used.

See [Graph Memory](@ref graph_memory) for a detailed guide.

## Factory Functions

Providers are instantiated through factory functions that read the config and return the appropriate concrete type:

- [`create_llm`](@ref) — creates an [`AbstractLLM`](@ref) from [`LlmConfig`](@ref)
- [`create_embedder`](@ref) — creates an [`AbstractEmbedder`](@ref) from [`EmbedderConfig`](@ref)
- [`create_vector_store`](@ref) — creates an [`AbstractVectorStore`](@ref) from [`VectorStoreConfig`](@ref)
- [`create_graph_store`](@ref) — creates an [`AbstractGraphStore`](@ref) from [`GraphStoreConfig`](@ref)

You typically don't call these directly — the [`Memory`](@ref) constructor calls them for you. They're useful when building custom pipelines.

## Implementing Custom Providers

### Custom LLM

To create a custom LLM provider:

1. Define a struct that subtypes [`AbstractLLM`](@ref)
2. Implement a `Dict{String, Any}` constructor for the factory pattern
3. Implement [`generate_response`](@ref)
4. Register with [`register_llm_provider!`](@ref)

```julia
using Mem0

struct MyLLM <: AbstractLLM
    model::String
    endpoint::String
end

function MyLLM(config::Dict{String, Any})
    MyLLM(
        get(config, "model", "default-model"),
        get(config, "endpoint", "http://localhost:8080"),
    )
end

function Mem0.generate_response(llm::MyLLM, messages::Vector;
                                response_format=nothing, tools=nothing,
                                tool_choice="auto")
    # Call your LLM API here...
    # Must return an LLMResponse
    return LLMResponse(content="response text", tool_calls=Dict{String, Any}[])
end

register_llm_provider!("my_llm", MyLLM)
```

### Custom Embedder

1. Subtype [`AbstractEmbedder`](@ref)
2. Implement [`embed`](@ref)
3. Register with [`register_embedder_provider!`](@ref)

```julia
struct MyEmbedder <: AbstractEmbedder
    model::String
    dims::Int
end

MyEmbedder(config::Dict{String, Any}) = MyEmbedder(
    get(config, "model", "default"),
    get(config, "embedding_dims", 384),
)

function Mem0.embed(emb::MyEmbedder, text::AbstractString; memory_action=nothing)::Vector{Float64}
    # Generate embedding vector here...
    return zeros(Float64, emb.dims)  # placeholder
end

register_embedder_provider!("my_embedder", MyEmbedder)
```

### Custom Vector Store

1. Subtype [`AbstractVectorStore`](@ref)
2. Implement the required interface methods:
   - `Base.insert!(store, vectors, payloads, ids)`
   - `Mem0.search(store, query_vector, limit; filters, threshold)`
   - `Base.delete!(store, vector_id)`
   - `Mem0.update!(store, vector_id; vector, payload)`
   - `Base.get(store, vector_id)`
   - `Mem0.list_records(store; filters, limit)`
   - `Mem0.reset!(store)`
3. Register with [`register_vector_store_provider!`](@ref)

```julia
struct MyVectorStore <: AbstractVectorStore
    # your fields...
end

MyVectorStore(config::Dict{String, Any}) = MyVectorStore(...)

# Implement all required methods...

register_vector_store_provider!("my_store", MyVectorStore)
```
