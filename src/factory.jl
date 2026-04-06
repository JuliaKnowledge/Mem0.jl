# Provider factory pattern for dynamic instantiation

"""Provider registries mapping provider name → constructor."""
const LLM_PROVIDERS = Dict{String, Any}(
    "openai" => OpenAILLM,
    "ollama" => OllamaLLM,
)

const EMBEDDER_PROVIDERS = Dict{String, Any}(
    "openai" => OpenAIEmbedding,
    "ollama" => OllamaEmbedding,
)

const VECTOR_STORE_PROVIDERS = Dict{String, Any}(
    "in_memory" => InMemoryVectorStore,
)

"""
    create_llm(config::LlmConfig) → AbstractLLM

Instantiate an LLM provider from configuration.
"""
function create_llm(config::LlmConfig)::AbstractLLM
    provider = config.provider
    haskey(LLM_PROVIDERS, provider) || throw(Mem0Error("Unsupported LLM provider: $provider. Available: $(join(keys(LLM_PROVIDERS), ", "))"))
    return LLM_PROVIDERS[provider](config.config)
end

"""
    create_embedder(config::EmbedderConfig) → AbstractEmbedder

Instantiate an embedding provider from configuration.
"""
function create_embedder(config::EmbedderConfig)::AbstractEmbedder
    provider = config.provider
    haskey(EMBEDDER_PROVIDERS, provider) || throw(Mem0Error("Unsupported embedder provider: $provider. Available: $(join(keys(EMBEDDER_PROVIDERS), ", "))"))
    return EMBEDDER_PROVIDERS[provider](config.config)
end

"""
    create_vector_store(config::VectorStoreConfig) → AbstractVectorStore

Instantiate a vector store from configuration.
"""
function create_vector_store(config::VectorStoreConfig)::AbstractVectorStore
    provider = config.provider
    haskey(VECTOR_STORE_PROVIDERS, provider) || throw(Mem0Error("Unsupported vector store provider: $provider. Available: $(join(keys(VECTOR_STORE_PROVIDERS), ", "))"))
    return VECTOR_STORE_PROVIDERS[provider](config.config)
end

"""
    create_graph_store(config::GraphStoreConfig, llm_config::LlmConfig, embedder_config::EmbedderConfig) → AbstractGraphStore

Instantiate a graph store from configuration.
"""
function create_graph_store(config::GraphStoreConfig, llm_config::LlmConfig,
                             embedder_config::EmbedderConfig)::AbstractGraphStore
    graph_llm = config.llm !== nothing ? create_llm(config.llm) : create_llm(llm_config)
    graph_embedder = create_embedder(embedder_config)

    if config.provider == "in_memory"
        return InMemoryGraphStore(
            llm = graph_llm,
            embedder = graph_embedder,
            threshold = config.threshold,
            custom_prompt = config.custom_prompt,
        )
    elseif config.provider == "neo4j"
        return Neo4jGraphStore(
            llm = graph_llm,
            embedder = graph_embedder,
            config = something(config.config, Dict{String, Any}()),
            threshold = config.threshold,
            custom_prompt = config.custom_prompt,
        )
    else
        throw(Mem0Error("Unsupported graph store provider: $(config.provider). Available: in_memory, neo4j"))
    end
end

"""
    register_llm_provider!(name, constructor)

Register a custom LLM provider.
"""
function register_llm_provider!(name::String, constructor)
    LLM_PROVIDERS[name] = constructor
end

"""
    register_embedder_provider!(name, constructor)

Register a custom embedding provider.
"""
function register_embedder_provider!(name::String, constructor)
    EMBEDDER_PROVIDERS[name] = constructor
end

"""
    register_vector_store_provider!(name, constructor)

Register a custom vector store provider.
"""
function register_vector_store_provider!(name::String, constructor)
    VECTOR_STORE_PROVIDERS[name] = constructor
end
