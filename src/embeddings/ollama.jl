# Ollama embedding provider

"""
    OllamaEmbedding <: AbstractEmbedder

Ollama embedding provider using the local Ollama API.
"""
Base.@kwdef mutable struct OllamaEmbedding <: AbstractEmbedder
    model::String = "nomic-embed-text"
    base_url::String = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
    embedding_dims::Int = 768
end

function OllamaEmbedding(config::Dict{String, Any})
    OllamaEmbedding(
        model = get(config, "model", "nomic-embed-text"),
        base_url = get(config, "base_url", get(ENV, "OLLAMA_HOST", "http://localhost:11434")),
        embedding_dims = get(config, "embedding_dims", 768),
    )
end

function embed(emb::OllamaEmbedding, text::AbstractString; memory_action=nothing)::Vector{Float64}
    url = "$(emb.base_url)/api/embed"

    body = Dict{String, Any}(
        "model" => emb.model,
        "input" => text,
    )

    resp = HTTP.post(url,
        ["Content-Type" => "application/json"],
        JSON3.write(body);
        status_exception=false,
    )

    if resp.status != 200
        throw(Mem0ProviderError("ollama_embedding", "API request failed ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    embeddings = data["embeddings"]
    return Float64.(embeddings[1])
end
