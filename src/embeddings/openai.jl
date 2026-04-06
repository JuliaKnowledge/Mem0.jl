# OpenAI embedding provider

"""
    OpenAIEmbedding <: AbstractEmbedder

OpenAI embedding provider using the Embeddings API.
"""
Base.@kwdef mutable struct OpenAIEmbedding <: AbstractEmbedder
    model::String = "text-embedding-3-small"
    api_key::String = get(ENV, "OPENAI_API_KEY", "")
    base_url::String = get(ENV, "OPENAI_API_BASE", "https://api.openai.com/v1")
    embedding_dims::Int = 1536
end

function OpenAIEmbedding(config::Dict{String, Any})
    OpenAIEmbedding(
        model = get(config, "model", "text-embedding-3-small"),
        api_key = get(config, "api_key", get(ENV, "OPENAI_API_KEY", "")),
        base_url = get(config, "base_url", get(ENV, "OPENAI_API_BASE", "https://api.openai.com/v1")),
        embedding_dims = get(config, "embedding_dims", 1536),
    )
end

function embed(emb::OpenAIEmbedding, text::AbstractString; memory_action=nothing)::Vector{Float64}
    url = "$(emb.base_url)/embeddings"
    clean_text = replace(text, "\n" => " ")

    body = Dict{String, Any}(
        "model" => emb.model,
        "input" => clean_text,
    )
    # Only pass dimensions for models that support it (text-embedding-3-*)
    if startswith(emb.model, "text-embedding-3")
        body["dimensions"] = emb.embedding_dims
    end

    resp = HTTP.post(url,
        Dict(
            "Authorization" => "Bearer $(emb.api_key)",
            "Content-Type" => "application/json",
        ),
        JSON3.write(body);
        status_exception=false,
    )

    if resp.status != 200
        throw(Mem0ProviderError("openai_embedding", "API request failed ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    return Float64.(data["data"][1]["embedding"])
end
