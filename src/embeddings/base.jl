# Abstract embedding interface

"""
    AbstractEmbedder

Abstract base type for embedding model providers.
"""
abstract type AbstractEmbedder end

"""
    embed(embedder, text; memory_action=nothing)

Generate an embedding vector for the given text.

# Arguments
- `text`: The text to embed
- `memory_action`: Optional hint ("add", "search", "update") for action-aware embeddings

# Returns
A `Vector{Float64}` embedding.
"""
function embed end
