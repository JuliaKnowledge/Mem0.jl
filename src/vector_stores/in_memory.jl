# In-memory vector store implementation

using LinearAlgebra: norm, dot

"""
    InMemoryVectorStore <: AbstractVectorStore

A simple in-memory vector store using brute-force cosine similarity search.
Suitable for development, testing, and small-scale deployments.
"""
mutable struct InMemoryVectorStore <: AbstractVectorStore
    collection_name::String
    embedding_dims::Int
    records::Dict{String, VectorRecord}

    function InMemoryVectorStore(; collection_name::String="mem0",
                                   embedding_model_dims::Int=1536)
        new(collection_name, embedding_model_dims, Dict{String, VectorRecord}())
    end
end

function InMemoryVectorStore(config::Dict{String, Any})
    InMemoryVectorStore(
        collection_name = get(config, "collection_name", "mem0"),
        embedding_model_dims = get(config, "embedding_model_dims", 1536),
    )
end

function create_collection!(store::InMemoryVectorStore, name::String, vector_size::Int;
                             distance::Symbol=:cosine)
    store.collection_name = name
    store.embedding_dims = vector_size
end

function Base.insert!(store::InMemoryVectorStore, vectors::Vector{<:AbstractVector},
                      payloads::Vector{<:AbstractDict},
                      ids::Vector{String})
    for (vec, payload, id) in zip(vectors, payloads, ids)
        store.records[id] = VectorRecord(
            id = id,
            vector = Float64.(vec),
            payload = Dict{String, Any}(string(k) => v for (k, v) in payload),
        )
    end
end

# Single-record insert convenience
function Base.insert!(store::InMemoryVectorStore, vector::AbstractVector,
                      payload::AbstractDict, id::String)
    Base.insert!(store, [vector], [payload], [id])
end

function _matches_filters(payload::Dict{String, Any}, filters::Dict{String, Any})::Bool
    for (key, value) in filters
        haskey(payload, key) || return false
        payload[key] != value && return false
    end
    return true
end

function search(store::InMemoryVectorStore, query_vector::AbstractVector, limit::Int;
                filters::Union{Nothing, Dict}=nothing,
                threshold::Union{Nothing, Float64}=nothing)
    results = Tuple{String, Float64, Dict{String, Any}}[]
    qvec = Float64.(query_vector)
    qnorm = norm(qvec)
    qnorm == 0.0 && return results

    for (id, record) in store.records
        # Apply filters
        if filters !== nothing && !_matches_filters(record.payload, filters)
            continue
        end

        rnorm = norm(record.vector)
        rnorm == 0.0 && continue
        score = dot(qvec, record.vector) / (qnorm * rnorm)

        if threshold !== nothing && score < threshold
            continue
        end

        push!(results, (id, score, record.payload))
    end

    # Sort by score descending
    sort!(results, by=x -> -x[2])
    return results[1:min(limit, length(results))]
end

function Base.delete!(store::InMemoryVectorStore, vector_id::String)
    delete!(store.records, vector_id)
end

function update!(store::InMemoryVectorStore, vector_id::String;
                 vector::Union{Nothing, AbstractVector}=nothing,
                 payload::Union{Nothing, AbstractDict}=nothing)
    haskey(store.records, vector_id) || return

    record = store.records[vector_id]
    if vector !== nothing
        record.vector = Float64.(vector)
    end
    if payload !== nothing
        merge!(record.payload, Dict{String, Any}(string(k) => v for (k, v) in payload))
    end
end

function Base.get(store::InMemoryVectorStore, vector_id::String)::Union{Nothing, VectorRecord}
    return get(store.records, vector_id, nothing)
end

function list_records(store::InMemoryVectorStore;
                      filters::Union{Nothing, Dict}=nothing,
                      limit::Union{Nothing, Int}=nothing)
    results = VectorRecord[]
    for (_, record) in store.records
        if filters !== nothing && !_matches_filters(record.payload, filters)
            continue
        end
        push!(results, record)
    end

    if limit !== nothing
        return results[1:min(limit, length(results))]
    end
    return results
end

function reset!(store::InMemoryVectorStore)
    empty!(store.records)
end
