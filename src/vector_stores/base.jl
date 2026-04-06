# Abstract vector store interface

"""
    AbstractVectorStore

Abstract base type for vector store backends.
"""
abstract type AbstractVectorStore end

"""
    VectorRecord

A single record in the vector store.
"""
Base.@kwdef mutable struct VectorRecord
    id::String = string(uuid4())
    vector::Vector{Float64} = Float64[]
    payload::Dict{String, Any} = Dict{String, Any}()
end

# Required interface methods — implementations should define:
# create_collection!(store, name, vector_size; distance=:cosine)
# Base.insert!(store, vectors, payloads, ids)
# search(store, query_vector, limit; filters, threshold)
# Base.delete!(store, vector_id)
# update!(store, vector_id; vector, payload)
# Base.get(store, vector_id)
# list_records(store; filters, limit)
# reset!(store)
