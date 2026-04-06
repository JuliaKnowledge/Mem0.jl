# Custom exception types for Mem0.jl

"""
    Mem0Error <: Exception

Base exception for all Mem0 errors.
"""
struct Mem0Error <: Exception
    message::String
end
Base.showerror(io::IO, e::Mem0Error) = print(io, "Mem0Error: ", e.message)

"""
    Mem0ValidationError <: Exception

Raised when input validation fails (missing filters, invalid config, etc.).
"""
struct Mem0ValidationError <: Exception
    message::String
    code::Union{Nothing, String}
    suggestion::Union{Nothing, String}
end
Mem0ValidationError(msg::String) = Mem0ValidationError(msg, nothing, nothing)

function Base.showerror(io::IO, e::Mem0ValidationError)
    print(io, "Mem0ValidationError: ", e.message)
    e.code !== nothing && print(io, " [code: ", e.code, "]")
    e.suggestion !== nothing && print(io, "\n  Suggestion: ", e.suggestion)
end

"""
    Mem0ProviderError <: Exception

Raised when an LLM, embedder, or vector store provider fails.
"""
struct Mem0ProviderError <: Exception
    provider::String
    message::String
    cause::Union{Nothing, Exception}
end
Mem0ProviderError(provider::String, msg::String) = Mem0ProviderError(provider, msg, nothing)

function Base.showerror(io::IO, e::Mem0ProviderError)
    print(io, "Mem0ProviderError [", e.provider, "]: ", e.message)
    e.cause !== nothing && print(io, "\n  Caused by: ", e.cause)
end
