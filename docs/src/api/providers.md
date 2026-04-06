# [Providers API](@id providers_api)

Abstract interfaces and concrete implementations for LLM, embedding, vector store, and graph store providers.

## Abstract Types

```@docs
Mem0.AbstractLLM
Mem0.AbstractEmbedder
Mem0.AbstractVectorStore
Mem0.AbstractGraphStore
```

## LLM Providers

```@docs
Mem0.OpenAILLM
Mem0.OllamaLLM
```

## Embedding Providers

```@docs
Mem0.OpenAIEmbedding
Mem0.OllamaEmbedding
```

## Vector Store Providers

```@docs
Mem0.InMemoryVectorStore
```

## Graph Store Providers

```@docs
Mem0.InMemoryGraphStore
Mem0.Neo4jGraphStore
```

## Provider Interface Methods

```@docs
Mem0.generate_response
Mem0.embed
```

## Factory Functions

```@docs
Mem0.create_llm
Mem0.create_embedder
Mem0.create_vector_store
Mem0.create_graph_store
```

## Provider Registration

```@docs
Mem0.register_llm_provider!
Mem0.register_embedder_provider!
Mem0.register_vector_store_provider!
```
