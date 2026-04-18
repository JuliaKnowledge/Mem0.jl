# [Types](@id types_api)

Core data types, configuration structs, enums, and exception types used throughout Mem0.jl.

## Configuration Types

```@docs
Mem0.MemoryConfig
Mem0.LlmConfig
Mem0.EmbedderConfig
Mem0.VectorStoreConfig
Mem0.GraphStoreConfig
Mem0.RerankerConfig
```

## Data Types

```@docs
Mem0.MemoryItem
Mem0.MemoryResult
Mem0.ChatMessage
```

## Memory Type Enum

```@docs
Mem0.MemoryType
```

The `MemoryType` enum has three values:

- `SEMANTIC_MEMORY` — facts and concepts
- `EPISODIC_MEMORY` — events and experiences
- `PROCEDURAL_MEMORY` — skills and procedures

## Internal Data Types

```@docs
Mem0.VectorRecord
Mem0.LLMResponse
```

## History Manager

```@docs
Mem0.HistoryManager
```

## Exceptions

```@docs
Mem0.Mem0Error
Mem0.Mem0ValidationError
Mem0.Mem0ProviderError
```
