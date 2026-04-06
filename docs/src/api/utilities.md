# [Utilities API](@id utilities_api)

Helper functions for message parsing, JSON extraction, hashing, filtering, vector math, history management, and graph operations.

## Message Parsing

```@docs
Mem0.parse_messages
Mem0.normalize_facts
```

## JSON Utilities

```@docs
Mem0.extract_json
Mem0.remove_code_blocks
```

## Hashing and Normalization

```@docs
Mem0.memory_hash
Mem0.normalize_entity
Mem0.now_iso
```

## Vector Math

```@docs
Mem0.cosine_similarity
```

## Filter Construction

```@docs
Mem0.build_filters_and_metadata
```

## History Operations

```@docs
Mem0.add_history!
Mem0.get_history
Mem0.reset_history!
```

## Graph Operations

```@docs
Mem0.add_to_graph!
Mem0.search_graph
Mem0.delete_from_graph!
Mem0.delete_all_graph!
Mem0.get_all_graph
```

## Vector Store Operations

```@docs
Mem0.list_records
```
