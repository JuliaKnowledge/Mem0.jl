# [Graph Memory](@id graph_memory)

Graph memory adds structured knowledge representation to Mem0.jl's vector-based memory system. While vector search finds semantically similar memories, graph memory captures explicit **entity-relationship** triples (e.g., "Alice → works_at → Acme Corp") for precise, structured retrieval.

## Why Graph Memory?

Vector search excels at finding "related" memories, but it can struggle with:

- **Precise relationships**: "Who does Alice work for?" requires understanding a specific relationship, not just semantic similarity.
- **Multi-hop reasoning**: "What company does the person who likes Julia work for?" requires traversing relationships.
- **Structured queries**: Finding all entities of a certain type or all relationships of a certain kind.

Graph memory complements vector search by maintaining a knowledge graph of entities and their relationships, extracted automatically by the LLM.

## How It Works

When graph memory is enabled, every call to [`add`](@ref) performs two parallel operations:

1. **Vector pipeline** — extracts facts, embeds them, stores in vector store (as usual)
2. **Graph pipeline** — extracts entities and relationships, stores them as nodes and edges

```
"Alice works at Acme Corp and loves Julia"
    │
    ├─ Vector: "Works at Acme Corp" (embedded and stored)
    │          "Loves Julia programming language" (embedded and stored)
    │
    └─ Graph:  alice ──works_at──► acme_corp
               alice ──loves────► julia
```

Similarly, [`search`](@ref) queries both the vector store and graph store, returning results in `"results"` and `"relations"` keys respectively.

## Entity and Relationship Extraction

The graph store uses LLM-based extraction with tool calling:

1. **Entity extraction** — identifies named entities and their types (person, organization, technology, etc.)
2. **Relationship extraction** — identifies directed relationships between entities (works_at, likes, lives_in, etc.)

Entity names are normalized to lowercase with underscores (e.g., "Acme Corp" → `acme_corp`). Similar entities are deduplicated using embedding cosine similarity above the configured `threshold`.

## InMemoryGraphStore Setup

The simplest way to enable graph memory:

```julia
using Mem0

config = MemoryConfig(
    llm = LlmConfig(
        provider = "ollama",
        config = Dict("model" => "qwen2.5"),
    ),
    embedder = EmbedderConfig(
        provider = "ollama",
        config = Dict("model" => "nomic-embed-text", "embedding_dims" => 768),
    ),
    vector_store = VectorStoreConfig(
        provider = "in_memory",
        config = Dict("collection_name" => "test", "embedding_model_dims" => 768),
    ),
    graph_store = GraphStoreConfig(
        provider = "in_memory",
        config = Dict{String, Any}(),  # non-nothing value enables graph memory
        threshold = 0.7,               # entity matching similarity threshold
    ),
)

mem = Memory(config=config)
```

!!! note "Enabling graph memory"
    Graph memory is **disabled** when `graph_store.config` is `nothing` (the default). Set it to any `Dict` — even an empty one — to enable it.

## Neo4jGraphStore Setup

For production use, [`Neo4jGraphStore`](@ref) persists the knowledge graph in Neo4j:

### Prerequisites

1. Install and run [Neo4j](https://neo4j.com/download/)
2. Ensure the HTTP API is accessible (default: port 7474)

### Configuration

```julia
config = MemoryConfig(
    llm = LlmConfig(provider="ollama", config=Dict("model" => "qwen2.5")),
    embedder = EmbedderConfig(provider="ollama", config=Dict("model" => "nomic-embed-text", "embedding_dims" => 768)),
    vector_store = VectorStoreConfig(provider="in_memory", config=Dict("collection_name" => "test", "embedding_model_dims" => 768)),
    graph_store = GraphStoreConfig(
        provider = "neo4j",
        config = Dict{String, Any}(
            "url" => "http://localhost:7474",
            "username" => "neo4j",
            "password" => "your-password",
            "database" => "neo4j",
        ),
        threshold = 0.7,
    ),
)

mem = Memory(config=config)
```

### URL Formats

The `url` config key accepts multiple formats:
- `http://localhost:7474` — direct HTTP API URL
- `https://my-neo4j.example.com:7473` — HTTPS
- `bolt://localhost:7687` — host is extracted, HTTP port 7474 is used
- `neo4j://localhost:7687` — same as bolt

### Authentication

Neo4j requires `username` and `password` in the config. These are sent as HTTP Basic Auth to the Transactional Cypher API.

### Using a Separate LLM for Graph Extraction

You can use a different (potentially more capable) LLM for graph entity/relationship extraction:

```julia
graph_store = GraphStoreConfig(
    provider = "neo4j",
    config = Dict{String, Any}(
        "url" => "http://localhost:7474",
        "username" => "neo4j",
        "password" => "password",
    ),
    llm = LlmConfig(
        provider = "openai",
        config = Dict("model" => "gpt-4.1-mini-2025-04-14"),
    ),
    threshold = 0.7,
)
```

If `graph_store.llm` is `nothing`, the main `MemoryConfig.llm` is used instead.

## Combined Vector + Graph Search

When graph memory is enabled, [`search`](@ref) automatically queries both stores:

```julia
results = search(mem, "What does Alice do?"; user_id="alice")

# Vector results (semantic similarity)
for r in results["results"]
    println("Memory: ", r["memory"], " (score: ", r["score"], ")")
end

# Graph results (entity relationships)
for rel in get(results, "relations", [])
    println(rel["source"], " --", rel["relationship"], "--> ", rel["destination"])
end
```

Similarly, [`get_all`](@ref) returns both `"results"` (vector) and `"relations"` (graph) when graph memory is enabled.

## Graph-Specific Operations

You can also interact with the graph store directly through these functions:

### Adding to the Graph

[`add_to_graph!`](@ref) extracts entities and relationships from text and stores them:

```julia
add_to_graph!(mem.graph, "Alice works at Acme Corp", Dict("user_id" => "alice"))
```

### Searching the Graph

[`search_graph`](@ref) finds entities similar to the query and returns connected relationships:

```julia
relations = search_graph(mem.graph, "Alice", Dict("user_id" => "alice"); limit=10)
for rel in relations
    println(rel["source"], " → ", rel["relationship"], " → ", rel["destination"])
end
```

### Retrieving All Relationships

[`get_all_graph`](@ref) returns all valid relationships matching the filters:

```julia
all_rels = get_all_graph(mem.graph, Dict("user_id" => "alice"); limit=100)
```

### Deleting from the Graph

[`delete_from_graph!`](@ref) soft-deletes relationships matching the data and filters:

```julia
delete_from_graph!(mem.graph, "Alice works at Acme Corp", Dict("user_id" => "alice"))
```

[`delete_all_graph!`](@ref) soft-deletes (in-memory) or hard-deletes (Neo4j) all matching relationships:

```julia
delete_all_graph!(mem.graph, Dict("user_id" => "alice"))
```

## Custom Extraction Prompts

You can customize how relationships are extracted by setting `custom_prompt` on the graph store config:

```julia
graph_store = GraphStoreConfig(
    provider = "in_memory",
    config = Dict{String, Any}(),
    custom_prompt = """
    Focus on professional and organizational relationships.
    Extract relationships like: works_at, manages, reports_to, collaborates_with.
    Ignore personal preferences and casual mentions.
    """,
)
```

## Entity Similarity Threshold

The `threshold` parameter (default `0.7`) controls how aggressively entities are deduplicated. When a new entity is extracted:

1. Its name is embedded
2. Existing nodes matching the filters are compared by cosine similarity
3. If any existing node scores ≥ `threshold`, it is reused instead of creating a new node

Lower thresholds merge more aggressively (risk of false merges); higher thresholds create more separate nodes (risk of duplicates).
