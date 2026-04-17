# Mem0.jl × Ollama semantic-memory vignette
#
# Narrative: a personal assistant remembers a user's preferences across
# conversations. We use Mem0 with 100% local components — Ollama for both
# the LLM (fact extraction) and the embedding model (vector recall), with
# the in-memory vector store. After a handful of raw + inferred writes,
# we show the assistant can answer questions whose answers live in memory.
#
# What this demonstrates
#   - Wiring Mem0 to Ollama (`OllamaLLM` + `OllamaEmbedding` via config)
#   - Raw storage path (`infer=false`) — pure embedding, no LLM call
#   - Inferred storage path — LLM extracts facts from a message list
#   - Vector search scoped to `user_id`
#   - Contradiction update via `update()`
#
# Prerequisites
#   - Ollama running at $OLLAMA_HOST (default http://localhost:11434)
#   - Pulled models: qwen3:8b (or set MEM0_OLLAMA_LLM) and
#     nomic-embed-text (or set MEM0_OLLAMA_EMBED)
#
# Run:  OLLAMA_HOST=http://localhost:11434 \
#       julia --project=. examples/ollama_semantic_memory.jl

using Mem0
using Test

const LLM_MODEL  = get(ENV, "MEM0_OLLAMA_LLM",   "qwen3:8b")
const EMB_MODEL  = get(ENV, "MEM0_OLLAMA_EMBED", "nomic-embed-text:latest")
const EMB_DIMS   = parse(Int, get(ENV, "MEM0_OLLAMA_EMBED_DIMS", "768"))
const OLLAMA_URL = get(ENV, "OLLAMA_HOST", "http://localhost:11434")

cfg = MemoryConfig(
    llm = LlmConfig(
        provider = "ollama",
        config   = Dict{String, Any}(
            "model"       => LLM_MODEL,
            "base_url"    => OLLAMA_URL,
            "temperature" => 0.0,
            "max_tokens"  => 512,
        ),
    ),
    embedder = EmbedderConfig(
        provider = "ollama",
        config   = Dict{String, Any}(
            "model"          => EMB_MODEL,
            "base_url"       => OLLAMA_URL,
            "embedding_dims" => EMB_DIMS,
        ),
    ),
    vector_store = VectorStoreConfig(
        provider = "in_memory",
        config   = Dict{String, Any}(
            "collection_name"      => "mem0_vignette",
            "embedding_model_dims" => EMB_DIMS,
        ),
    ),
)
mem = Mem0.Memory(config = cfg)
@info "Memory constructed" mem

# ── 1. Raw add: just embed-and-store, no LLM in the path ────────────────
for fact in [
    "Alice's favourite colour is blue.",
    "Alice has a cat named Whiskers.",
    "Alice works on the parser team.",
]
    add(mem, fact; user_id = "alice", infer = false)
end

all_alice = get_all(mem; user_id = "alice")
@test length(all_alice["results"]) == 3

# ── 2. Semantic recall via the embedder ──────────────────────────────────
hits = Mem0.search(mem, "what colour does Alice like?";
                    user_id = "alice", limit = 3)
top = [h["memory"] for h in hits["results"]]
println("\nSemantic recall:")
foreach(m -> println("  • ", m), top)
@test any(occursin("blue", lowercase(m)) for m in top)

# ── 3. Inferred add: let the LLM extract facts from a dialog ────────────
msgs = [
    Dict("role" => "user",
         "content" => "Hey, I'm Bob. I moved to Seattle last month and I drink green tea every morning."),
    Dict("role" => "assistant",
         "content" => "Welcome, Bob! Noted about Seattle and the green tea."),
]
r_infer = add(mem, msgs; user_id = "bob")
@info "Inferred writes" added=length(r_infer["results"])
@test !isempty(r_infer["results"])

bob_hits = Mem0.search(mem, "where does Bob live?"; user_id = "bob", limit = 3)
bob_top  = [h["memory"] for h in bob_hits["results"]]
println("\nBob's memory:")
foreach(m -> println("  • ", m), bob_top)
@test any(occursin("seattle", lowercase(m)) for m in bob_top)

# ── 4. Contradiction update: Alice's cat name changes ───────────────────
# Find the existing cat memory and update it in place.
cat_hits = Mem0.search(mem, "Alice's cat"; user_id = "alice", limit = 5)
cat_ids  = [h["id"] for h in cat_hits["results"] if occursin("cat", lowercase(h["memory"]))]
@test !isempty(cat_ids)
update(mem, first(cat_ids), "Alice's cat is named Mittens, not Whiskers.")

refreshed = Mem0.search(mem, "Alice's cat"; user_id = "alice", limit = 3)
@test any(occursin("mittens", lowercase(h["memory"])) for h in refreshed["results"])

# ── 5. Scope isolation: Bob's writes don't leak into Alice ──────────────
alice_for_seattle = Mem0.search(mem, "where does the user live?";
                                 user_id = "alice", limit = 3)
for h in alice_for_seattle["results"]
    @test !occursin("seattle", lowercase(h["memory"]))
end

println("\nPASS — Mem0 × Ollama semantic-memory vignette")
