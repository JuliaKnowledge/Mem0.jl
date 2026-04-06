using Test
using Mem0
using Dates
using UUIDs

# =====================================================================
#  Test helpers — Controllable Mock LLM and Embedder
# =====================================================================

"""Mock LLM with fine-grained response control for pipeline testing."""
mutable struct ScriptedLLM <: AbstractLLM
    call_log::Vector{Vector{Any}}          # all messages seen
    script::Vector{LLMResponse}            # queued responses (FIFO)
    default_facts::Vector{String}          # fallback fact extraction
    default_update_action::String           # fallback update action
end

ScriptedLLM(; facts=["extracted fact"], action="ADD") =
    ScriptedLLM(Vector{Any}[], LLMResponse[], facts, action)
ScriptedLLM(config::Dict{String, Any}) =
    ScriptedLLM(facts=get(config, "facts", ["extracted fact"]),
                action=get(config, "action", "ADD"))

function Mem0.generate_response(llm::ScriptedLLM, messages::Vector;
                                 response_format=nothing, tools=nothing, tool_choice="auto")
    push!(llm.call_log, messages)

    # Drain script first
    !isempty(llm.script) && return popfirst!(llm.script)

    sys = get(messages[1], "content", "")

    # Fact extraction call
    if occursin("Information Organizer", sys) || occursin("extract", lowercase(sys))
        facts_json = "[" * join(["\"$(f)\"" for f in llm.default_facts], ",") * "]"
        return LLMResponse(content = "{\"facts\": $(facts_json)}")
    end

    # Memory update call
    if occursin("smart memory manager", sys)
        ops = [Dict("id" => "new", "text" => f, "event" => llm.default_update_action) for f in llm.default_facts]
        ops_json = "[" * join(["{\"id\":\"new\",\"text\":\"$(op["text"])\",\"event\":\"$(op["event"])\"}" for op in ops], ",") * "]"
        return LLMResponse(content = "{\"memory\": $(ops_json)}")
    end

    # Procedural / generic
    return LLMResponse(content = "Scripted fallback response")
end

"""Embedder that returns controllable, dimension-correct embeddings."""
struct ControlledEmbedder <: AbstractEmbedder
    dims::Int
    # Mapping of text → known embedding; unknown text gets hashed
    known::Dict{String, Vector{Float64}}
end
ControlledEmbedder(dims::Int=8) = ControlledEmbedder(dims, Dict{String, Vector{Float64}}())
ControlledEmbedder(config::Dict{String, Any}) = ControlledEmbedder(get(config, "dims", 8))

function Mem0.embed(emb::ControlledEmbedder, text::AbstractString; memory_action=nothing)::Vector{Float64}
    haskey(emb.known, text) && return emb.known[text]
    h = hash(text)
    v = [Float64((h >> (i * 8)) & 0xFF) / 255.0 for i in 0:(emb.dims - 1)]
    n = sqrt(sum(x^2 for x in v))
    return n > 0 ? v ./ n : ones(emb.dims) ./ sqrt(emb.dims)
end

"""Create a Memory with scripted mocks for predictable testing."""
function make_memory(; llm=ScriptedLLM(), embedder=ControlledEmbedder(), enable_graph=false,
                       graph_threshold=0.5, custom_fact_prompt=nothing, custom_update_prompt=nothing)
    vs = InMemoryVectorStore(collection_name="test", embedding_model_dims=embedder.dims)
    db = HistoryManager(":memory:")
    graph = enable_graph ? InMemoryGraphStore(llm=ScriptedLLM(), embedder=embedder, threshold=graph_threshold) : nothing
    Mem0.Memory(MemoryConfig(), llm, embedder, vs, db, graph, enable_graph, "test",
                custom_fact_prompt, custom_update_prompt)
end


# =====================================================================
#  UNIT TESTS — deep component coverage
# =====================================================================

@testset "Mem0.jl — Extended Tests" begin

# --- 1. Utilities — edge cases ---

@testset "Utilities — edge cases" begin
    @testset "parse_messages with symbol keys" begin
        msgs = [Dict(:role => "system", :content => "You are helpful")]
        @test occursin("System: You are helpful", parse_messages(msgs))
    end

    @testset "parse_messages with mixed key types" begin
        msgs = [
            Dict("role" => "user", "content" => "first"),
            Dict(:role => "assistant", :content => "second"),
        ]
        r = parse_messages(msgs)
        @test occursin("User: first", r)
        @test occursin("Assistant: second", r)
    end

    @testset "normalize_facts — numeric and mixed" begin
        @test normalize_facts([42, 3.14]) isa Vector{String}
        @test length(normalize_facts([42, 3.14])) == 2
        @test normalize_facts([Dict("content" => "x")]) == ["x"]
        @test normalize_facts([Dict("unknown_key" => "y")]) == String[]  # no recognized key
    end

    @testset "normalize_facts — deeply nested ignored" begin
        @test normalize_facts([Dict("fact" => " trimmed ")]) == ["trimmed"]
    end

    @testset "extract_json — nested JSON" begin
        nested = """{"memory": [{"id": "1", "text": "hi", "event": "ADD"}]}"""
        parsed = extract_json(nested)
        @test parsed !== nothing
        @test parsed["memory"][1]["event"] == "ADD"
    end

    @testset "extract_json — with surrounding text" begin
        text = "Here is the result:\n```json\n{\"facts\": [\"a\"]}\n```\nDone."
        parsed = extract_json(text)
        @test parsed !== nothing
        @test parsed["facts"] == ["a"]
    end

    @testset "extract_json — empty braces" begin
        result = extract_json("{}")
        @test result isa Dict
        @test isempty(result)
    end

    @testset "remove_code_blocks — multiple blocks" begin
        text = "```python\nprint('hi')\n```\nThen:\n```json\n{}\n```"
        cleaned = Mem0.remove_code_blocks(text)
        @test !occursin("```", cleaned)
    end

    @testset "remove_code_blocks — no blocks" begin
        @test Mem0.remove_code_blocks("plain text") == "plain text"
    end

    @testset "cosine_similarity — negative values" begin
        @test cosine_similarity([1.0, 0.0], [-1.0, 0.0]) ≈ -1.0
    end

    @testset "cosine_similarity — identical" begin
        v = [0.3, 0.4, 0.5]
        @test cosine_similarity(v, v) ≈ 1.0 atol=1e-10
    end

    @testset "cosine_similarity — orthogonal 3D" begin
        @test cosine_similarity([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) ≈ 0.0
    end

    @testset "build_filters_and_metadata — with input extras" begin
        meta, filters = build_filters_and_metadata(
            user_id="u1",
            input_metadata=Dict{String, Any}("custom_key" => "custom_val"),
            input_filters=Dict{String, Any}("tag" => "important"),
        )
        @test meta["custom_key"] == "custom_val"
        @test meta["user_id"] == "u1"
        @test filters["tag"] == "important"
        @test filters["user_id"] == "u1"
    end

    @testset "build_filters_and_metadata — agent_id only" begin
        meta, filters = build_filters_and_metadata(agent_id="bot1")
        @test meta["agent_id"] == "bot1"
        @test !haskey(meta, "user_id")
    end

    @testset "build_filters_and_metadata — run_id only" begin
        meta, filters = build_filters_and_metadata(run_id="run-42")
        @test meta["run_id"] == "run-42"
    end

    @testset "format_entities" begin
        ents = [
            Dict("source" => "alice", "relationship" => "works_at", "destination" => "microsoft"),
            Dict("source" => "bob", "relationship" => "lives_in", "destination" => "seattle"),
        ]
        formatted = Mem0.format_entities(ents)
        @test occursin("alice -- works_at -- microsoft", formatted)
        @test occursin("bob -- lives_in -- seattle", formatted)
    end

    @testset "format_entities — symbol keys" begin
        ents = [Dict(:source => "a", :relationship => "r", :destination => "b")]
        @test occursin("a -- r -- b", Mem0.format_entities(ents))
    end

    @testset "normalize_entity — special chars" begin
        @test normalize_entity("") == ""
        @test normalize_entity("UPPER") == "upper"
        @test normalize_entity("  multi  space  ") == "multi__space"
    end

    @testset "memory_hash — deterministic" begin
        @test memory_hash("test") == memory_hash("test")
        @test memory_hash("a") != memory_hash("b")
    end

    @testset "now_iso — format" begin
        ts = now_iso()
        @test length(ts) >= 19
        @test ts[5] == '-' && ts[8] == '-'
        @test occursin("T", ts)
    end
end

# --- 2. HistoryManager — deeper ---

@testset "HistoryManager — deep" begin
    @testset "actor_id and role fields" begin
        mgr = HistoryManager(":memory:")
        add_history!(mgr, "m1", nothing, "data", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00",
                     actor_id="actor_alice", role="user")
        h = get_history(mgr, "m1")
        @test h[1]["actor_id"] == "actor_alice"
        @test h[1]["role"] == "user"
    end

    @testset "NULL old_memory preserved" begin
        mgr = HistoryManager(":memory:")
        add_history!(mgr, "m2", nothing, "new data", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        h = get_history(mgr, "m2")
        @test h[1]["old_memory"] === missing || h[1]["old_memory"] === nothing
    end

    @testset "Multiple memory_ids isolation" begin
        mgr = HistoryManager(":memory:")
        add_history!(mgr, "alpha", nothing, "alpha data", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        add_history!(mgr, "beta", nothing, "beta data", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        @test length(get_history(mgr, "alpha")) == 1
        @test length(get_history(mgr, "beta")) == 1
        @test get_history(mgr, "alpha")[1]["new_memory"] == "alpha data"
    end

    @testset "Chronological ordering" begin
        mgr = HistoryManager(":memory:")
        add_history!(mgr, "ord", nothing, "v1", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        add_history!(mgr, "ord", "v1", "v2", "UPDATE";
                     created_at="2024-01-02T00:00:00", updated_at="2024-01-02T00:00:00")
        add_history!(mgr, "ord", "v2", "v3", "UPDATE";
                     created_at="2024-01-03T00:00:00", updated_at="2024-01-03T00:00:00")
        h = get_history(mgr, "ord")
        @test h[1]["new_memory"] == "v1"
        @test h[2]["new_memory"] == "v2"
        @test h[3]["new_memory"] == "v3"
    end

    @testset "Reset clears all memory IDs" begin
        mgr = HistoryManager(":memory:")
        add_history!(mgr, "x", nothing, "d", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        add_history!(mgr, "y", nothing, "d", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        reset_history!(mgr)
        @test isempty(get_history(mgr, "x"))
        @test isempty(get_history(mgr, "y"))
    end
end

# --- 3. InMemoryVectorStore — deeper ---

@testset "InMemoryVectorStore — deep" begin
    @testset "Batch insert" begin
        store = InMemoryVectorStore(collection_name="batch", embedding_model_dims=3)
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        payloads = [Dict{String, Any}("n" => "a"), Dict{String, Any}("n" => "b"), Dict{String, Any}("n" => "c")]
        ids = ["a", "b", "c"]
        Base.insert!(store, vecs, payloads, ids)
        @test length(list_records(store)) == 3
        @test Base.get(store, "b").payload["n"] == "b"
    end

    @testset "Update vector only" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        Base.insert!(store, [1.0, 0.0], Dict{String, Any}("k" => "v"), "x")
        Mem0.update!(store, "x"; vector=[0.0, 1.0])
        r = Base.get(store, "x")
        @test r.vector ≈ [0.0, 1.0]
        @test r.payload["k"] == "v"  # unchanged
    end

    @testset "Update nonexistent record — no-op" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        Mem0.update!(store, "ghost"; payload=Dict{String, Any}("k" => "v"))
        @test Base.get(store, "ghost") === nothing
    end

    @testset "Search on empty store" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        results = Mem0.search(store, [1.0, 0.0], 10)
        @test isempty(results)
    end

    @testset "Search with zero query vector" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        Base.insert!(store, [1.0, 0.0], Dict{String, Any}(), "a")
        results = Mem0.search(store, [0.0, 0.0], 10)
        @test isempty(results)
    end

    @testset "Search limit respected" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        for i in 1:10
            Base.insert!(store, [Float64(i), 0.0], Dict{String, Any}("i" => i), "id$i")
        end
        results = Mem0.search(store, [1.0, 0.0], 3)
        @test length(results) == 3
    end

    @testset "create_collection! updates metadata" begin
        store = InMemoryVectorStore(collection_name="old", embedding_model_dims=2)
        Mem0.create_collection!(store, "new_name", 128)
        @test store.collection_name == "new_name"
        @test store.embedding_dims == 128
    end

    @testset "Overwrite existing ID" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        Base.insert!(store, [1.0, 0.0], Dict{String, Any}("v" => 1), "same")
        Base.insert!(store, [0.0, 1.0], Dict{String, Any}("v" => 2), "same")
        @test length(list_records(store)) == 1
        @test Base.get(store, "same").payload["v"] == 2
    end

    @testset "Filter miss returns empty" begin
        store = InMemoryVectorStore(collection_name="t", embedding_model_dims=2)
        Base.insert!(store, [1.0, 0.0], Dict{String, Any}("user_id" => "alice"), "a")
        results = Mem0.search(store, [1.0, 0.0], 10; filters=Dict{String,Any}("user_id" => "bob"))
        @test isempty(results)
    end
end

# --- 4. LLM Provider config parsing ---

@testset "LLM Provider constructors" begin
    @testset "OpenAILLM from Dict" begin
        llm = OpenAILLM(Dict{String, Any}(
            "model" => "gpt-4o",
            "temperature" => 0.5,
            "max_tokens" => 1000,
            "top_p" => 0.9,
            "base_url" => "https://custom.api/v1",
        ))
        @test llm.model == "gpt-4o"
        @test llm.temperature == 0.5
        @test llm.max_tokens == 1000
        @test llm.top_p == 0.9
        @test llm.base_url == "https://custom.api/v1"
    end

    @testset "OpenAILLM defaults" begin
        llm = OpenAILLM()
        @test llm.model == "gpt-4.1-nano-2025-04-14"
        @test llm.temperature == 0.1
        @test llm.max_tokens == 2000
    end

    @testset "OllamaLLM from Dict" begin
        llm = OllamaLLM(Dict{String, Any}("model" => "qwen2.5", "temperature" => 0.0))
        @test llm.model == "qwen2.5"
        @test llm.temperature == 0.0
    end

    @testset "OllamaLLM defaults" begin
        llm = OllamaLLM()
        @test llm.model == "llama3.1"
        @test llm.base_url == "http://localhost:11434"
    end
end

# --- 5. Embedding Provider config parsing ---

@testset "Embedding Provider constructors" begin
    @testset "OpenAIEmbedding from Dict" begin
        emb = OpenAIEmbedding(Dict{String, Any}(
            "model" => "text-embedding-3-large",
            "embedding_dims" => 3072,
        ))
        @test emb.model == "text-embedding-3-large"
        @test emb.embedding_dims == 3072
    end

    @testset "OpenAIEmbedding defaults" begin
        emb = OpenAIEmbedding()
        @test emb.model == "text-embedding-3-small"
        @test emb.embedding_dims == 1536
    end

    @testset "OllamaEmbedding from Dict" begin
        emb = OllamaEmbedding(Dict{String, Any}("model" => "mxbai-embed-large"))
        @test emb.model == "mxbai-embed-large"
    end

    @testset "OllamaEmbedding defaults" begin
        emb = OllamaEmbedding()
        @test emb.model == "nomic-embed-text"
        @test emb.embedding_dims == 768
    end
end

# --- 6. Prompts — deeper ---

@testset "Prompts — detailed" begin
    @testset "User extraction prompt contains today's date" begin
        p = Mem0.user_memory_extraction_prompt()
        today = Dates.format(Dates.now(), "yyyy-mm-dd")
        @test occursin(today, p)
    end

    @testset "Agent extraction prompt contains today's date" begin
        p = Mem0.agent_memory_extraction_prompt()
        today = Dates.format(Dates.now(), "yyyy-mm-dd")
        @test occursin(today, p)
    end

    @testset "get_update_memory_messages with custom prompt" begin
        msgs = Mem0.get_update_memory_messages("old", "new"; custom_prompt="CUSTOM INSTRUCTIONS")
        @test msgs[1]["content"] == "CUSTOM INSTRUCTIONS"
    end

    @testset "get_update_memory_messages default prompt" begin
        msgs = Mem0.get_update_memory_messages("old", "new")
        @test occursin("smart memory manager", msgs[1]["content"])
        @test occursin("old", msgs[2]["content"])
        @test occursin("new", msgs[2]["content"])
    end

    @testset "MEMORY_ANSWER_PROMPT exists" begin
        @test occursin("expert at answering", Mem0.MEMORY_ANSWER_PROMPT)
    end
end

# --- 7. InMemoryGraphStore — deeper ---

@testset "InMemoryGraphStore — deep" begin
    @testset "Node deduplication via similarity" begin
        emb = ControlledEmbedder(4)
        # Pre-set known embeddings so "alice" and "alice" produce same vector
        emb.known["alice"] = [1.0, 0.0, 0.0, 0.0]
        emb.known["bob"] = [0.0, 1.0, 0.0, 0.0]

        graph = InMemoryGraphStore(llm=ScriptedLLM(), embedder=emb, threshold=0.9)
        filters = Dict{String, Any}("user_id" => "test")

        # Create first alice node
        id1 = Mem0._get_or_create_node!(graph, "alice", "person", filters)
        # Should find existing node (same embedding)
        id2 = Mem0._get_or_create_node!(graph, "alice", "person", filters)
        @test id1 == id2  # deduplicated

        # Bob should be a new node
        id3 = Mem0._get_or_create_node!(graph, "bob", "person", filters)
        @test id3 != id1
        @test length(graph.nodes) == 2
    end

    @testset "Multi-tenant graph isolation" begin
        emb = ControlledEmbedder(4)
        graph = InMemoryGraphStore(llm=ScriptedLLM(), embedder=emb, threshold=0.3)

        user_a = Dict{String, Any}("user_id" => "userA")
        user_b = Dict{String, Any}("user_id" => "userB")

        # Manually add edges for different users
        lock(graph.lock) do
            n1 = "n1"; n2 = "n2"; n3 = "n3"
            graph.nodes[n1] = Dict{String,Any}("name" => "x", "type" => "t",
                "embedding" => Mem0.embed(emb, "x"), "metadata" => copy(user_a))
            graph.nodes[n2] = Dict{String,Any}("name" => "y", "type" => "t",
                "embedding" => Mem0.embed(emb, "y"), "metadata" => copy(user_a))
            graph.nodes[n3] = Dict{String,Any}("name" => "z", "type" => "t",
                "embedding" => Mem0.embed(emb, "z"), "metadata" => copy(user_b))
            graph.edges["e1"] = Dict{String,Any}("source" => n1, "destination" => n2,
                "relationship" => "rel", "metadata" => copy(user_a), "valid" => true)
            graph.edges["e2"] = Dict{String,Any}("source" => n3, "destination" => n3,
                "relationship" => "self", "metadata" => copy(user_b), "valid" => true)
        end

        @test length(get_all_graph(graph, user_a)) == 1
        @test length(get_all_graph(graph, user_b)) == 1
        @test get_all_graph(graph, user_a)[1]["relationship"] == "rel"
    end

    @testset "Soft-delete preserves edges in storage" begin
        emb = ControlledEmbedder()
        graph = InMemoryGraphStore(llm=ScriptedLLM(), embedder=emb, threshold=0.3)
        filters = Dict{String, Any}("user_id" => "u")

        lock(graph.lock) do
            graph.nodes["a"] = Dict{String,Any}("name" => "a", "type" => "t",
                "embedding" => Mem0.embed(emb, "a"), "metadata" => copy(filters))
            graph.nodes["b"] = Dict{String,Any}("name" => "b", "type" => "t",
                "embedding" => Mem0.embed(emb, "b"), "metadata" => copy(filters))
            graph.edges["e"] = Dict{String,Any}("source" => "a", "destination" => "b",
                "relationship" => "r", "metadata" => copy(filters), "valid" => true)
        end

        delete_all_graph!(graph, filters)
        # Edge still exists in storage, just marked invalid
        @test length(graph.edges) == 1
        @test graph.edges["e"]["valid"] == false
        @test haskey(graph.edges["e"], "invalidated_at")
        # But hidden from query
        @test isempty(get_all_graph(graph, filters))
    end

    @testset "search_graph — limit" begin
        emb = ControlledEmbedder(4)
        emb.known["query"] = [1.0, 0.0, 0.0, 0.0]
        graph = InMemoryGraphStore(llm=ScriptedLLM(), embedder=emb, threshold=0.0)
        filters = Dict{String, Any}("user_id" => "u")

        lock(graph.lock) do
            for i in 1:5
                nid = "n$i"
                graph.nodes[nid] = Dict{String,Any}("name" => "node$i", "type" => "t",
                    "embedding" => Mem0.embed(emb, "node$i"), "metadata" => copy(filters))
            end
            for i in 1:4
                graph.edges["e$i"] = Dict{String,Any}(
                    "source" => "n$i", "destination" => "n$(i+1)",
                    "relationship" => "r$i", "metadata" => copy(filters), "valid" => true)
            end
        end

        results = search_graph(graph, "query", filters; limit=2)
        @test length(results) <= 2
    end
end

# --- 8. Factory — deeper ---

@testset "Factory — deeper" begin
    @testset "Graph store factory — in_memory" begin
        gc = GraphStoreConfig(provider="in_memory", config=Dict{String,Any}(), threshold=0.8)
        lc = LlmConfig(provider="openai")
        ec = EmbedderConfig(provider="openai")
        gs = create_graph_store(gc, lc, ec)
        @test gs isa InMemoryGraphStore
        @test gs.threshold == 0.8
    end

    @testset "Graph store factory — with custom LLM" begin
        custom_llm = LlmConfig(provider="ollama", config=Dict{String,Any}("model" => "qwen2.5"))
        gc = GraphStoreConfig(provider="in_memory", config=Dict{String,Any}(), llm=custom_llm)
        lc = LlmConfig(provider="openai")  # main LLM, should be overridden
        ec = EmbedderConfig(provider="openai")
        gs = create_graph_store(gc, lc, ec)
        @test gs.llm isa OllamaLLM
        @test gs.llm.model == "qwen2.5"
    end

    @testset "Graph store factory — unsupported" begin
        gc = GraphStoreConfig(provider="neo4j", config=Dict{String,Any}())
        @test_throws Mem0Error create_graph_store(gc, LlmConfig(), EmbedderConfig())
    end

    @testset "Register and use custom embedder" begin
        register_embedder_provider!("controlled", ControlledEmbedder)
        emb = create_embedder(EmbedderConfig(provider="controlled", config=Dict{String,Any}("dims" => 4)))
        @test emb isa ControlledEmbedder
    end

    @testset "Register and use custom vector store" begin
        register_vector_store_provider!("in_memory_v2", InMemoryVectorStore)
        vs = create_vector_store(VectorStoreConfig(provider="in_memory_v2",
            config=Dict{String,Any}("collection_name" => "v2", "embedding_model_dims" => 64)))
        @test vs isa InMemoryVectorStore
        @test vs.collection_name == "v2"
    end
end


# =====================================================================
#  INTEGRATION TESTS — multi-component workflows
# =====================================================================

@testset "Integration — full memory lifecycle" begin
    mem = make_memory(llm=ScriptedLLM(facts=["Alice likes Julia"]))

    # Add
    r = add(mem, "Hi, I'm Alice and I love Julia programming"; user_id="alice")
    @test length(r["results"]) >= 1
    mem_id = r["results"][1]["id"]

    # Search
    sr = Mem0.search(mem, "What does Alice like?"; user_id="alice")
    @test length(sr["results"]) >= 1
    @test sr["results"][1]["memory"] == "Alice likes Julia"

    # Update
    update(mem, mem_id, "Alice now prefers Rust")
    retrieved = get_memory(mem, mem_id)
    @test retrieved["memory"] == "Alice now prefers Rust"

    # Search again — updated content
    sr2 = Mem0.search(mem, "What does Alice like?"; user_id="alice")
    @test any(r -> r["memory"] == "Alice now prefers Rust", sr2["results"])

    # History tracks full lifecycle
    h = history(mem, mem_id)
    @test length(h) == 2
    @test h[1]["event"] == "ADD"
    @test h[2]["event"] == "UPDATE"

    # Delete
    delete(mem, mem_id)
    @test get_memory(mem, mem_id) === nothing

    h2 = history(mem, mem_id)
    @test length(h2) == 3
    @test h2[3]["event"] == "DELETE"
    @test h2[3]["is_deleted"] == true
end

@testset "Integration — multi-user isolation" begin
    mem = make_memory()

    add(mem, "Alice's secret"; user_id="alice", infer=false)
    add(mem, "Bob's secret"; user_id="bob", infer=false)
    add(mem, "Shared context"; user_id="alice", agent_id="bot1", infer=false)

    # get_all with user_id="alice" returns both alice-only AND alice+bot1 records
    # because user_id filter matches on the user_id field in both
    alice_mems = get_all(mem; user_id="alice")
    bob_mems = get_all(mem; user_id="bob")

    @test length(alice_mems["results"]) == 2  # both have user_id="alice"
    @test all(r -> r["user_id"] == "alice", alice_mems["results"])

    # Bob only sees his
    @test length(bob_mems["results"]) == 1
    @test bob_mems["results"][1]["memory"] == "Bob's secret"

    # Scoped query narrows to alice+bot1
    scoped = get_all(mem; user_id="alice", agent_id="bot1")
    @test length(scoped["results"]) == 1
    @test scoped["results"][1]["memory"] == "Shared context"

    # Different agent_id → no results
    empty_scoped = get_all(mem; user_id="alice", agent_id="bot999")
    @test isempty(empty_scoped["results"])
end

@testset "Integration — graph + vector combined" begin
    mem = make_memory(enable_graph=true)

    result = add(mem, "Alice works at Microsoft"; user_id="graph_user", infer=false)
    @test haskey(result, "results")
    @test haskey(result, "relations")
    @test length(result["results"]) == 1

    # Search returns both vector results and graph relations
    sr = Mem0.search(mem, "Alice"; user_id="graph_user")
    @test haskey(sr, "results")
    @test haskey(sr, "relations")

    # get_all also has relations key
    all_r = get_all(mem; user_id="graph_user")
    @test haskey(all_r, "relations")
end

@testset "Integration — inference pipeline with multiple facts" begin
    llm = ScriptedLLM(facts=["Name is Alice", "Loves Julia", "Works at GitHub"])
    mem = make_memory(llm=llm)

    result = add(mem, "My name is Alice. I love Julia. I work at GitHub."; user_id="alice")
    @test length(result["results"]) == 3
    @test all(r -> r["event"] == "ADD", result["results"])

    # All three stored
    all_mems = get_all(mem; user_id="alice")
    @test length(all_mems["results"]) == 3

    # LLM was called twice (fact extraction + update decision)
    @test length(llm.call_log) == 2
end

@testset "Integration — inference with empty facts" begin
    llm = ScriptedLLM(facts=String[])
    mem = make_memory(llm=llm)

    result = add(mem, "Hello, how are you?"; user_id="bob")
    @test isempty(result["results"])
end

@testset "Integration — LLM returns null content" begin
    llm = ScriptedLLM()
    push!(llm.script, LLMResponse(content=nothing))  # null for fact extraction
    mem = make_memory(llm=llm)

    result = add(mem, "Should produce nothing"; user_id="carol")
    @test isempty(result["results"])
end

@testset "Integration — custom fact extraction prompt" begin
    llm = ScriptedLLM(facts=["custom extracted"])
    mem = make_memory(llm=llm, custom_fact_prompt="Extract only technical preferences")

    add(mem, "I use Vim and Julia"; user_id="dave")
    # The first LLM call should use our custom prompt
    first_call = llm.call_log[1]
    @test first_call[1]["content"] == "Extract only technical preferences"
end

@testset "Integration — custom update memory prompt" begin
    llm = ScriptedLLM(facts=["some fact"])
    mem = make_memory(llm=llm, custom_update_prompt="CUSTOM UPDATE RULES")

    add(mem, "test"; user_id="eve")
    # The second LLM call (update decision) should use custom prompt
    @test length(llm.call_log) >= 2
    @test llm.call_log[2][1]["content"] == "CUSTOM UPDATE RULES"
end

@testset "Integration — add with message list" begin
    mem = make_memory()
    msgs = [
        Dict("role" => "user", "content" => "I prefer dark themes"),
        Dict("role" => "assistant", "content" => "Noted, I'll remember that."),
    ]
    result = add(mem, msgs; user_id="frank")
    @test length(result["results"]) >= 1
end

@testset "Integration — add with string input" begin
    mem = make_memory()
    result = add(mem, "plain string message"; user_id="grace", infer=false)
    @test result["results"][1]["memory"] == "plain string message"
end

@testset "Integration — metadata preservation" begin
    mem = make_memory()
    result = add(mem, "meta test"; user_id="heidi", agent_id="bot",
                 metadata=Dict{String, Any}("custom" => "value"), infer=false)
    mem_id = result["results"][1]["id"]

    retrieved = get_memory(mem, mem_id)
    @test retrieved["metadata"]["custom"] == "value"
    @test retrieved["metadata"]["user_id"] == "heidi"
    @test retrieved["metadata"]["agent_id"] == "bot"
end

@testset "Integration — hash changes on update" begin
    mem = make_memory()
    result = add(mem, "original text"; user_id="ivan", infer=false)
    mem_id = result["results"][1]["id"]

    original_hash = get_memory(mem, mem_id)["hash"]
    update(mem, mem_id, "updated text")
    new_hash = get_memory(mem, mem_id)["hash"]

    @test original_hash != new_hash
    @test new_hash == memory_hash("updated text")
end

@testset "Integration — reset clears everything" begin
    mem = make_memory(enable_graph=true)

    add(mem, "data for reset"; user_id="judy", infer=false)
    @test length(get_all(mem; user_id="judy")["results"]) == 1

    Mem0.reset!(mem)

    @test isempty(get_all(mem; user_id="judy")["results"])
    # History also cleared
    # (we can't check specific mem_id since it was auto-generated, but new adds work)
    new_result = add(mem, "fresh after reset"; user_id="judy", infer=false)
    @test length(get_all(mem; user_id="judy")["results"]) == 1
end

@testset "Integration — procedural memory with custom prompt" begin
    llm = ScriptedLLM()
    push!(llm.script, LLMResponse(content="1. Step A\n2. Step B"))
    mem = make_memory(llm=llm)

    result = add(mem, "Do A then B"; agent_id="agent1",
                 memory_type="procedural_memory", prompt="Record steps verbatim")

    @test length(result["results"]) == 1
    # LLM received our custom prompt
    @test llm.call_log[1][1]["content"] == "Record steps verbatim"
end

@testset "Integration — delete with graph cleanup" begin
    mem = make_memory(enable_graph=true)

    # Add with graph
    result = add(mem, "Alice works at Google"; user_id="alice_g", infer=false)
    mem_id = result["results"][1]["id"]

    # Delete should not throw even with graph enabled
    delete(mem, mem_id)
    @test get_memory(mem, mem_id) === nothing
end

@testset "Integration — search with threshold" begin
    emb = ControlledEmbedder(4)
    emb.known["exact match"] = [1.0, 0.0, 0.0, 0.0]
    emb.known["query text"] = [1.0, 0.0, 0.0, 0.0]  # same as exact match

    mem = make_memory(embedder=emb)
    add(mem, "exact match"; user_id="karl", infer=false)
    add(mem, "something else"; user_id="karl", infer=false)

    # High threshold should only return the exact match
    results = Mem0.search(mem, "query text"; user_id="karl", threshold=0.99)
    @test length(results["results"]) == 1
    @test results["results"][1]["memory"] == "exact match"
end

@testset "Integration — search with limit" begin
    mem = make_memory()
    for i in 1:10
        add(mem, "Memory item $i"; user_id="lisa", infer=false)
    end

    results = Mem0.search(mem, "any query"; user_id="lisa", limit=3)
    @test length(results["results"]) == 3
end

@testset "Integration — get_all with limit" begin
    mem = make_memory()
    for i in 1:5
        add(mem, "Item $i"; user_id="mike", infer=false)
    end

    all_r = get_all(mem; user_id="mike", limit=2)
    @test length(all_r["results"]) == 2
end

@testset "Integration — update preserves created_at" begin
    mem = make_memory()
    result = add(mem, "original"; user_id="nancy", infer=false)
    mem_id = result["results"][1]["id"]

    original_created = get_memory(mem, mem_id)["created_at"]
    # Small delay to ensure timestamps differ
    update(mem, mem_id, "updated")

    retrieved = get_memory(mem, mem_id)
    # created_at should still be in metadata (payload preserved)
    @test retrieved["metadata"]["created_at"] == original_created
end

@testset "Integration — update with metadata" begin
    mem = make_memory()
    result = add(mem, "base"; user_id="otto", infer=false)
    mem_id = result["results"][1]["id"]

    update(mem, mem_id, "updated"; metadata=Dict{String, Any}("priority" => "high"))
    retrieved = get_memory(mem, mem_id)
    @test retrieved["metadata"]["priority"] == "high"
end

@testset "Integration — multiple updates history" begin
    mem = make_memory()
    result = add(mem, "v1"; user_id="pat", infer=false)
    mem_id = result["results"][1]["id"]

    for i in 2:5
        update(mem, mem_id, "v$i")
    end

    h = history(mem, mem_id)
    @test length(h) == 5  # 1 ADD + 4 UPDATEs
    @test h[1]["event"] == "ADD"
    @test all(h[i]["event"] == "UPDATE" for i in 2:5)
    @test h[5]["new_memory"] == "v5"
    @test h[5]["old_memory"] == "v4"
end

@testset "Integration — concurrent-safe (basic)" begin
    mem = make_memory()
    # Add multiple memories without race conditions
    tasks = [Threads.@spawn add(mem, "concurrent $i"; user_id="quinn", infer=false) for i in 1:10]
    foreach(wait, tasks)
    all_r = get_all(mem; user_id="quinn")
    @test length(all_r["results"]) == 10
end

end  # top-level extended testset
