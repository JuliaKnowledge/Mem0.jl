using Test
using Mem0
using Dates
using UUIDs

# --- Mock LLM and Embedder for testing ---

"""A mock LLM that returns predictable responses based on message content."""
mutable struct MockLLM <: AbstractLLM
    call_count::Int
    responses::Vector{LLMResponse}
    last_messages::Vector{Any}
end

MockLLM() = MockLLM(0, LLMResponse[], Any[])
MockLLM(config::Dict{String, Any}) = MockLLM()

function Mem0.generate_response(llm::MockLLM, messages::Vector;
                                 response_format=nothing,
                                 tools=nothing,
                                 tool_choice="auto")
    llm.call_count += 1
    llm.last_messages = messages

    # Return queued response if available
    if !isempty(llm.responses)
        return popfirst!(llm.responses)
    end

    # Default: extract facts from user content
    user_msg = ""
    for m in messages
        if get(m, "role", "") == "user"
            user_msg = get(m, "content", "")
        end
    end

    # Check if this is an update memory call
    sys_msg = get(messages[1], "content", "")
    if occursin("smart memory manager", sys_msg)
        return LLMResponse(
            content = """{"memory": [{"id": "new", "text": "$(replace(user_msg, "\"" => "'"))", "event": "ADD"}]}""",
        )
    end

    # Default fact extraction response
    return LLMResponse(
        content = """{"facts": ["Test fact from conversation"]}""",
    )
end

"""A mock embedder that returns deterministic embeddings."""
struct MockEmbedder <: AbstractEmbedder
    dims::Int
end
MockEmbedder() = MockEmbedder(8)

function Mem0.embed(emb::MockEmbedder, text::AbstractString; memory_action=nothing)::Vector{Float64}
    # Generate deterministic embedding based on text hash
    h = hash(text)
    rng = [Float64((h >> (i * 8)) & 0xFF) / 255.0 for i in 0:(emb.dims-1)]
    n = sqrt(sum(x^2 for x in rng))
    return n > 0 ? rng ./ n : ones(emb.dims) ./ sqrt(emb.dims)
end

"""Helper to create a Memory with mocked providers."""
function create_test_memory(; enable_graph::Bool=false)
    llm = MockLLM()
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore(collection_name="test", embedding_model_dims=8)
    db = HistoryManager(":memory:")

    graph = nothing
    if enable_graph
        graph = InMemoryGraphStore(llm=MockLLM(), embedder=embedder, threshold=0.5)
    end

    return Mem0.Memory(
        MemoryConfig(),
        llm, embedder, vector_store, db,
        graph, enable_graph, "test",
        nothing, nothing,
    )
end

# ==================== Test Suites ====================

@testset "Mem0.jl" begin

    @testset "Exceptions" begin
        @test_throws Mem0Error throw(Mem0Error("test"))
        @test_throws Mem0ValidationError throw(Mem0ValidationError("bad input"))

        e = Mem0ValidationError("missing", "CODE_01", "try this")
        buf = IOBuffer()
        showerror(buf, e)
        s = String(take!(buf))
        @test occursin("missing", s)
        @test occursin("CODE_01", s)
        @test occursin("try this", s)

        pe = Mem0ProviderError("openai", "timeout", nothing)
        buf = IOBuffer()
        showerror(buf, pe)
        @test occursin("openai", String(take!(buf)))
    end

    @testset "Types" begin
        # MemoryConfig defaults
        cfg = MemoryConfig()
        @test cfg.llm.provider == "openai"
        @test cfg.embedder.provider == "openai"
        @test cfg.vector_store.provider == "in_memory"
        @test cfg.history_db_path == ":memory:"
        @test cfg.reranker === nothing

        # LlmConfig
        lc = LlmConfig(provider="ollama", config=Dict{String, Any}("model" => "qwen2.5"))
        @test lc.provider == "ollama"
        @test lc.config["model"] == "qwen2.5"

        # EmbedderConfig
        ec = EmbedderConfig(provider="ollama")
        @test ec.provider == "ollama"

        # MemoryType enum
        @test SEMANTIC_MEMORY isa MemoryType
        @test EPISODIC_MEMORY isa MemoryType
        @test PROCEDURAL_MEMORY isa MemoryType

        # ChatMessage
        msg = ChatMessage(role="user", content="hello")
        @test msg.role == "user"
        @test msg.content == "hello"
    end

    @testset "Utilities" begin
        # parse_messages
        msgs = [
            Dict("role" => "user", "content" => "Hi there"),
            Dict("role" => "assistant", "content" => "Hello!"),
        ]
        result = parse_messages(msgs)
        @test occursin("User: Hi there", result)
        @test occursin("Assistant: Hello!", result)
        @test parse_messages("raw text") == "raw text"

        # normalize_facts
        @test normalize_facts(["fact1", "  fact2  "]) == ["fact1", "fact2"]
        @test normalize_facts([Dict("fact" => "a"), Dict("text" => "b")]) == ["a", "b"]
        @test normalize_facts([Dict("memory" => "c")]) == ["c"]
        @test normalize_facts([""]) == String[]

        # remove_code_blocks
        @test Mem0.remove_code_blocks("```json\n{\"a\":1}\n```") == "{\"a\":1}"
        @test Mem0.remove_code_blocks("<think>stuff</think>clean") == "clean"

        # extract_json
        @test extract_json("{\"key\": \"value\"}")["key"] == "value"
        @test extract_json("```json\n{\"a\": 1}\n```")["a"] == 1
        @test extract_json("no json here") === nothing

        # memory_hash
        h1 = memory_hash("hello")
        h2 = memory_hash("hello")
        h3 = memory_hash("world")
        @test h1 == h2
        @test h1 != h3
        @test length(h1) == 64  # SHA-256 hex

        # normalize_entity
        @test normalize_entity("John Smith") == "john_smith"
        @test normalize_entity("  Hello World  ") == "hello_world"

        # cosine_similarity
        @test cosine_similarity([1.0, 0.0], [1.0, 0.0]) ≈ 1.0
        @test cosine_similarity([1.0, 0.0], [0.0, 1.0]) ≈ 0.0
        @test cosine_similarity([0.0, 0.0], [1.0, 0.0]) ≈ 0.0

        # build_filters_and_metadata
        meta, filters = build_filters_and_metadata(user_id="alice")
        @test meta["user_id"] == "alice"
        @test filters["user_id"] == "alice"

        meta, filters = build_filters_and_metadata(user_id="a", agent_id="b", run_id="c")
        @test length(meta) == 3
        @test length(filters) == 3

        @test_throws Mem0ValidationError build_filters_and_metadata()

        # now_iso
        ts = now_iso()
        @test occursin("T", ts)
    end

    @testset "Prompts" begin
        sp = Mem0.user_memory_extraction_prompt()
        @test occursin("Personal Information Organizer", sp)
        @test occursin("facts", sp)

        ap = Mem0.agent_memory_extraction_prompt()
        @test occursin("Assistant Information Organizer", ap)

        @test occursin("smart memory manager", Mem0.DEFAULT_UPDATE_MEMORY_PROMPT)
        @test occursin("execution history", Mem0.PROCEDURAL_MEMORY_SYSTEM_PROMPT)

        sys, user = Mem0.get_fact_retrieval_messages("Hello, I'm Alice")
        @test occursin("Personal Information", sys)
        @test user == "Hello, I'm Alice"

        sys2, _ = Mem0.get_fact_retrieval_messages("test"; is_agent_memory=true)
        @test occursin("Assistant Information", sys2)

        update_msgs = Mem0.get_update_memory_messages("existing mem", "new facts")
        @test length(update_msgs) == 2
        @test update_msgs[1]["role"] == "system"
        @test occursin("existing mem", update_msgs[2]["content"])
    end

    @testset "Storage - HistoryManager" begin
        mgr = HistoryManager(":memory:")

        # Empty history
        h = get_history(mgr, "nonexistent")
        @test isempty(h)

        # Add and retrieve
        add_history!(mgr, "mem1", nothing, "first memory", "ADD";
                     created_at="2024-01-01T00:00:00", updated_at="2024-01-01T00:00:00")
        h = get_history(mgr, "mem1")
        @test length(h) == 1
        @test h[1]["event"] == "ADD"
        @test h[1]["new_memory"] == "first memory"
        @test h[1]["is_deleted"] == false

        # Update
        add_history!(mgr, "mem1", "first memory", "updated memory", "UPDATE";
                     created_at="2024-01-02T00:00:00", updated_at="2024-01-02T00:00:00")
        h = get_history(mgr, "mem1")
        @test length(h) == 2

        # Delete record
        add_history!(mgr, "mem1", "updated memory", nothing, "DELETE";
                     created_at="2024-01-03T00:00:00", updated_at="2024-01-03T00:00:00",
                     is_deleted=1)
        h = get_history(mgr, "mem1")
        @test length(h) == 3
        @test h[3]["is_deleted"] == true

        # Reset
        reset_history!(mgr)
        @test isempty(get_history(mgr, "mem1"))
    end

    @testset "InMemoryVectorStore" begin
        store = InMemoryVectorStore(collection_name="test", embedding_model_dims=4)

        # Insert
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        v3 = [0.5, 0.5, 0.5, 0.5]
        Base.insert!(store, v1, Dict{String, Any}("data" => "mem1", "user_id" => "alice"), "id1")
        Base.insert!(store, v2, Dict{String, Any}("data" => "mem2", "user_id" => "bob"), "id2")
        Base.insert!(store, v3, Dict{String, Any}("data" => "mem3", "user_id" => "alice"), "id3")

        # Get
        r = Base.get(store, "id1")
        @test r !== nothing
        @test r.id == "id1"
        @test r.payload["data"] == "mem1"
        @test Base.get(store, "nonexistent") === nothing

        # Search without filters
        results = Mem0.search(store, [1.0, 0.0, 0.0, 0.0], 10)
        @test length(results) == 3
        @test results[1][1] == "id1"  # Most similar
        @test results[1][2] ≈ 1.0

        # Search with filters
        results = Mem0.search(store, [1.0, 0.0, 0.0, 0.0], 10; filters=Dict{String,Any}("user_id" => "alice"))
        @test length(results) == 2
        @test all(r -> r[3]["user_id"] == "alice", results)

        # Search with threshold
        results = Mem0.search(store, [1.0, 0.0, 0.0, 0.0], 10; threshold=0.95)
        @test length(results) == 1
        @test results[1][1] == "id1"

        # Update
        Mem0.update!(store, "id1"; payload=Dict{String, Any}("data" => "updated_mem1"))
        r = Base.get(store, "id1")
        @test r.payload["data"] == "updated_mem1"

        # List
        records = list_records(store)
        @test length(records) == 3

        records = list_records(store; filters=Dict{String,Any}("user_id" => "bob"))
        @test length(records) == 1

        records = list_records(store; limit=2)
        @test length(records) == 2

        # Delete
        Base.delete!(store, "id2")
        @test Base.get(store, "id2") === nothing
        @test length(list_records(store)) == 2

        # Reset
        Mem0.reset!(store)
        @test isempty(list_records(store))
    end

    @testset "MockLLM" begin
        llm = MockLLM()
        resp = Mem0.generate_response(llm, [Dict("role" => "user", "content" => "test")])
        @test resp.content !== nothing
        @test occursin("facts", resp.content)
        @test llm.call_count == 1

        # Queued responses
        push!(llm.responses, LLMResponse(content="custom response"))
        resp = Mem0.generate_response(llm, [Dict("role" => "user", "content" => "x")])
        @test resp.content == "custom response"
    end

    @testset "MockEmbedder" begin
        emb = MockEmbedder()
        v1 = Mem0.embed(emb, "hello")
        v2 = Mem0.embed(emb, "hello")
        v3 = Mem0.embed(emb, "world")
        @test length(v1) == 8
        @test v1 ≈ v2  # Deterministic
        @test v1 != v3  # Different text → different embedding
        @test isapprox(sum(x^2 for x in v1), 1.0; atol=1e-10)  # Normalized
    end

    @testset "Memory - add with inference" begin
        mem = create_test_memory()
        result = add(mem, "Hi, my name is Alice and I love Julia programming"; user_id="alice")

        @test haskey(result, "results")
        @test length(result["results"]) >= 1
        @test result["results"][1]["event"] == "ADD"

        # Verify stored in vector store
        all_mems = get_all(mem; user_id="alice")
        @test length(all_mems["results"]) >= 1
    end

    @testset "Memory - add without inference" begin
        mem = create_test_memory()
        result = add(mem, "Raw message to store"; user_id="bob", infer=false)

        @test length(result["results"]) == 1
        @test result["results"][1]["event"] == "ADD"

        all_mems = get_all(mem; user_id="bob")
        @test length(all_mems["results"]) == 1
        @test all_mems["results"][1]["memory"] == "Raw message to store"
    end

    @testset "Memory - search" begin
        mem = create_test_memory()
        add(mem, "I prefer dark mode in my IDE"; user_id="carol", infer=false)
        add(mem, "My favorite language is Rust"; user_id="carol", infer=false)

        results = Mem0.search(mem, "What IDE preferences?"; user_id="carol")
        @test haskey(results, "results")
        @test length(results["results"]) == 2  # Both returned since mock embeddings
    end

    @testset "Memory - update" begin
        mem = create_test_memory()
        result = add(mem, "I like Python"; user_id="dave", infer=false)
        mem_id = result["results"][1]["id"]

        update(mem, mem_id, "I like Julia now")
        retrieved = get_memory(mem, mem_id)
        @test retrieved !== nothing
        @test retrieved["memory"] == "I like Julia now"

        # History should show ADD + UPDATE
        h = history(mem, mem_id)
        @test length(h) == 2
        @test h[1]["event"] == "ADD"
        @test h[2]["event"] == "UPDATE"
    end

    @testset "Memory - delete" begin
        mem = create_test_memory()
        result = add(mem, "Temporary memory"; user_id="eve", infer=false)
        mem_id = result["results"][1]["id"]

        @test get_memory(mem, mem_id) !== nothing
        delete(mem, mem_id)
        @test get_memory(mem, mem_id) === nothing

        h = history(mem, mem_id)
        @test length(h) == 2
        @test h[2]["event"] == "DELETE"
    end

    @testset "Memory - get_memory" begin
        mem = create_test_memory()
        result = add(mem, "Specific memory"; user_id="frank", infer=false)
        mem_id = result["results"][1]["id"]

        retrieved = get_memory(mem, mem_id)
        @test retrieved !== nothing
        @test retrieved["memory"] == "Specific memory"
        @test retrieved["user_id"] == "frank"
        @test retrieved["id"] == mem_id
        @test retrieved["created_at"] !== nothing
        @test retrieved["hash"] !== nothing

        @test get_memory(mem, "nonexistent") === nothing
    end

    @testset "Memory - get_all" begin
        mem = create_test_memory()
        add(mem, "Memory 1"; user_id="grace", infer=false)
        add(mem, "Memory 2"; user_id="grace", infer=false)
        add(mem, "Memory 3"; user_id="heidi", infer=false)

        grace_mems = get_all(mem; user_id="grace")
        @test length(grace_mems["results"]) == 2
        @test all(r -> r["user_id"] == "grace", grace_mems["results"])

        heidi_mems = get_all(mem; user_id="heidi")
        @test length(heidi_mems["results"]) == 1
    end

    @testset "Memory - history" begin
        mem = create_test_memory()
        result = add(mem, "Version 1"; user_id="ivan", infer=false)
        mem_id = result["results"][1]["id"]

        update(mem, mem_id, "Version 2")
        update(mem, mem_id, "Version 3")

        h = history(mem, mem_id)
        @test length(h) == 3
        @test h[1]["event"] == "ADD"
        @test h[2]["event"] == "UPDATE"
        @test h[3]["event"] == "UPDATE"
        @test h[1]["new_memory"] == "Version 1"
        @test h[2]["old_memory"] == "Version 1"
        @test h[2]["new_memory"] == "Version 2"
    end

    @testset "Memory - reset" begin
        mem = create_test_memory()
        add(mem, "To be deleted"; user_id="judy", infer=false)
        @test length(get_all(mem; user_id="judy")["results"]) == 1

        Mem0.reset!(mem)
        @test length(get_all(mem; user_id="judy")["results"]) == 0
    end

    @testset "Memory - procedural memory" begin
        mem = create_test_memory()
        result = add(mem, "Step 1: Do A. Step 2: Do B.";
                     agent_id="my_agent", memory_type="procedural_memory")

        @test length(result["results"]) == 1
        @test result["results"][1]["event"] == "ADD"
    end

    @testset "Memory - multiple session IDs" begin
        mem = create_test_memory()
        add(mem, "Scoped memory"; user_id="alice", agent_id="bot1", run_id="run1", infer=false)

        results = get_all(mem; user_id="alice", agent_id="bot1", run_id="run1")
        @test length(results["results"]) == 1

        # Different agent_id → no results (filter mismatch)
        results = get_all(mem; user_id="alice", agent_id="bot2", run_id="run1")
        @test length(results["results"]) == 0
    end

    @testset "Memory - validation errors" begin
        mem = create_test_memory()

        # Missing session IDs
        @test_throws Mem0ValidationError add(mem, "test")
        @test_throws Mem0ValidationError Mem0.search(mem, "test")
        @test_throws Mem0ValidationError get_all(mem)

        # Nonexistent memory update
        @test_throws Mem0ValidationError update(mem, "nonexistent_id", "data")

        # Nonexistent memory delete
        @test_throws Mem0ValidationError delete(mem, "nonexistent_id")
    end

    @testset "Memory - with graph store" begin
        mem = create_test_memory(enable_graph=true)

        # Queue up responses for entity/relation extraction
        graph_llm = mem.graph.llm

        result = add(mem, "Alice works at Microsoft. She lives in Seattle."; user_id="test_user")
        @test haskey(result, "results")
        @test haskey(result, "relations")
    end

    @testset "Factory - create providers" begin
        # LLM factory
        llm_config = LlmConfig(provider="openai")
        llm = create_llm(llm_config)
        @test llm isa OpenAILLM

        llm_config2 = LlmConfig(provider="ollama", config=Dict{String, Any}("model" => "qwen2.5"))
        llm2 = create_llm(llm_config2)
        @test llm2 isa OllamaLLM
        @test llm2.model == "qwen2.5"

        # Embedder factory
        emb_config = EmbedderConfig(provider="openai")
        emb = create_embedder(emb_config)
        @test emb isa OpenAIEmbedding

        emb_config2 = EmbedderConfig(provider="ollama")
        emb2 = create_embedder(emb_config2)
        @test emb2 isa OllamaEmbedding

        # Vector store factory
        vs_config = VectorStoreConfig(provider="in_memory")
        vs = create_vector_store(vs_config)
        @test vs isa InMemoryVectorStore

        # Invalid provider
        @test_throws Mem0Error create_llm(LlmConfig(provider="nonexistent"))
        @test_throws Mem0Error create_embedder(EmbedderConfig(provider="nonexistent"))
        @test_throws Mem0Error create_vector_store(VectorStoreConfig(provider="nonexistent"))
    end

    @testset "Factory - register custom providers" begin
        register_llm_provider!("mock", MockLLM)
        llm = create_llm(LlmConfig(provider="mock", config=Dict{String, Any}()))
        @test llm isa MockLLM
    end

    @testset "InMemoryGraphStore" begin
        embedder = MockEmbedder()
        llm = MockLLM()
        graph = InMemoryGraphStore(llm=llm, embedder=embedder, threshold=0.3)

        filters = Dict{String, Any}("user_id" => "test")

        # Manually add nodes and edges for testing
        lock(graph.lock) do
            nid1 = "node1"
            nid2 = "node2"
            graph.nodes[nid1] = Dict{String, Any}(
                "name" => "alice", "type" => "person",
                "embedding" => Mem0.embed(embedder, "alice"),
                "metadata" => copy(filters), "mentions" => 1,
            )
            graph.nodes[nid2] = Dict{String, Any}(
                "name" => "microsoft", "type" => "company",
                "embedding" => Mem0.embed(embedder, "microsoft"),
                "metadata" => copy(filters), "mentions" => 1,
            )
            graph.edges["edge1"] = Dict{String, Any}(
                "source" => nid1, "destination" => nid2,
                "relationship" => "works_at",
                "metadata" => copy(filters),
                "valid" => true,
                "created_at" => now_iso(),
            )
        end

        # Search
        results = search_graph(graph, "alice", filters)
        @test length(results) >= 1
        @test any(r -> r["source"] == "alice" || r["destination"] == "alice", results)

        # Get all
        all_rels = get_all_graph(graph, filters)
        @test length(all_rels) == 1
        @test all_rels[1]["relationship"] == "works_at"

        # Soft delete
        delete_all_graph!(graph, filters)
        all_rels = get_all_graph(graph, filters)
        @test isempty(all_rels)  # Soft-deleted, so hidden from get_all

        # Different user → no results
        other_rels = get_all_graph(graph, Dict{String, Any}("user_id" => "other"))
        @test isempty(other_rels)
    end

    @testset "LLMResponse" begin
        r = LLMResponse()
        @test r.content === nothing
        @test isempty(r.tool_calls)
        @test r.role == "assistant"

        r2 = LLMResponse(content="hello", tool_calls=[Dict{String,Any}("name" => "f")])
        @test r2.content == "hello"
        @test length(r2.tool_calls) == 1
    end

    @testset "Graph tools definitions" begin
        @test length(Mem0.EXTRACT_ENTITIES_TOOL) == 1
        @test Mem0.EXTRACT_ENTITIES_TOOL[1]["function"]["name"] == "extract_entities"

        @test length(Mem0.EXTRACT_RELATIONS_TOOL) == 1
        @test Mem0.EXTRACT_RELATIONS_TOOL[1]["function"]["name"] == "establish_relationships"

        @test length(Mem0.GRAPH_UPDATE_TOOL) == 1

        prompt = Mem0.extract_relations_prompt(user_id="alice")
        @test occursin("alice", prompt)

        prompt2 = Mem0.extract_relations_prompt(custom_prompt="Only extract work relations")
        @test occursin("Only extract work relations", prompt2)
    end

end  # top-level testset

# Run extended test suite
include("test_extended.jl")
