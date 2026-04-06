# Neo4j graph store unit tests
# These tests verify connectivity and CRUD operations against a running Neo4j instance.
# Skip gracefully if Neo4j is not available.

using Test
using Mem0
using Random

const NEO4J_URL = get(ENV, "NEO4J_URL", "http://localhost:7474")
const NEO4J_USER = get(ENV, "NEO4J_USER", "neo4j")
const NEO4J_PASS = get(ENV, "NEO4J_PASS", "password")
const NEO4J_DB   = get(ENV, "NEO4J_DB",   "neo4j")

function neo4j_available()
    try
        resp = HTTP.post("$(NEO4J_URL)/db/$(NEO4J_DB)/tx/commit",
            ["Content-Type" => "application/json",
             "Authorization" => "Basic " * Base64.base64encode("$(NEO4J_USER):$(NEO4J_PASS)")],
            JSON3.write(Dict("statements" => [Dict("statement" => "RETURN 1")]));
            status_exception=false)
        return resp.status in 200:299
    catch
        return false
    end
end

using HTTP, JSON3, Base64

if !neo4j_available()
    @info "Neo4j not available at $NEO4J_URL — skipping Neo4j tests"
else

@testset "Neo4jGraphStore" begin

    # Cleanup helper: delete all test-scoped nodes
    function cleanup_neo4j!(graph, user_id)
        Mem0.delete_all_graph!(graph, Dict{String, Any}("user_id" => user_id))
    end

    # --- Construction tests ---
    @testset "Construction" begin
        # Mock LLM and embedder for construction-only tests
        llm_config = LlmConfig(provider="ollama",
            config=Dict{String, Any}("model" => "qwen3:8b",
                                     "temperature" => 0.1, "max_tokens" => 1000))
        embed_config = EmbedderConfig(provider="ollama",
            config=Dict{String, Any}("model" => "nomic-embed-text",
                                     "embedding_dims" => 768))
        llm = create_llm(llm_config)
        embedder = create_embedder(embed_config)

        @testset "basic construction" begin
            gs = Neo4jGraphStore(llm=llm, embedder=embedder,
                config=Dict{String,Any}("url" => NEO4J_URL,
                                        "username" => NEO4J_USER,
                                        "password" => NEO4J_PASS,
                                        "database" => NEO4J_DB))
            @test gs isa AbstractGraphStore
            @test gs.database == NEO4J_DB
        end

        @testset "bolt URL resolution" begin
            gs = Neo4jGraphStore(llm=llm, embedder=embedder,
                config=Dict{String,Any}("url" => "neo4j://127.0.0.1:7687",
                                        "username" => NEO4J_USER,
                                        "password" => NEO4J_PASS))
            @test gs.http_url == "http://127.0.0.1:7474"
        end

        @testset "missing credentials" begin
            @test_throws Mem0Error Neo4jGraphStore(llm=llm, embedder=embedder,
                config=Dict{String,Any}("url" => NEO4J_URL))
        end

        @testset "factory construction" begin
            gc = GraphStoreConfig(
                provider="neo4j",
                config=Dict{String,Any}("url" => NEO4J_URL,
                                        "username" => NEO4J_USER,
                                        "password" => NEO4J_PASS,
                                        "database" => NEO4J_DB),
                threshold=0.6,
            )
            gs = create_graph_store(gc, llm_config, embed_config)
            @test gs isa Neo4jGraphStore
            @test gs.threshold == 0.6
        end
    end

    # --- Cypher connectivity ---
    @testset "Cypher connectivity" begin
        llm = create_llm(LlmConfig(provider="ollama",
            config=Dict{String,Any}("model" => "qwen3:8b")))
        embedder = create_embedder(EmbedderConfig(provider="ollama",
            config=Dict{String,Any}("model" => "nomic-embed-text", "embedding_dims" => 768)))

        gs = Neo4jGraphStore(llm=llm, embedder=embedder,
            config=Dict{String,Any}("url" => NEO4J_URL,
                                    "username" => NEO4J_USER,
                                    "password" => NEO4J_PASS,
                                    "database" => NEO4J_DB))

        result = Mem0._cypher_query(gs, "RETURN 42 AS answer")
        @test length(result.rows) == 1
        @test result.rows[1][1] == 42
        @test result.columns == ["answer"]
    end

    # --- CRUD operations ---
    @testset "CRUD operations" begin
        llm = create_llm(LlmConfig(provider="ollama",
            config=Dict{String,Any}("model" => "qwen3:8b",
                                    "temperature" => 0.1, "max_tokens" => 1000)))
        embedder = create_embedder(EmbedderConfig(provider="ollama",
            config=Dict{String,Any}("model" => "nomic-embed-text", "embedding_dims" => 768)))

        gs = Neo4jGraphStore(llm=llm, embedder=embedder,
            config=Dict{String,Any}("url" => NEO4J_URL,
                                    "username" => NEO4J_USER,
                                    "password" => NEO4J_PASS,
                                    "database" => NEO4J_DB),
            threshold=0.5)

        test_uid = "neo4j_test_$(randstring(8))"
        filters = Dict{String, Any}("user_id" => test_uid)

        try
            @testset "get_all — initially empty" begin
                rels = get_all_graph(gs, filters)
                @test isempty(rels)
            end

            @testset "add_to_graph!" begin
                rels = add_to_graph!(gs, "Alice works at Microsoft. Bob works at Google.", filters)
                @test rels isa Vector
                @info "Extracted $(length(rels)) relations from text"
            end

            @testset "get_all — after add" begin
                rels = get_all_graph(gs, filters)
                @test length(rels) >= 1
                for r in rels
                    @test haskey(r, "source")
                    @test haskey(r, "relationship")
                    @test haskey(r, "destination")
                end
                @info "Retrieved $(length(rels)) relations" rels
            end

            @testset "search_graph" begin
                results = search_graph(gs, "Alice", filters)
                @test results isa Vector
                @info "Search for 'Alice' returned $(length(results)) results" results
            end

            @testset "delete_from_graph! (soft delete)" begin
                delete_from_graph!(gs, "Alice works at Microsoft", filters)
                # After soft delete, get_all should return fewer or zero
                rels = get_all_graph(gs, filters)
                @info "After soft delete: $(length(rels)) valid relations"
            end

            @testset "delete_all_graph!" begin
                delete_all_graph!(gs, filters)
                rels = get_all_graph(gs, filters)
                @test isempty(rels)
            end
        finally
            # Cleanup: remove all test nodes
            try
                cleanup_neo4j!(gs, test_uid)
            catch e
                @warn "Cleanup failed" exception=e
            end
        end
    end

    # --- Filter isolation ---
    @testset "Filter isolation" begin
        llm = create_llm(LlmConfig(provider="ollama",
            config=Dict{String,Any}("model" => "qwen3:8b",
                                    "temperature" => 0.1, "max_tokens" => 1000)))
        embedder = create_embedder(EmbedderConfig(provider="ollama",
            config=Dict{String,Any}("model" => "nomic-embed-text", "embedding_dims" => 768)))

        gs = Neo4jGraphStore(llm=llm, embedder=embedder,
            config=Dict{String,Any}("url" => NEO4J_URL,
                                    "username" => NEO4J_USER,
                                    "password" => NEO4J_PASS,
                                    "database" => NEO4J_DB),
            threshold=0.5)

        uid_a = "neo4j_isolation_a_$(randstring(6))"
        uid_b = "neo4j_isolation_b_$(randstring(6))"

        try
            add_to_graph!(gs, "Charlie likes Julia programming.", Dict{String,Any}("user_id" => uid_a))
            add_to_graph!(gs, "Diana prefers Python programming.", Dict{String,Any}("user_id" => uid_b))

            rels_a = get_all_graph(gs, Dict{String,Any}("user_id" => uid_a))
            rels_b = get_all_graph(gs, Dict{String,Any}("user_id" => uid_b))

            # Each user's data should be isolated
            for r in rels_a
                @test !any(lowercase(r["source"]) == "diana" || lowercase(r["destination"]) == "diana"
                           for _ in 1:1)
            end
            for r in rels_b
                @test !any(lowercase(r["source"]) == "charlie" || lowercase(r["destination"]) == "charlie"
                           for _ in 1:1)
            end
        finally
            try; delete_all_graph!(gs, Dict{String,Any}("user_id" => uid_a)); catch; end
            try; delete_all_graph!(gs, Dict{String,Any}("user_id" => uid_b)); catch; end
        end
    end

end # testset Neo4jGraphStore

end # if neo4j_available
