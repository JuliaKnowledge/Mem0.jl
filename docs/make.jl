using Documenter
using Mem0

makedocs(
    sitename = "Mem0.jl",
    modules = [Mem0],
    remotes = nothing,
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://sdwfrost.github.io/Mem0.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Configuration" => "guide/configuration.md",
            "Providers" => "guide/providers.md",
            "Graph Memory" => "guide/graph_memory.md",
        ],
        "API Reference" => [
            "Memory" => "api/memory.md",
            "Types" => "api/types.md",
            "Providers" => "api/providers.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo = "github.com/sdwfrost/Mem0.jl.git",
    push_preview = true,
)
