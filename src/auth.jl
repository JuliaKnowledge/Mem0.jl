# Credential / bearer-token resolution helper.
#
# OpenAI / Azure-OpenAI providers accept either a raw API key string or an
# external credential object (e.g. an `AzureIdentity.AbstractAzureCredential`).
# `_resolve_bearer` turns whatever the user supplied into a concrete bearer
# token string at request time.
#
# Extension packages (e.g. `Mem0AzureIdentityExt`) override this generic
# function to handle their own credential types without adding a hard
# dependency to Mem0.jl.

const AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

"""
    _resolve_bearer(source, scope=AZURE_COGNITIVE_SERVICES_SCOPE) -> String

Resolve `source` into a bearer token string.

- `AbstractString` → returned verbatim.
- `Nothing` → `""`.
- `Function` (zero-arg callable) → `String(source())`.
- Other types → error. Package extensions add methods for credential types
  (e.g. `AzureIdentity.AbstractAzureCredential`).
"""
_resolve_bearer(s::AbstractString, scope=AZURE_COGNITIVE_SERVICES_SCOPE) = String(s)
_resolve_bearer(::Nothing, scope=AZURE_COGNITIVE_SERVICES_SCOPE) = ""
_resolve_bearer(f::Function, scope=AZURE_COGNITIVE_SERVICES_SCOPE) = String(f())
_resolve_bearer(x, scope=AZURE_COGNITIVE_SERVICES_SCOPE) =
    error("Mem0: cannot resolve bearer token from object of type $(typeof(x)). " *
          "Pass a String API key, a zero-arg Function, or load a credential-aware " *
          "package extension (e.g. `using AzureIdentity`).")
