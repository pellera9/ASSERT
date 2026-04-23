"""Task 0: Prereq audit for phoenix demos."""
import importlib, importlib.util, os
from dotenv import load_dotenv
load_dotenv()

frameworks = {
    "litellm":      {"sdk": "litellm",        "cred": "any"},
    "langchain":    {"sdk": "langchain_openai","cred": "openai"},
    "openai":       {"sdk": "openai",         "cred": "openai"},
    "anthropic":    {"sdk": "anthropic",      "cred": "anthropic"},
    "groq":         {"sdk": "groq",           "cred": "groq"},
    "mistralai":    {"sdk": "mistralai",      "cred": "mistral"},
    "crewai":       {"sdk": "crewai",         "cred": "openai"},
    "dspy":         {"sdk": "dspy",           "cred": "openai"},
    "llamaindex":   {"sdk": "llama_index",    "cred": "openai"},
    "bedrock":      {"sdk": "boto3",          "cred": "aws"},
    "google_genai": {"sdk": "google.genai",   "cred": "gemini"},
    "google_adk":   {"sdk": "google.adk",     "cred": "gemini"},
    "portkey":      {"sdk": "portkey_ai",     "cred": "portkey"},
}

creds = {
    "any":       True,
    "openai":    bool(os.environ.get("OPENAI_API_KEY")),
    "azure":     bool(os.environ.get("AZURE_API_KEY")),
    "gemini":    bool(os.environ.get("GEMINI_API_KEY")),
    "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    "groq":      bool(os.environ.get("GROQ_API_KEY")),
    "mistral":   bool(os.environ.get("MISTRAL_API_KEY")),
    "aws":       bool(os.environ.get("AWS_ACCESS_KEY_ID")),
    "portkey":   bool(os.environ.get("PORTKEY_API_KEY")),
}

print("Available credentials:", [k for k, v in creds.items() if v])
print()
print(f"{'Framework':<14s} {'SDK':>10s} {'Cred':>8s} {'Status':>10s}")
print("-" * 48)

runnable = []
installable = []
for name, info in frameworks.items():
    sdk_ok = importlib.util.find_spec(info["sdk"].split(".")[0]) is not None
    cred_ok = creds.get(info["cred"], False)
    if sdk_ok and cred_ok:
        status = "RUNNABLE"
        runnable.append(name)
    elif not sdk_ok and cred_ok:
        status = "INSTALL"
        installable.append(name)
    elif sdk_ok and not cred_ok:
        status = "NO CRED"
    else:
        status = "SKIP"
    print(f"  {name:<12s} {'OK' if sdk_ok else 'MISS':>10s} {'OK' if cred_ok else 'MISS':>8s} {status:>10s}")

print(f"\nRunnable now:  {runnable}")
print(f"Installable:   {installable}")
print(f"Skip (no cred):{[n for n,i in frameworks.items() if n not in runnable and n not in installable]}")
