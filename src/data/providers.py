"""Seed data: LLM provider capability definitions for semantic routing."""

from dataclasses import dataclass, field


@dataclass
class ProviderDef:
    """Definition of an LLM provider with capability scores per vertical."""

    label: str
    description: str
    vertical_strengths: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

Anthropic = ProviderDef(
    label="Anthropic",
    description=(
        "Excels at nuanced reasoning, safety-conscious responses, detailed analysis, "
        "and complex document understanding. Strong at handling ambiguity and providing "
        "well-structured, thoughtful answers with careful caveats."
    ),
    vertical_strengths={
        "HR": 0.85,
        "Legal": 0.95,
        "Compliance": 0.90,
        "Sales": 0.70,
        "Engineering": 0.80,
    },
)

DeepSeek = ProviderDef(
    label="DeepSeek",
    description=(
        "Highly capable at code generation, technical problem-solving, mathematical "
        "reasoning, and structured technical tasks. Offers strong performance on "
        "engineering and analytical queries with excellent instruction following."
    ),
    vertical_strengths={
        "HR": 0.55,
        "Legal": 0.60,
        "Compliance": 0.70,
        "Sales": 0.65,
        "Engineering": 0.95,
    },
)

Gemini = ProviderDef(
    label="Gemini",
    description=(
        "Combines broad world knowledge with strong multimodal understanding. "
        "Well-suited for research-oriented queries, creative tasks, and questions "
        "requiring up-to-date information synthesis."
    ),
    vertical_strengths={
        "HR": 0.70,
        "Legal": 0.65,
        "Compliance": 0.75,
        "Sales": 0.85,
        "Engineering": 0.75,
    },
)

Llama = ProviderDef(
    label="Llama",
    description=(
        "Efficient for general-purpose tasks with strong instruction-following "
        "capabilities. Good balance of performance and resource efficiency across "
        "diverse query types."
    ),
    vertical_strengths={
        "HR": 0.75,
        "Legal": 0.60,
        "Compliance": 0.65,
        "Sales": 0.80,
        "Engineering": 0.70,
    },
)

Kimi = ProviderDef(
    label="Kimi",
    description=(
        "Excels at long-context understanding (up to 200K tokens), Chinese-English "
        "bilingual tasks, and processing large documents. Particularly strong at "
        "comprehensive document analysis and summarization."
    ),
    vertical_strengths={
        "HR": 0.65,
        "Legal": 0.75,
        "Compliance": 0.80,
        "Sales": 0.60,
        "Engineering": 0.60,
    },
)

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_PROVIDERS = [Anthropic, DeepSeek, Gemini, Llama, Kimi]

PROVIDER_LABELS: dict[str, ProviderDef] = {p.label: p for p in _PROVIDERS}


def get_all_providers() -> list[ProviderDef]:
    """Return all provider definitions."""
    return list(_PROVIDERS)
