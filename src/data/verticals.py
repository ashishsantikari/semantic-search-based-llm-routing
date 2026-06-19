"""Seed data: vertical category definitions and exemplar queries for semantic routing."""

from dataclasses import dataclass, field


@dataclass
class VerticalDef:
    """Definition of a vertical category with exemplar queries for semantic routing."""

    label: str
    description: str
    exemplar_queries: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vertical definitions
# ---------------------------------------------------------------------------

HR = VerticalDef(
    label="HR",
    description=(
        "Human Resources queries covering employee management, recruitment, "
        "onboarding, payroll, benefits, performance reviews, and workplace policies."
    ),
    exemplar_queries=[
        "How do I handle a employee performance improvement plan?",
        "What is the process for processing payroll for new hires?",
        "Write a job description for a senior software engineer position",
        "How do I calculate overtime pay for salaried employees?",
        "What are best practices for conducting exit interviews?",
    ],
)

Legal = VerticalDef(
    label="Legal",
    description=(
        "Legal queries covering contract law, regulations, intellectual property, "
        "compliance requirements, dispute resolution, and corporate governance."
    ),
    exemplar_queries=[
        "Draft a non-disclosure agreement for a partnership",
        "What are the legal requirements for terminating an employee in California?",
        "Explain the difference between a contractor and an employee for tax purposes",
        "How do I file a trademark application?",
        "What are the key clauses to include in a SaaS service agreement?",
    ],
)

Compliance = VerticalDef(
    label="Compliance",
    description=(
        "Regulatory compliance queries covering data privacy (GDPR, CCPA), industry "
        "regulations, audit requirements, policy enforcement, and risk management."
    ),
    exemplar_queries=[
        "What steps do I need to take to become GDPR compliant?",
        "How do I conduct a data protection impact assessment?",
        "What are the reporting requirements for a data breach?",
        "Create a compliance checklist for SOC 2 Type II",
        "How do I handle a regulatory audit request?",
    ],
)

Sales = VerticalDef(
    label="Sales",
    description=(
        "Sales queries covering lead generation, CRM management, sales pipelines, "
        "negotiation strategies, pricing, forecasting, and customer relationship management."
    ),
    exemplar_queries=[
        "Write a cold email template for enterprise SaaS prospects",
        "How do I calculate customer acquisition cost (CAC)?",
        "What's the best way to structure a sales territory plan?",
        "Create a qualification checklist using BANT methodology",
        "How do I handle pricing objections in enterprise sales?",
    ],
)

Engineering = VerticalDef(
    label="Engineering",
    description=(
        "Engineering and technical queries covering software architecture, code review, "
        "system design, DevOps, debugging, testing, and technology stack decisions."
    ),
    exemplar_queries=[
        "Design a scalable microservices architecture for an e-commerce platform",
        "How do I optimize a PostgreSQL query that's running slow?",
        "Explain the difference between REST and GraphQL API design",
        "What are best practices for CI/CD pipeline setup?",
        "How do I implement rate limiting in a Python FastAPI application?",
    ],
)

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_VERTICALS = [HR, Legal, Compliance, Sales, Engineering]

VERTICAL_LABELS: dict[str, VerticalDef] = {v.label: v for v in _VERTICALS}


def get_all_verticals() -> list[VerticalDef]:
    """Return all vertical definitions."""
    return list(_VERTICALS)
