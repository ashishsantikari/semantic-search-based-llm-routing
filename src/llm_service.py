"""Mock LLM response generator with per-provider response templates and simulated latency."""

import asyncio
import random

from src.config import MOCK_LATENCY_MIN, MOCK_LATENCY_MAX


# ---------------------------------------------------------------------------
# 25 response templates: 5 providers × 5 verticals
# ---------------------------------------------------------------------------

_TEMPLATES: dict[tuple[str, str], str] = {}

# ── Anthropic: Verbose, careful language with caveats ─────────────────────
_TEMPLATES[("Anthropic", "HR")] = (
    "I want to provide a thorough analysis of your HR query: \"{query}\". "
    "In the context of human resources best practices, it is important to consider "
    "applicable employment laws and organizational policies. I recommend reviewing "
    "the specific regulatory framework before proceeding, as there may be jurisdiction-specific "
    "requirements that could affect the recommended approach. Please note that this "
    "analysis should be validated against your organization's current policies."
)
_TEMPLATES[("Anthropic", "Legal")] = (
    "I want to provide a thorough analysis of your legal question: \"{query}\". "
    "Based on established legal principles, several considerations arise. "
    "First, the specific factual context significantly influences the applicable "
    "legal framework. I would recommend consulting with a qualified attorney "
    "before relying on this analysis, as legal outcomes are highly fact-dependent "
    "and jurisdictional variations may apply. Caveat: this does not constitute "
    "legal advice or establish an attorney-client relationship."
)
_TEMPLATES[("Anthropic", "Compliance")] = (
    "I want to provide a thorough analysis of your compliance query: \"{query}\". "
    "Regulatory compliance requires careful attention to applicable frameworks "
    "and evolving standards. The analysis must account for both current requirements "
    "and anticipated regulatory developments. I recommend conducting a comprehensive "
    "gap assessment against the relevant regulatory baseline. As with all compliance "
    "matters, documentation of the decision-making process is strongly advised."
)
_TEMPLATES[("Anthropic", "Sales")] = (
    "I want to provide a thorough analysis of your sales question: \"{query}\". "
    "An effective sales strategy depends on a nuanced understanding of your target "
    "market, buyer personas, and competitive positioning. I would recommend a phased "
    "approach that begins with market research before scaling. Please bear in mind "
    "that sales tactics should be adapted to your specific industry context and "
    "organizational resources, as generic approaches often underperform."
)
_TEMPLATES[("Anthropic", "Engineering")] = (
    "I want to provide a thorough analysis of your engineering question: \"{query}\". "
    "When evaluating technical approaches, it is important to consider trade-offs "
    "between performance, maintainability, and scalability. I recommend conducting "
    "a thorough requirements analysis and considering edge cases before implementation. "
    "The proposed solution should be validated against your specific infrastructure "
    "constraints and team capabilities. Alternative architectures may merit consideration."
)

# ── DeepSeek: Technical, structured, code-like ────────────────────────────
_TEMPLATES[("DeepSeek", "HR")] = (
    "Here is a structured breakdown for the HR query: \"{query}\"\n\n"
    "## Analysis\n"
    "1. Identify core HR requirement\n"
    "2. Map to applicable policy framework\n"
    "3. Define process workflow\n\n"
    "## Recommendations\n"
    "- Follow standardized HR operating procedures\n"
    "- Document all decisions per compliance requirements\n"
    "- Reference organizational hierarchy for approval paths\n\n"
    "## Implementation\n"
    "```\nprocess = HRWorkflow()\nprocess.load_policy(query_context)\nresult = process.execute()\n```"
)
_TEMPLATES[("DeepSeek", "Legal")] = (
    "Here is a structured breakdown for the legal query: \"{query}\"\n\n"
    "## Legal Analysis\n"
    "1. Jurisdictional scope\n"
    "2. Applicable statutes and precedents\n"
    "3. Risk assessment matrix\n\n"
    "## Key Provisions\n"
    "- Identify relevant contract clauses\n"
    "- Determine statutory obligations\n"
    "- Assess liability exposure\n\n"
    "## Framework\n"
    "```\nlegal_framework = LegalAnalyzer()\nlegal_framework.set_jurisdiction(context)\nrisk_score = legal_framework.evaluate(query)\n```"
)
_TEMPLATES[("DeepSeek", "Compliance")] = (
    "Here is a structured breakdown for the compliance query: \"{query}\"\n\n"
    "## Regulatory Scope\n"
    "1. Identify applicable regulations\n"
    "2. Map control requirements\n"
    "3. Assess current compliance posture\n\n"
    "## Control Mapping\n"
    "| Control ID | Requirement | Status |\n"
    "|------------|-------------|--------|\n"
    "| REG-01     | Data handling| Pending |\n"
    "| REG-02     | Reporting    | Pending |\n\n"
    "## Remediation Steps\n"
    "```\ncontrols = ComplianceControls()\ncontrols.assess_gap(query)\ncontrols.generate_remediation_plan()\n```"
)
_TEMPLATES[("DeepSeek", "Sales")] = (
    "Here is a structured breakdown for the sales query: \"{query}\"\n\n"
    "## Pipeline Analysis\n"
    "1. Define target segments\n"
    "2. Calculate ICP fit score\n"
    "3. Optimize conversion funnel\n\n"
    "## Metrics\n"
    "- CAC: [calculated]\n"
    "- LTV: [projected]\n"
    "- Conversion rate: [baseline]\n\n"
    "## Strategy\n"
    "```\nstrategy = SalesStrategy()\nstrategy.segment_market(query_context)\nstrategy.allocate_resources()\n```"
)
_TEMPLATES[("DeepSeek", "Engineering")] = (
    "Here is a structured breakdown for the engineering query: \"{query}\"\n\n"
    "## Architecture Analysis\n"
    "1. Requirements decomposition\n"
    "2. Component identification\n"
    "3. Interface definition\n\n"
    "## Technical Specification\n"
    "- Input: query specifications\n"
    "- Process: optimized algorithm selection\n"
    "- Output: validated implementation\n\n"
    "## Implementation Plan\n"
    "```python\ndef solve(query_context):\n    # Parse requirements\n    reqs = parse_requirements(query_context)\n    # Design solution\n    solution = design_architecture(reqs)\n    # Validate\n    return validate(solution)\n```"
)

# ── Gemini: Balanced, informative, uses bullet points ─────────────────────
_TEMPLATES[("Gemini", "HR")] = (
    "Based on my analysis, here's a comprehensive overview for your HR query: \"{query}\"\n\n"
    "Here are the key points to consider:\n"
    "- Review your organization's HR policies and employment contracts thoroughly\n"
    "- Consider the employee's history and any prior performance documentation\n"
    "- Ensure compliance with local labor laws and statutory requirements\n"
    "- Document all steps taken for audit trail purposes\n"
    "- Consult with legal counsel for complex or high-risk situations\n\n"
    "In summary, a balanced approach that addresses both organizational needs "
    "and employee welfare tends to yield the best outcomes."
)
_TEMPLATES[("Gemini", "Legal")] = (
    "Based on my analysis, here's a comprehensive overview for your legal query: \"{query}\"\n\n"
    "Here are the key points to consider:\n"
    "- The specific jurisdiction and governing law will significantly impact the analysis\n"
    "- Review relevant statutes, regulations, and case law for your situation\n"
    "- Consider the statute of limitations and any procedural requirements\n"
    "- Document agreements in writing to avoid future disputes\n"
    "- Seek qualified legal representation for formal proceedings\n\n"
    "A methodical approach to legal analysis helps identify both risks and opportunities."
)
_TEMPLATES[("Gemini", "Compliance")] = (
    "Based on my analysis, here's a comprehensive overview for your compliance query: \"{query}\"\n\n"
    "Here are the key points to consider:\n"
    "- Identify all applicable regulatory frameworks for your industry and region\n"
    "- Conduct a thorough gap analysis against current requirements\n"
    "- Establish clear ownership and accountability for each control\n"
    "- Implement monitoring and reporting mechanisms\n"
    "- Schedule regular compliance reviews and updates\n\n"
    "A proactive compliance posture reduces risk and builds organizational resilience."
)
_TEMPLATES[("Gemini", "Sales")] = (
    "Based on my analysis, here's a comprehensive overview for your sales query: \"{query}\"\n\n"
    "Here are the key points to consider:\n"
    "- Define your ideal customer profile and target segments clearly\n"
    "- Align sales strategy with overall business objectives\n"
    "- Leverage data-driven insights for pipeline management\n"
    "- Invest in sales enablement and training for your team\n"
    "- Track and optimize key metrics: CAC, LTV, conversion rates\n\n"
    "A structured sales approach combined with adaptability drives consistent results."
)
_TEMPLATES[("Gemini", "Engineering")] = (
    "Based on my analysis, here's a comprehensive overview for your engineering query: \"{query}\"\n\n"
    "Here are the key points to consider:\n"
    "- Evaluate multiple architectural approaches before committing to one\n"
    "- Consider scalability, maintainability, and operational costs\n"
    "- Implement comprehensive testing at all levels\n"
    "- Plan for observability and monitoring from day one\n"
    "- Document design decisions and trade-offs for the team\n\n"
    "The best engineering solutions balance technical excellence with practical constraints."
)

# ── Llama: Direct, concise, practical ─────────────────────────────────────
_TEMPLATES[("Llama", "HR")] = (
    "Here's what you need to know about: \"{query}\"\n\n"
    "Key action items:\n"
    "1. Check your company's HR policy manual first\n"
    "2. Follow legal requirements in your jurisdiction\n"
    "3. Document everything thoroughly\n"
    "4. Escalate to HR management if needed\n\n"
    "Focus on clear communication and consistent policy application."
)
_TEMPLATES[("Llama", "Legal")] = (
    "Here's what you need to know about: \"{query}\"\n\n"
    "Key action items:\n"
    "1. Identify the relevant legal jurisdiction\n"
    "2. Review applicable contracts and agreements\n"
    "3. Check for deadlines and filing requirements\n"
    "4. Get everything in writing\n\n"
    "When in doubt, consult a qualified attorney."
)
_TEMPLATES[("Llama", "Compliance")] = (
    "Here's what you need to know about: \"{query}\"\n\n"
    "Key action items:\n"
    "1. Identify which regulations apply to you\n"
    "2. Check your current compliance status\n"
    "3. Address gaps immediately\n"
    "4. Set up ongoing monitoring\n\n"
    "Compliance is about consistency, not just one-time fixes."
)
_TEMPLATES[("Llama", "Sales")] = (
    "Here's what you need to know about: \"{query}\"\n\n"
    "Key action items:\n"
    "1. Know your customer and their pain points\n"
    "2. Focus on value proposition over features\n"
    "3. Follow up consistently\n"
    "4. Track everything in your CRM\n\n"
    "Sales success comes from disciplined execution."
)
_TEMPLATES[("Llama", "Engineering")] = (
    "Here's what you need to know about: \"{query}\"\n\n"
    "Key action items:\n"
    "1. Start with a clear requirements doc\n"
    "2. Choose the simplest viable approach\n"
    "3. Write tests before implementation\n"
    "4. Review and iterate\n\n"
    "Ship early, ship often, but maintain quality."
)

# ── Kimi: Thorough, detail-oriented with contextual analysis ──────────────
_TEMPLATES[("Kimi", "HR")] = (
    "Let me provide a detailed analysis with context for your HR query: \"{query}\"\n\n"
    "Context: Human Resources queries require balancing organizational objectives "
    "with employee rights and well-being. The specific approach depends on company "
    "size, industry sector, and regional labor laws.\n\n"
    "Detailed breakdown:\n"
    "• Policy Framework: Review existing HR policies and handbooks for applicable procedures\n"
    "• Legal Context: Consider employment law requirements including discrimination, "
    "harassment, and labor standards\n"
    "• Operational Impact: Assess how decisions affect team dynamics and productivity\n"
    "• Documentation: Maintain comprehensive records for audit and legal purposes\n\n"
    "In context, a methodical approach that considers both organizational and individual "
    "factors will produce the most balanced outcome."
)
_TEMPLATES[("Kimi", "Legal")] = (
    "Let me provide a detailed analysis with context for your legal query: \"{query}\"\n\n"
    "Context: Legal analysis requires careful consideration of jurisdictional "
    "frameworks, precedent, and statutory interpretation. The specific legal "
    "landscape varies significantly by region and practice area.\n\n"
    "Detailed breakdown:\n"
    "• Jurisdictional Analysis: Determine which laws and courts have authority\n"
    "• Statutory Framework: Identify relevant statutes, regulations, and case law\n"
    "• Risk Assessment: Evaluate potential liabilities and exposure\n"
    "• Procedural Requirements: Check for filing deadlines and procedural rules\n\n"
    "In context, thorough legal analysis should identify both strengths and vulnerabilities "
    "in your position."
)
_TEMPLATES[("Kimi", "Compliance")] = (
    "Let me provide a detailed analysis with context for your compliance query: \"{query}\"\n\n"
    "Context: Regulatory compliance is increasingly complex with overlapping "
    "requirements from multiple authorities. Organizations must navigate both "
    "industry-specific regulations and general data protection/privacy laws.\n\n"
    "Detailed breakdown:\n"
    "• Regulatory Landscape: Map all applicable regulations by jurisdiction and scope\n"
    "• Control Framework: Identify required controls and current implementation status\n"
    "• Gap Analysis: Document deficiencies and remediation priorities\n"
    "• Monitoring: Establish ongoing compliance monitoring and reporting\n\n"
    "In context, a mature compliance program treats regulation as a strategic "
    "differentiator rather than a burden."
)
_TEMPLATES[("Kimi", "Sales")] = (
    "Let me provide a detailed analysis with context for your sales query: \"{query}\"\n\n"
    "Context: Modern sales requires a data-driven approach that balances "
    "relationship-building with systematic pipeline management. Success depends "
    "on understanding market dynamics and buyer behavior.\n\n"
    "Detailed breakdown:\n"
    "• Market Context: Analyze industry trends and competitive landscape\n"
    "• Buyer Persona: Define target segments and their decision criteria\n"
    "• Sales Process: Map the customer journey and optimize touchpoints\n"
    "• Performance Metrics: Track conversion rates, deal velocity, and win rates\n\n"
    "In context, sales effectiveness comes from consistent execution of a "
    "well-defined process adapted to your specific market."
)
_TEMPLATES[("Kimi", "Engineering")] = (
    "Let me provide a detailed analysis with context for your engineering query: \"{query}\"\n\n"
    "Context: Engineering decisions involve complex trade-offs between "
    "performance, scalability, cost, and developer experience. The optimal approach "
    "depends on specific requirements, constraints, and team capabilities.\n\n"
    "Detailed breakdown:\n"
    "• Requirements Analysis: Understand functional and non-functional requirements\n"
    "• Architecture Evaluation: Compare alternative approaches and trade-offs\n"
    "• Implementation Strategy: Plan development phases and resource allocation\n"
    "• Quality Assurance: Define testing strategy and performance benchmarks\n\n"
    "In context, the best engineering solution is one that your team can reliably "
    "build, operate, and evolve over time."
)


class MockLLMService:
    """Async mock LLM response generator with per-provider response templates."""

    async def generate(
        self, provider: str, vertical: str, query: str
    ) -> tuple[str, float]:
        """Simulate an LLM generation call.

        Args:
            provider: LLM provider name (e.g. "Anthropic", "DeepSeek").
            vertical: Vertical category (e.g. "Legal", "Engineering").
            query: The user's original query text.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        # Simulate network latency
        latency_seconds = random.uniform(MOCK_LATENCY_MIN, MOCK_LATENCY_MAX)
        await asyncio.sleep(latency_seconds)

        # Look up the provider-vertical template
        template = _TEMPLATES.get((provider, vertical))
        if template is None:
            response = (
                f"Response from {provider} regarding your {vertical} query: "
                f"\"{query}\". This provider does not have a specific template "
                f"for this vertical."
            )
        else:
            response = template.format(query=query)

        return response, latency_seconds * 1000.0
