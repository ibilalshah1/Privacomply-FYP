"""
RAG Compliance Pipeline — Privacy Policy Violation Detector

Flow:
  1. Preload  → at startup, retrieve Qdrant chunks using query_hints only (policy-independent)
               and build static cached system prompts per batch
  2. Decompose → split policy into per-category relevant clauses (Claude, system cached)
  3. Judge    → 3 batched LLM calls; system (legal chunks + labels) is cached,
               only policy excerpts are sent fresh per request

Prompt caching strategy:
  - decompose: system = instruction + all 15 category names → cached
               user   = policy text → fresh
  - judge batch i: system = judge instructions + preloaded legal chunks + label defs → cached
                   user   = regulation + policy excerpts for 5 categories → fresh
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [_ROOT, os.path.join(_ROOT, "metadata")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import anthropic
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from config import (
    ANTHROPIC_API_KEY,
    CATEGORIES,
    CATEGORY_MAP,
    COLLECTION_NAME,
    EMBED_MODEL,
    JUDGE_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    TOP_K_PER_CATEGORY,
)
from models import CategoryResult, ComplianceReport, LabelResult

# ─────────────────────────────────────────────
embedding_model  = SentenceTransformer(EMBED_MODEL)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
qdrant           = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

# Holds preloaded Qdrant context per regulation — built once at first analyze() call
_preloaded_data: dict[str, dict] = {}


# ═══════════════════════════════════════════════
# SHARED LLM CALL — Claude with prompt caching
# ═══════════════════════════════════════════════

def _claude_call(
    system_text: str,
    fresh_user_text: str,
    max_tokens: int,
    retries: int = 4,
) -> str:
    """
    Call Claude Haiku with the system prompt cached (ephemeral).
    system_text is sent with cache_control so repeated calls with the same
    system hit the cache instead of re-tokenizing.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            response = anthropic_client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": fresh_user_text,
                }],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError as e:
            wait = 2 ** attempt
            print(f"  [claude] rate limited — waiting {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code in (529, 503):
                wait = 2 ** attempt
                print(f"  [claude] overloaded ({e.status_code}) — waiting {wait}s")
                time.sleep(wait)
                last_exc = e
            else:
                raise
    raise last_exc


# ═══════════════════════════════════════════════
# QDRANT HELPERS
# ═══════════════════════════════════════════════

def embed_query(text: str) -> list[float]:
    return embedding_model.encode(text).tolist()


def embed_queries_batch(texts: list[str]) -> list[list[float]]:
    vectors = embedding_model.encode(texts, batch_size=len(texts), show_progress_bar=False)
    return [v.tolist() for v in vectors]


def build_qdrant_filter(regulation: str, category_id: int) -> Filter:
    if regulation == "both":
        reg_condition = FieldCondition(
            key="regulation",
            match=MatchAny(any=["gdpr", "pdpa", "eprivacy"]),
        )
    else:
        reg_condition = FieldCondition(
            key="regulation",
            match=MatchAny(any=[regulation, "eprivacy"]),
        )
    cat_condition = FieldCondition(
        key="categories",
        match=MatchValue(value=category_id),
    )
    return Filter(must=[reg_condition, cat_condition])


def retrieve_for_category(
    category: dict,
    regulation: str,
    policy_excerpt: str,
    query_vec: list[float] | None = None,
) -> list[dict]:
    if query_vec is None:
        query_text = f"{category['query_hint']}\n\n{policy_excerpt}"
        query_vec = embed_query(query_text)

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=build_qdrant_filter(regulation, category["id"]),
        limit=TOP_K_PER_CATEGORY,
        with_payload=True,
    )
    return [
        {
            "text":          hit.payload.get("text", ""),
            "source_title":  hit.payload.get("source_title", ""),
            "article":       hit.payload.get("article", ""),
            "article_title": hit.payload.get("article_title", ""),
            "regulation":    hit.payload.get("regulation", ""),
            "score":         hit.score,
        }
        for hit in response.points
    ]


# ═══════════════════════════════════════════════
# PRELOAD — build cached batch contexts at startup
# ═══════════════════════════════════════════════

_JUDGE_SYSTEM_BASE = """\
You are an expert data protection lawyer and privacy compliance auditor specialising in GDPR and \
Pakistan's Personal Data Protection Bill (PDPA).

Evaluate a website's privacy policy against specific compliance labels using the retrieved legal \
reference material below to ground your judgments.

Be strict and precise. If a label is not clearly addressed in the policy, mark it as MISSING.

Return ONLY a JSON object. Keys are exact category names. Values are JSON arrays of label result objects.
No preamble, no markdown fences.

Label result schema:
{
  "label": "<label text>",
  "priority": "<Critical|High|Medium>",
  "compliant": <bool>,
  "violation": <bool>,
  "missing": <bool>,
  "explanation": "<1-2 sentences on what was found or not found>",
  "policy_excerpt": "<exact policy text evaluated, or empty string>",
  "legal_basis": "<article/section + document name>",
  "recommendation": "<what must be added/changed to comply, or empty if compliant>"
}
"""

_DECOMPOSE_SYSTEM = """\
You are a legal analyst decomposing a website privacy policy for compliance review.

Given the full privacy policy text, extract the specific clauses, sentences, and paragraphs \
that are relevant to EACH of the following compliance categories.

For each category, quote or closely paraphrase the EXACT policy text that relates to it.
If the policy says nothing about a category, return an empty string for that category.

Output ONLY a JSON object with category names as keys and relevant policy text as values.
No preamble, no explanation, no markdown fences.

Compliance categories and what to look for:

- Data Collection: types of personal data collected, purpose, data sources, voluntary vs mandatory
- Data Processing: lawful basis (consent/contract/legitimate interest), processing purpose, methods, secondary use limits
- Third-Party Sharing: which third parties receive data, sharing purpose, consent requirements, data sharing agreements
- User Rights and Control: consent mechanisms, opt-out, right to access / rectify / erase / port / restrict / object
- Data Security: technical and organisational security measures, encryption, pseudonymisation, unauthorised access prevention
- Data Retention: how long data is kept, retention purpose, deletion or anonymisation process
- Data Breach Notification: breach notification to users and/or authorities, timing commitments
- Cross-Border Transfers: international data transfers, adequacy decisions, standard contractual clauses, data localisation
- Transparency: plain language, accessible policy location, notification of policy changes
- Accountability: Data Protection Officer appointment, compliance audits, Data Protection Impact Assessments
- Children's Data: age thresholds, parental consent, child-specific safeguards
- Automated Decisions: profiling, automated decision-making, user rights against solely automated processing
- Cookies & Tracking: cookie usage, consent options, tracking purposes and categories
- Special Category Data: sensitive data (health, biometrics, racial, religious, etc.), additional safeguards
- Supervisory Authority: supervisory authority contact details, complaint lodging process
"""


def _preload_for_regulation(regulation: str):
    """
    Pre-retrieve Qdrant chunks using query_hints only (no policy text).
    Chunks are policy-independent, so the built system prompts will be identical
    across requests → Claude caches them after the first call.
    """
    if regulation in _preloaded_data:
        return

    print(f"[pipeline] Preloading Qdrant context for '{regulation}'...")

    hints = [cat["query_hint"] for cat in CATEGORIES]
    hint_vectors = embed_queries_batch(hints)

    def _fetch(args):
        cat, qvec = args
        try:
            return cat["id"], retrieve_for_category(cat, regulation, "", query_vec=qvec)
        except Exception as e:
            print(f"  [WARN] Preload retrieval failed for {cat['name']}: {e}")
            return cat["id"], []

    with ThreadPoolExecutor(max_workers=15) as ex:
        chunks_map: dict[int, list[dict]] = dict(
            ex.map(_fetch, zip(CATEGORIES, hint_vectors))
        )

    # Build one cached system string per batch of 5 categories
    BATCH_SIZE = 5
    batches = [CATEGORIES[i:i + BATCH_SIZE] for i in range(0, len(CATEGORIES), BATCH_SIZE)]
    batch_systems: dict[int, str] = {}

    for bi, batch_cats in enumerate(batches):
        # Label definitions (static)
        label_defs_parts = []
        for cat in batch_cats:
            labels_str = "\n".join(
                f"  - [{l['priority']}] {l['text']}" for l in cat["labels"]
            )
            label_defs_parts.append(f"### {cat['name']}\nLabels to evaluate:\n{labels_str}")
        label_defs_str = "\n\n".join(label_defs_parts)

        # Legal chunks (preloaded from Qdrant)
        context_parts = []
        for cat in batch_cats:
            for chunk in chunks_map.get(cat["id"], [])[:3]:
                context_parts.append(
                    f"[{cat['name']} | {chunk['regulation'].upper()} {chunk['article']}"
                    f" — {chunk['source_title']}]\n{chunk['text'][:500]}"
                )
        legal_context = "\n---\n".join(context_parts) if context_parts else "No legal context available."

        batch_systems[bi] = (
            _JUDGE_SYSTEM_BASE
            + "\n\n## Compliance Categories and Labels\n\n"
            + label_defs_str
            + "\n\n## Retrieved Legal Reference Material\n\n"
            + legal_context
        )

    _preloaded_data[regulation] = {
        "chunks_map":    chunks_map,
        "batch_systems": batch_systems,
    }
    print(f"[pipeline] Preload complete — {len(chunks_map)} categories loaded.")


# ═══════════════════════════════════════════════
# STEP 1 — DECOMPOSE POLICY
# ═══════════════════════════════════════════════

def decompose_policy(policy_text: str) -> dict[str, str]:
    """
    Ask Claude to extract per-category relevant clauses from the policy.
    System prompt (with all 15 categories) is cached; only policy text is sent fresh.
    """
    words = policy_text.split()
    if len(words) > 6000:
        policy_text = " ".join(words[:6000]) + "\n[... truncated ...]"

    raw = _claude_call(
        system_text=_DECOMPOSE_SYSTEM,
        fresh_user_text=f"Privacy Policy:\n\n{policy_text}",
        max_tokens=4096,
    )

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        category_names = [c["name"] for c in CATEGORIES]
        return {name: policy_text for name in category_names}


# ═══════════════════════════════════════════════
# STEP 2 — JUDGE CATEGORIES (batched, cached)
# ═══════════════════════════════════════════════

def _parse_label_results(items: list, category: dict) -> list[LabelResult]:
    if not items:
        return [
            LabelResult(
                label=l["text"], priority=l["priority"],
                compliant=False, violation=False, missing=True,
                explanation="No LLM response for this category.",
                policy_excerpt="", legal_basis="", recommendation="Review manually.",
            )
            for l in category["labels"]
        ]
    return [
        LabelResult(
            label=item.get("label", ""),
            priority=item.get("priority", "Medium"),
            compliant=item.get("compliant", False),
            violation=item.get("violation", False),
            missing=item.get("missing", True),
            explanation=item.get("explanation", ""),
            policy_excerpt=item.get("policy_excerpt", ""),
            legal_basis=item.get("legal_basis", ""),
            recommendation=item.get("recommendation", ""),
        )
        for item in items
    ]


def judge_categories_batch(
    batch_index: int,
    categories: list[dict],
    regulation: str,
    policy_excerpts: dict[str, str],
    cached_system: str,
) -> dict[int, list[LabelResult]]:
    """
    Evaluate 5 categories in one Claude call.
    cached_system (legal chunks + label defs) is sent as a cached system prompt.
    Only regulation label + policy excerpts are sent fresh per request.
    """
    reg_label = {
        "gdpr": "GDPR (EU Regulation 2016/679)",
        "pdpa": "Pakistan Personal Data Protection Bill 2023",
        "both": "GDPR and Pakistan PDPA",
    }.get(regulation, regulation)

    excerpt_parts = []
    for cat in categories:
        excerpt = policy_excerpts.get(cat["name"]) or "[No relevant text found in policy]"
        excerpt_parts.append(f"### {cat['name']}\n{excerpt}")
    excerpts_block = "\n\n---\n\n".join(excerpt_parts)

    fresh_user = (
        f"Regulation: {reg_label}\n\n"
        f"## Privacy Policy Excerpts (per category)\n\n"
        f"{excerpts_block}\n\n"
        f"Evaluate every label for every category above and return the JSON object."
    )

    raw = _claude_call(
        system_text=cached_system,
        fresh_user_text=fresh_user,
        max_tokens=8000,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return {
        cat["id"]: _parse_label_results(data.get(cat["name"], []), cat)
        for cat in categories
    }


# ═══════════════════════════════════════════════
# STEP 3 — AGGREGATE INTO REPORT
# ═══════════════════════════════════════════════

def generate_summary(url: str, regulation: str, overall_score: float,
                     total_critical: int, total_high: int,
                     worst_names: list[str]) -> str:
    reg_label = {
        "gdpr": "GDPR (EU 2016/679)",
        "pdpa": "Pakistan PDPA 2023",
        "both": "GDPR and Pakistan PDPA",
    }.get(regulation, regulation.upper())

    if overall_score >= 0.8:
        risk, status = "low risk", "largely compliant"
    elif overall_score >= 0.5:
        risk, status = "medium risk", "partially compliant"
    else:
        risk, status = "high risk", "significantly non-compliant"

    parts = [
        f"The privacy policy of {url or 'this website'} is {status} with {reg_label}, "
        f"scoring {overall_score:.0%} overall ({risk})."
    ]
    if total_critical:
        cats = f" in: {', '.join(worst_names)}" if worst_names else ""
        parts.append(
            f"There are {total_critical} critical violation(s) requiring immediate attention{cats}."
        )
    if total_high:
        parts.append(f"Additionally, {total_high} high-priority gap(s) were identified.")
    if overall_score < 1.0:
        parts.append(
            "Immediate remediation is recommended: add missing disclosures, specify lawful bases, "
            "and provide clear mechanisms for all data subject rights."
        )
    else:
        parts.append("No violations detected — policy meets all evaluated requirements.")
    return " ".join(parts)


# ═══════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════

class CompliancePipeline:

    def __init__(self, regulation: str = "gdpr"):
        assert regulation in ("gdpr", "pdpa", "both"), "Invalid regulation"
        self.regulation = regulation

    def analyze(self, policy_text: str, url: str = "") -> ComplianceReport:
        """
        Full pipeline: preload → decompose → batch-judge (3 parallel) → report.
        LLM calls: 1 decompose + 3 batch-judge = 4 total.
        After the first call, system prompts are cached — subsequent requests
        only pay for fresh policy text tokens.
        """
        print(f"\n[pipeline] Analysing policy for: {url}")
        print(f"[pipeline] Regulation: {self.regulation.upper()}")
        print(f"[pipeline] Policy length: {len(policy_text.split())} words\n")

        # ── Preload Qdrant (no-op after first call) ──
        _preload_for_regulation(self.regulation)
        preloaded = _preloaded_data[self.regulation]

        # ── Step 1: Decompose ────────────────────────
        print("[1/3] Decomposing policy into category excerpts...")
        category_excerpts = decompose_policy(policy_text)

        # ── Step 2: Batch judge — 3 parallel LLM calls ──
        print("[2/3] Judging violations (3 parallel batched LLM calls)...")

        BATCH_SIZE = 5
        batches = [CATEGORIES[i:i + BATCH_SIZE] for i in range(0, len(CATEGORIES), BATCH_SIZE)]

        def _judge_batch(args) -> dict[int, list[LabelResult]]:
            bi, batch_cats = args
            try:
                result = judge_categories_batch(
                    bi,
                    batch_cats,
                    self.regulation,
                    category_excerpts,
                    preloaded["batch_systems"][bi],
                )
                for cat in batch_cats:
                    print(f"  ✓ {cat['name']}")
                return result
            except Exception as e:
                import traceback
                print(f"  [WARN] Batch {bi} judge failed: {e}")
                traceback.print_exc()
                return {
                    cat["id"]: [
                        LabelResult(
                            label=l["text"], priority=l["priority"],
                            compliant=False, violation=False, missing=True,
                            explanation="Judgment failed. Manual review required.",
                            policy_excerpt="", legal_basis="", recommendation="Review manually.",
                        )
                        for l in cat["labels"]
                    ]
                    for cat in batch_cats
                }

        label_results_map: dict[int, list[LabelResult]] = {}
        with ThreadPoolExecutor(max_workers=len(batches)) as ex:
            for result in ex.map(_judge_batch, enumerate(batches)):
                label_results_map.update(result)

        # ── Aggregate ────────────────────────────────
        print("[3/3] Aggregating results...")
        category_results: list[CategoryResult] = []
        for cat in CATEGORIES:
            label_results = label_results_map.get(cat["id"], [])
            total = len(label_results)
            compliant_count = sum(1 for r in label_results if r.compliant)
            category_results.append(CategoryResult(
                category_id=cat["id"],
                category_name=cat["name"],
                regulation=self.regulation,
                label_results=label_results,
                score=compliant_count / total if total else 0.0,
                critical_violations=sum(
                    1 for r in label_results if (r.violation or r.missing) and r.priority == "Critical"
                ),
                high_violations=sum(
                    1 for r in label_results if (r.violation or r.missing) and r.priority == "High"
                ),
            ))

        total_labels    = sum(len(c.label_results) for c in category_results)
        total_compliant = sum(sum(1 for r in c.label_results if r.compliant) for c in category_results)
        overall_score   = total_compliant / total_labels if total_labels else 0.0
        total_critical  = sum(c.critical_violations for c in category_results)
        total_high      = sum(c.high_violations for c in category_results)
        worst_names     = [
            c.category_name
            for c in sorted(category_results, key=lambda c: c.score)[:3]
            if c.has_violations
        ]

        summary = generate_summary(
            url, self.regulation, overall_score, total_critical, total_high, worst_names
        )

        return ComplianceReport(
            url=url,
            regulation=self.regulation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall_score,
            total_critical_violations=total_critical,
            total_high_violations=total_high,
            category_results=category_results,
            summary=summary,
        )


# ═══════════════════════════════════════════════
# REPORT FORMATTER
# ═══════════════════════════════════════════════

def print_report(report: ComplianceReport):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print(f"\n{'═'*60}", style="bold")
    console.print("  Privacy Policy Compliance Report", style="bold white")
    console.print(f"  URL: {report.url}", style="dim")
    console.print(f"  Regulation: {report.regulation.upper()}", style="dim")
    console.print(f"  Timestamp: {report.timestamp}", style="dim")
    console.print(f"{'═'*60}", style="bold")

    risk_color = {
        "HIGH RISK": "red", "MEDIUM RISK": "yellow", "LOW RISK": "green",
    }.get(report.risk_level, "white")

    console.print(f"\n  Risk Level:    [{risk_color}]{report.risk_level}[/{risk_color}]")
    console.print(f"  Overall Score: {report.overall_score:.0%}")
    console.print(f"  Critical:      {report.total_critical_violations} violation(s)")
    console.print(f"  High:          {report.total_high_violations} violation(s)")
    console.print(f"\n  Summary:\n  {report.summary}\n")

    table = Table(title="Category Breakdown", box=box.ROUNDED)
    table.add_column("Category",  style="cyan", width=28)
    table.add_column("Score",     justify="center", width=8)
    table.add_column("Status",    justify="center", width=12)
    table.add_column("Critical",  justify="center", width=10)
    table.add_column("High",      justify="center", width=8)

    for cat in report.category_results:
        sev_color = {
            "CRITICAL": "red", "HIGH": "yellow",
            "MEDIUM": "orange3", "COMPLIANT": "green",
        }.get(cat.severity, "white")
        table.add_row(
            cat.category_name, f"{cat.score:.0%}",
            f"[{sev_color}]{cat.severity}[/{sev_color}]",
            str(cat.critical_violations), str(cat.high_violations),
        )
    console.print(table)

    for cat in report.category_results:
        if not cat.has_violations:
            continue
        console.print(f"\n[bold]{cat.category_name}[/bold]")
        for r in cat.label_results:
            if r.compliant:
                continue
            status = "[red]VIOLATION[/red]" if r.violation else "[yellow]MISSING[/yellow]"
            console.print(f"  {status} [{r.priority}] {r.label}")
            console.print(f"    → {r.explanation}", style="dim")
            if r.recommendation:
                console.print(f"    ✏ {r.recommendation}", style="italic")


def export_json(report: ComplianceReport) -> dict:
    return {
        "url":                       report.url,
        "regulation":                report.regulation,
        "timestamp":                 report.timestamp,
        "risk_level":                report.risk_level,
        "overall_score":             round(report.overall_score, 3),
        "total_critical_violations": report.total_critical_violations,
        "total_high_violations":     report.total_high_violations,
        "summary":                   report.summary,
        "categories": [
            {
                "id":                 cat.category_id,
                "name":               cat.category_name,
                "score":              round(cat.score, 3),
                "severity":           cat.severity,
                "critical_violations": cat.critical_violations,
                "high_violations":    cat.high_violations,
                "labels": [
                    {
                        "label":          r.label,
                        "priority":       r.priority,
                        "compliant":      r.compliant,
                        "violation":      r.violation,
                        "missing":        r.missing,
                        "explanation":    r.explanation,
                        "policy_excerpt": r.policy_excerpt,
                        "legal_basis":    r.legal_basis,
                        "recommendation": r.recommendation,
                    }
                    for r in cat.label_results
                ],
            }
            for cat in report.category_results
        ],
    }


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse a privacy policy for GDPR/PDPA violations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="URL of the website whose privacy policy to fetch and analyse")
    group.add_argument("--file", help="Local text file containing the privacy policy")
    parser.add_argument("--regulation", choices=["gdpr", "pdpa", "both"], default="gdpr")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.url:
        import httpx
        from bs4 import BeautifulSoup
        print(f"Fetching: {args.url}")
        resp = httpx.get(args.url, follow_redirects=True, timeout=20)
        soup = BeautifulSoup(resp.content, "lxml")
        policy_text = soup.get_text(separator=" ", strip=True)
    else:
        with open(args.file) as f:
            policy_text = f.read()

    pipeline = CompliancePipeline(regulation=args.regulation)
    report   = pipeline.analyze(policy_text=policy_text, url=args.url or args.file)

    if args.json:
        print(json.dumps(export_json(report), indent=2))
    else:
        print_report(report)
