"""Opt4: previous_sections digests shrink prompts while keeping anti-repetition signals."""

from pathlib import Path

from app.services.report_agent import ReportAgent


SAMPLE_SECTION = """## The Future That Forms Around the Vote

In this rehearsal of the future, the downtown congestion fee does not wait for the gavel. **City Council is set to vote next week**, and social media organizes around that date.

> "The City Council will vote next week on a downtown congestion fee."

**The climate-and-transit camp that forms**

Climate activists support the fee.

> "Climate activists support the downtown congestion fee, viewing the reduction in car trips as a climate gain."

> "Climate activists suggest voting yes on the fee and putting the money into buses."

**The Main Street camp that forms**

Business owners warn about empty streets.

> "Small business owners state that a congestion fee that empties Main Street is not a win."

- Climate activists: vote yes
- Small-business owners: empties Main Street
- Commuters: unfair toll
"""


def test_short_section_returned_unchanged():
    short = "## Title\n\nShort body."
    assert ReportAgent._digest_previous_section(short, max_chars=1200) == short.strip()


def test_digest_preserves_title_topics_quotes():
    # Force digestion with a low budget so structure extraction runs
    long_section = SAMPLE_SECTION + ("\n\nAdditional paragraph expanding the rehearsal narrative.\n" * 30)
    digest = ReportAgent._digest_previous_section(long_section, max_chars=900)
    assert "## The Future That Forms Around the Vote" in digest
    assert "Topics covered:" in digest
    assert "climate-and-transit camp" in digest.lower() or "City Council" in digest
    assert "Key quotes already used:" in digest
    assert "City Council will vote next week" in digest
    assert len(digest) < len(long_section)
    assert len(digest) <= 900


def test_digest_respects_max_chars():
    digest = ReportAgent._digest_previous_section(SAMPLE_SECTION * 5, max_chars=500)
    assert len(digest) <= 500


def test_format_previous_uses_larger_budget_for_latest():
    older = SAMPLE_SECTION
    newer = SAMPLE_SECTION.replace("Future That Forms", "How Each Group Acts")
    # Make them long enough to force digests
    older = older + ("\n\nMore filler about camps.\n" * 40)
    newer = newer + ("\n\nMore filler about tactics.\n" * 40)
    formatted = ReportAgent._format_previous_sections_context([older, newer])
    assert "How Each Group Acts" in formatted
    assert "Future That Forms" in formatted
    # Latest budget 1600, older 900 → total well under 2*4000
    assert len(formatted) < 1600 + 900 + 50
    assert len(formatted) < len(older) + len(newer)


def test_format_empty_previous():
    assert ReportAgent._format_previous_sections_context([]) == "(This is the first section)"


def test_real_report_sections_digest_size_reduction():
    """Use audited report sections as size evidence when artifacts exist."""
    base = Path("uploads/reports/report_7e5a3a86f9bf")
    if not base.exists():
        base = Path("backend/uploads/reports/report_7e5a3a86f9bf")
    if not base.exists():
        # Skip gracefully if artifacts not present in this environment
        return

    sections = []
    for i in range(1, 5):
        sections.append((base / f"section_{i:02d}.md").read_text())

    # Old behavior: up to 4000 each
    old_parts = []
    for sec in sections[:3]:  # what section 4 would see
        old_parts.append(sec[:4000] + ("..." if len(sec) > 4000 else ""))
    old_total = len("\n\n---\n\n".join(old_parts))

    new_ctx = ReportAgent._format_previous_sections_context(sections[:3])
    assert len(new_ctx) < old_total
    # Still carries anti-rep signals from early sections
    assert "City Council" in new_ctx or "congestion" in new_ctx.lower()
    print(f"PERF previous_sections sec4 old={old_total} new={len(new_ctx)}")
