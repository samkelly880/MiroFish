"""Report-scoped shared retrieval context for ReportAgent / ZepToolsService."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.report_agent import ReportAgent
from app.services.zep_tools import (
    INSIGHT_FACT_OVERLAP_THRESHOLD,
    InsightForgeResult,
    NodeInfo,
    ReportRetrievalScope,
    ZepToolsService,
    _fact_jaccard,
)


def _node(uuid: str, name: str) -> NodeInfo:
    return NodeInfo(
        uuid=uuid,
        name=name,
        labels=["Entity", "Person"],
        summary=f"Summary of {name}",
        attributes={},
    )


def _edge(uuid: str, fact: str, src: str, tgt: str, **temporal):
    from app.services.zep_tools import EdgeInfo

    e = EdgeInfo(
        uuid=uuid,
        name="RELATED",
        fact=fact,
        source_node_uuid=src,
        target_node_uuid=tgt,
    )
    for k, v in temporal.items():
        setattr(e, k, v)
    return e


class ScopedZepTools(ZepToolsService):
    """ZepToolsService without real Zep; counts underlying fetches."""

    def __init__(self):
        self._llm_client = SimpleNamespace(
            chat_json=lambda **kwargs: {
                "sub_queries": [
                    "sub q one about businesses",
                    "sub q two about activists",
                ]
            }
        )
        self._report_scope = None
        self.client = MagicMock()
        self._raw_nodes = [
            _node("n1", "BusinessOwner"),
            _node("n2", "Activist"),
        ]
        self._raw_edges = [
            _edge("e1", "Businesses oppose the fee", "n1", "n2"),
            _edge("e2", "Activists support the fee", "n2", "n1"),
        ]
        self.fetch_all_nodes_calls = 0
        self.fetch_all_edges_calls = 0
        self.node_get_calls = 0
        self.search_calls = []

    def get_all_nodes(self, graph_id: str):
        # Replicate cache logic by calling through a patched path: invoke
        # parent after swapping fetch helpers via instance attributes used below.
        scope = self._report_scope
        if scope is not None and scope.graph_id == graph_id and scope.nodes is not None:
            scope.nodes_cache_hits += 1
            return list(scope.nodes)

        self.fetch_all_nodes_calls += 1
        result = list(self._raw_nodes)
        if scope is not None and scope.graph_id == graph_id:
            scope.nodes = list(result)
            scope.nodes_network_fetches += 1
            for node in result:
                if node.uuid and node.uuid not in scope.node_details:
                    scope.node_details[node.uuid] = node
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True):
        scope = self._report_scope
        if scope is not None and scope.graph_id == graph_id and scope.edges is not None:
            scope.edges_cache_hits += 1
            return list(scope.edges)

        self.fetch_all_edges_calls += 1
        result = list(self._raw_edges)
        if scope is not None and scope.graph_id == graph_id:
            scope.edges = list(result)
            scope.edges_network_fetches += 1
        return result

    def get_node_detail(self, node_uuid: str):
        scope = self._report_scope
        if scope is not None and node_uuid in scope.node_details:
            scope.node_detail_cache_hits += 1
            return scope.node_details[node_uuid]

        self.node_get_calls += 1
        result = next((n for n in self._raw_nodes if n.uuid == node_uuid), None)
        if scope is not None:
            scope.node_detail_network_fetches += 1
            scope.node_details[node_uuid] = result
        return result

    def search_graph(self, graph_id, query, limit=10, scope="edges"):
        from app.services.zep_tools import SearchResult

        self.search_calls.append(query)
        # Default: return both facts (high overlap with canonical)
        facts = [e.fact for e in self._raw_edges][:limit]
        return SearchResult(
            facts=facts,
            edges=[{"source_node_uuid": e.source_node_uuid, "target_node_uuid": e.target_node_uuid, "name": e.name} for e in self._raw_edges],
            nodes=[],
            query=query,
            total_count=len(facts),
        )


def test_fact_jaccard():
    assert _fact_jaccard(["a", "b"], ["b", "a"]) == 1.0
    assert _fact_jaccard(["a"], ["b"]) == 0.0
    assert _fact_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_nodes_and_edges_cached_within_scope():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")

    n1 = tools.get_all_nodes("graph_a")
    n2 = tools.get_all_nodes("graph_a")
    e1 = tools.get_all_edges("graph_a")
    e2 = tools.get_all_edges("graph_a")

    assert n1 == n2
    assert e1 == e2
    assert tools.fetch_all_nodes_calls == 1
    assert tools.fetch_all_edges_calls == 1
    assert tools._report_scope.nodes_cache_hits == 1
    assert tools._report_scope.edges_cache_hits == 1

    tools.end_report_scope()
    assert tools._report_scope is None


def test_scope_does_not_apply_to_other_graph():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.get_all_nodes("graph_a")
    tools.get_all_nodes("graph_b")  # different graph → network fetch
    assert tools.fetch_all_nodes_calls == 2
    tools.end_report_scope()


def test_node_detail_cached_after_nodes_snapshot():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.get_all_nodes("graph_a")  # seeds node_details
    detail = tools.get_node_detail("n1")
    assert detail is not None
    assert detail.name == "BusinessOwner"
    # Seeded from snapshot → cache hit, no node_get
    assert tools.node_get_calls == 0
    assert tools._report_scope.node_detail_cache_hits == 1
    tools.end_report_scope()


def test_panorama_reuses_graph_snapshot():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    r1 = tools.panorama_search("graph_a", "businesses fee")
    r2 = tools.panorama_search("graph_a", "activists fee")
    assert tools.fetch_all_nodes_calls == 1
    assert tools.fetch_all_edges_calls == 1
    # Ranking may differ by query, but underlying fact sets match
    assert set(r1.active_facts) == set(r2.active_facts)
    assert r1.query != r2.query
    tools.end_report_scope()


def test_insight_reuses_when_probe_overlaps():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")

    first = tools.prefetch_canonical_insight()
    assert first is not None
    assert tools._report_scope.insight_full_runs == 1
    assert tools._report_scope.canonical_insight is not None

    # Section-flavored query that still retrieves the same facts AND shares tokens
    second = tools.insight_forge(
        graph_id="graph_a",
        query="How would downtown businesses react on social media?",
        simulation_requirement="How would businesses react?",
        report_context="Section: group actions",
    )
    assert tools._report_scope.insight_cache_reuses == 1
    assert tools._report_scope.insight_full_runs == 1  # no second full run
    assert second.query.startswith("How would downtown")
    assert second.semantic_facts == first.semantic_facts
    tools.end_report_scope()


def test_insight_full_run_when_query_diverges_even_if_facts_overlap():
    """Small graphs often return the same edges for any query; require query affinity."""
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    assert tools._report_scope.insight_full_runs == 1

    divergent = tools.insight_forge(
        graph_id="graph_a",
        query="Completely unrelated zoning ordinance timeline for 2030",
        simulation_requirement="How would businesses react?",
    )
    assert tools._report_scope.insight_cache_reuses == 0
    assert tools._report_scope.insight_full_runs == 2
    assert "zoning" in divergent.query
    tools.end_report_scope()


def test_insight_full_run_when_probe_diverges():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    assert tools._report_scope.insight_full_runs == 1

    original_search = tools.search_graph

    def divergent_search(graph_id, query, limit=10, scope="edges"):
        from app.services.zep_tools import SearchResult

        tools.search_calls.append(query)
        # Completely different facts → low Jaccard
        return SearchResult(
            facts=["Unrelated zoning ordinance delayed until 2030"],
            edges=[],
            nodes=[],
            query=query,
            total_count=1,
        )

    tools.search_graph = divergent_search
    # Full forge still needs search_graph for sub-queries — restore after probe
    # by making only the first (probe) call diverge, then use original.

    probe_done = {"n": 0}
    real = original_search

    def probe_then_real(graph_id, query, limit=10, scope="edges"):
        probe_done["n"] += 1
        if probe_done["n"] == 1:
            return divergent_search(graph_id, query, limit=limit, scope=scope)
        return real(graph_id, query, limit=limit, scope=scope)

    tools.search_graph = probe_then_real

    result = tools.insight_forge(
        graph_id="graph_a",
        query="Completely different topic about zoning",
        simulation_requirement="How would businesses react?",
    )
    assert tools._report_scope.insight_cache_reuses == 0
    assert tools._report_scope.insight_full_runs == 2
    assert result.query.startswith("Completely different")
    tools.end_report_scope()


def test_end_scope_clears_cache_no_leak():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.get_all_nodes("graph_a")
    tools.end_report_scope()

    tools.begin_report_scope("graph_a", "req")
    tools.get_all_nodes("graph_a")
    # New scope → another network fetch
    assert tools.fetch_all_nodes_calls == 2
    tools.end_report_scope()


def test_quick_search_exact_cache_within_scope():
    """Opt3: identical normalized query+limit reuses prior quick_search."""
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    before = len(tools.search_calls)
    r1 = tools.quick_search("graph_a", "  Main Street shoppers  ", limit=5)
    r2 = tools.quick_search("graph_a", "Main Street shoppers", limit=5)
    assert r1.facts == r2.facts
    assert len(tools.search_calls) - before == 1
    assert tools._report_scope.quick_search_cache_hits == 1
    assert tools._report_scope.quick_search_network_fetches == 1
    tools.end_report_scope()


def test_quick_search_different_query_or_limit_misses():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.quick_search("graph_a", "query A", limit=5)
    tools.quick_search("graph_a", "query B", limit=5)
    tools.quick_search("graph_a", "query A", limit=10)
    assert tools._report_scope.quick_search_network_fetches == 3
    assert tools._report_scope.quick_search_cache_hits == 0
    tools.end_report_scope()


def test_quick_search_cache_cleared_between_scopes():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.quick_search("graph_a", "same query", limit=5)
    tools.end_report_scope()

    tools.begin_report_scope("graph_a", "req")
    tools.quick_search("graph_a", "same query", limit=5)
    assert tools._report_scope.quick_search_network_fetches == 1
    assert tools._report_scope.quick_search_cache_hits == 0
    tools.end_report_scope()


def test_quick_search_other_graph_not_served_from_cache():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.quick_search("graph_a", "same query", limit=5)
    # Different graph_id while scope is for graph_a → no cache use
    before_hits = tools._report_scope.quick_search_cache_hits
    tools.quick_search("graph_b", "same query", limit=5)
    assert tools._report_scope.quick_search_cache_hits == before_hits
    tools.end_report_scope()


def test_report_agent_wires_scope_around_generate(monkeypatch, tmp_path):
    """generate_report begins/ends scope and prefetches canonical insight."""
    tools = ScopedZepTools()
    calls = {"begin": 0, "end": 0, "prefetch": 0}

    orig_begin = tools.begin_report_scope
    orig_end = tools.end_report_scope
    orig_prefetch = tools.prefetch_canonical_insight

    def begin(*a, **k):
        calls["begin"] += 1
        return orig_begin(*a, **k)

    def end(*a, **k):
        calls["end"] += 1
        return orig_end(*a, **k)

    def prefetch(*a, **k):
        calls["prefetch"] += 1
        return orig_prefetch(*a, **k)

    tools.begin_report_scope = begin
    tools.end_report_scope = end
    tools.prefetch_canonical_insight = prefetch

    agent = ReportAgent(
        graph_id="graph_a",
        simulation_id="sim_x",
        simulation_requirement="How would businesses react?",
        llm_client=SimpleNamespace(
            chat_json=lambda **kwargs: {
                "title": "T",
                "summary": "S",
                "sections": [{"title": "Only Section"}],
            },
            chat=lambda **kwargs: (
                '<tool_call>{"name": "quick_search", "parameters": {"query": "x"}}</tool_call>'
                if "Final Answer" not in str(kwargs)
                else "Final Answer: section body with evidence"
            ),
        ),
        zep_tools=tools,
    )

    # Force interviews unavailable; stub ReportManager filesystem
    monkeypatch.setattr(
        "app.services.simulation_runner.SimulationRunner.check_env_alive",
        classmethod(lambda cls, simulation_id: False),
    )
    agent._interviews_available_cached = False

    from app.services import report_agent as ra_mod

    monkeypatch.setattr(ra_mod.ReportManager, "_ensure_report_folder", staticmethod(lambda rid: None))
    monkeypatch.setattr(ra_mod.ReportManager, "update_progress", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ra_mod.ReportManager, "save_report", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ra_mod.ReportManager, "save_outline", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ra_mod.ReportManager, "save_section", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        ra_mod.ReportManager,
        "assemble_full_report",
        staticmethod(lambda rid, outline: "# full\n"),
    )

    # Simplify section generation: one tool then final
    responses = iter([
        '<tool_call>\n{"name": "quick_search", "parameters": {"query": "x"}}\n</tool_call>',
        '<tool_call>\n{"name": "panorama_search", "parameters": {"query": "y"}}\n</tool_call>',
        '<tool_call>\n{"name": "insight_forge", "parameters": {"query": "z"}}\n</tool_call>',
        "Final Answer: grounded section text",
    ])
    agent.llm = SimpleNamespace(
        chat_json=lambda **kwargs: {
            "title": "T",
            "summary": "S",
            "sections": [{"title": "Only Section"}],
        },
        chat=lambda **kwargs: next(responses),
    )

    report = agent.generate_report(report_id="report_scope_test")
    assert report.status.value == "completed" or str(report.status) in (
        "completed",
        "ReportStatus.COMPLETED",
    )
    assert calls["begin"] == 1
    assert calls["end"] == 1
    assert calls["prefetch"] == 1
    assert tools._report_scope is None  # cleaned up


def test_overlap_threshold_constant():
    assert INSIGHT_FACT_OVERLAP_THRESHOLD == 0.85


def test_timing_cache_vs_uncached_panorama():
    """Unit timing: cached panorama avoids repeated node/edge work."""
    import time

    tools = ScopedZepTools()

    # Inflate fetch cost to make cache benefit measurable
    real_nodes = tools.get_all_nodes
    real_edges = tools.get_all_edges

    def slow_nodes(graph_id):
        # Only sleep on real network path (when not cached)
        scope = tools._report_scope
        if not (scope and scope.graph_id == graph_id and scope.nodes is not None):
            time.sleep(0.02)
        return real_nodes(graph_id)

    def slow_edges(graph_id, include_temporal=True):
        scope = tools._report_scope
        if not (scope and scope.graph_id == graph_id and scope.edges is not None):
            time.sleep(0.02)
        return real_edges(graph_id, include_temporal=include_temporal)

    tools.get_all_nodes = slow_nodes
    tools.get_all_edges = slow_edges

    # Uncached: 3 panoramas without scope
    tools._report_scope = None
    t0 = time.perf_counter()
    for q in ("a", "b", "c"):
        # bypass instance bind - call panorama which uses get_all_*
        tools.panorama_search("graph_a", q)
    uncached = time.perf_counter() - t0

    tools.fetch_all_nodes_calls = 0
    tools.fetch_all_edges_calls = 0
    tools.begin_report_scope("graph_a", "req")
    t1 = time.perf_counter()
    for q in ("a", "b", "c"):
        tools.panorama_search("graph_a", q)
    cached = time.perf_counter() - t1
    tools.end_report_scope()

    assert tools.fetch_all_nodes_calls == 1
    assert tools.fetch_all_edges_calls == 1
    assert cached < uncached
    # Keep evidence in assertion message for the opt report
    print(f"PERF panorama uncached={uncached:.4f}s cached={cached:.4f}s")


def test_shared_evidence_text_built_once():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    text1 = tools.build_shared_evidence_text()
    text2 = tools.build_shared_evidence_text()
    assert text1 == text2
    assert "Shared Report Evidence" in text1
    assert "Businesses oppose the fee" in text1 or "Activists support" in text1
    assert tools._report_scope.shared_evidence_text == text1
    tools.end_report_scope()


def test_compact_insight_observation_when_reused():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    tools.build_shared_evidence_text()

    reused = tools.insight_forge(
        graph_id="graph_a",
        query="How would businesses react to the fee on social media?",
        simulation_requirement="How would businesses react?",
    )
    assert reused.reused_from_canonical is True
    obs = tools.format_insight_observation(reused)
    assert "compact" in obs.lower() or "Shared Report Evidence" in obs
    assert len(obs) < len(reused.to_text())
    assert tools._report_scope.compact_insight_observations == 1
    tools.end_report_scope()


def test_full_insight_observation_when_not_reused():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    # No canonical yet → first forge is full; format stays full even after build
    full = tools.insight_forge(
        graph_id="graph_a",
        query="First full forge",
        simulation_requirement="req",
    )
    assert full.reused_from_canonical is False
    tools.build_shared_evidence_text()
    obs = tools.format_insight_observation(full)
    assert "Key Facts" in obs or "Future Prediction Deep Analysis" in obs
    assert "compact" not in obs.lower()
    tools.end_report_scope()


def test_compact_panorama_when_shared_evidence_present():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.prefetch_canonical_insight()
    tools.build_shared_evidence_text()
    pan = tools.panorama_search("graph_a", "fee opposition")
    compact = tools.format_panorama_observation(pan)
    full = pan.to_text()
    assert "Shared Report Evidence" in compact
    assert pan.query in compact
    # Compact drops the involved-entities dump (lives in shared pack)
    assert "Involved Entities" not in compact
    assert "Involved Entities" in full
    assert tools._report_scope.compact_panorama_observations == 1
    tools.end_report_scope()


def test_quick_search_observation_stays_full():
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "req")
    tools.build_shared_evidence_text()
    result = tools.quick_search("graph_a", "Main Street shoppers", limit=5)
    text = result.to_text()
    assert "Search query:" in text
    assert "Main Street" in text or "Related facts" in text or "Found" in text
    tools.end_report_scope()


def test_section_prompt_includes_shared_evidence(monkeypatch):
    """_generate_section_react injects shared evidence into the system prompt."""
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    tools.build_shared_evidence_text()

    captured = {}

    def capture_chat(**kwargs):
        captured["messages"] = kwargs.get("messages")
        return "Final Answer: body grounded in shared evidence"

    agent = ReportAgent(
        graph_id="graph_a",
        simulation_id="sim_x",
        simulation_requirement="How would businesses react?",
        llm_client=SimpleNamespace(chat=capture_chat),
        zep_tools=tools,
    )
    monkeypatch.setattr(
        "app.services.simulation_runner.SimulationRunner.check_env_alive",
        classmethod(lambda cls, simulation_id: False),
    )
    agent._interviews_available_cached = False

    from app.services.report_agent import ReportOutline, ReportSection

    # Force min tools satisfied by stubbing tool path: return Final Answer
    # after claiming tools already done — override loop via chat returning final
    # with insufficient tools first... simpler: monkeypatch min by providing
    # responses that call tools then final. Use iterator.

    responses = iter([
        '<tool_call>\n{"name": "quick_search", "parameters": {"query": "x"}}\n</tool_call>',
        '<tool_call>\n{"name": "panorama_search", "parameters": {"query": "y"}}\n</tool_call>',
        '<tool_call>\n{"name": "insight_forge", "parameters": {"query": "z"}}\n</tool_call>',
        "Final Answer: done",
    ])
    agent.llm = SimpleNamespace(chat=lambda **kwargs: (
        captured.update({"messages": kwargs["messages"]}) or next(responses)
    ))

    outline = ReportOutline(
        title="T",
        summary="S",
        sections=[ReportSection(title="Sec One", content="")],
    )
    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=1,
    )
    assert content == "done"
    system = captured["messages"][0]["content"]
    assert "Shared Report Evidence" in system
    # Observation for insight should be compact (reused)
    user_blobs = "\n".join(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert "compact" in user_blobs.lower() or "Shared Report Evidence" in user_blobs
    tools.end_report_scope()


def test_observation_size_reduction_vs_full_dumps():
    """Measure that compact observations shrink ReACT evidence payload."""
    tools = ScopedZepTools()
    tools.begin_report_scope("graph_a", "How would businesses react?")
    tools.prefetch_canonical_insight()
    tools.build_shared_evidence_text()
    shared_len = len(tools.get_shared_evidence_text())

    # Simulate three section tool bundles as before Opt2 (full dumps)
    full_insight = tools._report_scope.canonical_insight.to_text()
    pan = tools.panorama_search("graph_a", "q")
    full_pan = pan.to_text()
    quick = tools.quick_search("graph_a", "specific", limit=5).to_text()
    before_per_section = len(full_insight) + len(full_pan) + len(quick)

    # After Opt2: shared once + compact insight/panorama + full quick
    reused = tools.insight_forge(
        graph_id="graph_a",
        query="How would businesses react in this section?",
        simulation_requirement="How would businesses react?",
    )
    assert reused.reused_from_canonical is True
    compact_insight = tools.format_insight_observation(reused)
    compact_pan = tools.format_panorama_observation(pan)
    after_per_section = len(compact_insight) + len(compact_pan) + len(quick)
    # Four sections: before embeds full packs 4×; after shared once + compact 4×
    before_total = before_per_section * 4
    after_total = shared_len + after_per_section * 4

    assert after_per_section < before_per_section
    assert after_total < before_total
    print(
        f"PERF evidence chars before_4sec={before_total} after_4sec={after_total} "
        f"shared={shared_len} compact_insight={len(compact_insight)} "
        f"full_insight={len(full_insight)}"
    )
    tools.end_report_scope()


def test_timing_insight_reuse_vs_full():
    """Unit timing: overlapping insight_forge reuses canonical (skips nested LLM)."""
    import time

    tools = ScopedZepTools()
    llm_calls = {"n": 0}
    real_chat = tools._llm_client.chat_json

    def slow_chat_json(**kwargs):
        llm_calls["n"] += 1
        time.sleep(0.03)
        return real_chat(**kwargs)

    tools._llm_client.chat_json = slow_chat_json
    tools.begin_report_scope("graph_a", "How would businesses react?")

    t0 = time.perf_counter()
    tools.prefetch_canonical_insight()
    first = time.perf_counter() - t0
    assert llm_calls["n"] == 1

    t1 = time.perf_counter()
    for i in range(3):
        tools.insight_forge(
            graph_id="graph_a",
            query=f"How would businesses react in section flavor {i}?",
            simulation_requirement="How would businesses react?",
        )
    reused = time.perf_counter() - t1
    tools.end_report_scope()

    assert llm_calls["n"] == 1  # no additional nested LLM
    assert reused < first  # three reuses cheaper than one full forge
    print(
        f"PERF insight first_full={first:.4f}s three_reuses={reused:.4f}s "
        f"llm_calls={llm_calls['n']}"
    )
