from app.cli.verdict import generate_verdict, normalize_verdict


class FakeClient:
    def chat_json(self, **kwargs):
        return {
            "prediction": "Opinion polarizes quickly.",
            "confidence": 0.7,
            "key_dynamics": ["amplification"],
            "signals": ["repost spikes"],
            "insufficient_data": False,
        }


def test_generate_verdict_short_report_insufficient():
    verdict = generate_verdict("too short", "req")
    assert verdict["insufficient_data"] is True
    assert verdict["confidence"] == 0.0


def test_generate_verdict_with_client():
    report = "# Report\n\n" + ("Evidence line.\n" * 20)
    verdict = generate_verdict(report, "How does X unfold?", llm=FakeClient())
    assert verdict["prediction"].startswith("Opinion")
    assert 0.0 <= verdict["confidence"] <= 1.0
    assert verdict["insufficient_data"] is False


def test_normalize_clamps_confidence():
    assert normalize_verdict({"prediction": "x", "confidence": 1.5, "key_dynamics": [], "signals": [], "insufficient_data": False})["confidence"] == 1.0
