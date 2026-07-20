import unittest

from src.debate import build_debate_prompt
from src.evidence import format_credible_sources_for_prompt, relevant_credible_sources


class PersonaBehaviorTest(unittest.TestCase):
    def test_minimal_harm_reduction_prompt_includes_persona_and_orientation(self):
        persona_id = "Clinician (Minimal Harm Reduction)"
        prompt = build_debate_prompt(
            persona_id=persona_id,
            debate_topic="Should the state fund supervised consumption sites?",
            transcript=[{"role": "user", "content": "I support narrowing interventions to naloxone and wound care."}],
            evidence_items=[],
            retrieved_source_hits=[],
        )

        self.assertIn(persona_id, prompt)
        self.assertIn("Minimal Harm Reduction", prompt)
        self.assertIn("SOURCE STRICT MODE: ON", prompt)

    def test_minimal_harm_reduction_source_ranking_prefers_high_quality_public_health_sources(self):
        persona_id = "Clinician (Minimal Harm Reduction)"
        sources = relevant_credible_sources(persona_id)
        self.assertGreater(len(sources), 0)

        top_source = sources[0]
        self.assertIn("Minimal Harm Reduction", top_source["best_for"])
        self.assertIn("public health", top_source["specialty"].lower())

    def test_format_credible_sources_for_prompt_returns_top_sources(self):
        persona_id = "Clinician (Minimal Harm Reduction)"
        formatted = format_credible_sources_for_prompt(persona_id, limit=3)

        self.assertTrue(formatted.startswith("1."))
        self.assertIn("American Journal of Public Health", formatted)
        self.assertIn("Minimal Harm Reduction", formatted)

    def test_community_advocate_minimal_harm_reduction_prompt_includes_role_and_orientation(self):
        persona_id = "Community Advocate (Minimal Harm Reduction)"
        prompt = build_debate_prompt(
            persona_id=persona_id,
            debate_topic="Should the state expand naloxone distribution in rural counties?",
            transcript=[{"role": "user", "content": "I want a focused, low-barrier approach that still protects public health."}],
            evidence_items=[],
            retrieved_source_hits=[],
        )

        self.assertIn("Community Advocate", prompt)
        self.assertIn("Minimal Harm Reduction", prompt)
        self.assertIn("You are a community advocate", prompt)


if __name__ == "__main__":
    unittest.main()
