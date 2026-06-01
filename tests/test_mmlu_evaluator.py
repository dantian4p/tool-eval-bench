"""Tests for MMLU evaluator — answer extraction and comparison."""

from __future__ import annotations

from tool_eval_bench.plugins.mmlu.dataset import (
    CATEGORIES,
    SUBJECT_CATEGORIES,
    MMLUItem,
)
from tool_eval_bench.plugins.mmlu.evaluator import (
    evaluate_answer,
    extract_answer,
)
from tool_eval_bench.plugins.mmlu.prompts import (
    build_messages,
)

# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    """Test multiple-choice letter extraction."""

    def test_exact_single_letter(self):
        assert extract_answer("B") == ("B", "exact")

    def test_exact_lowercase(self):
        assert extract_answer("c") == ("C", "exact")

    def test_exact_with_period(self):
        assert extract_answer("A.") == ("A", "exact")

    def test_exact_with_paren(self):
        assert extract_answer("(D)") == ("D", "exact")

    def test_the_answer_is(self):
        assert extract_answer("The answer is B") == ("B", "answer_pattern")

    def test_answer_is_with_colon(self):
        assert extract_answer("After analysis, the answer is: C") == ("C", "answer_pattern")

    def test_answer_colon(self):
        assert extract_answer("Answer: D") == ("D", "answer_pattern")

    def test_answer_colon_with_paren(self):
        assert extract_answer("Answer: (A)") == ("A", "answer_pattern")

    def test_first_standalone_letter(self):
        assert extract_answer("I think the best choice here is definitely B because of the reasons stated.") == ("B", "first_letter")

    def test_empty_string(self):
        assert extract_answer("") == (None, "none")

    def test_no_valid_letter(self):
        assert extract_answer("I'm not sure about this question") == (None, "none")

    def test_whitespace_only(self):
        assert extract_answer("   ") == (None, "none")

    def test_multiline_response(self):
        resp = "Let me think step by step.\nThe answer is A."
        assert extract_answer(resp) == ("A", "answer_pattern")

    def test_long_explanation_with_answer(self):
        resp = (
            "This is a complex problem. First, we need to consider X. "
            "Then, applying theorem Y, we get Z. Therefore, the answer is D."
        )
        assert extract_answer(resp) == ("D", "answer_pattern")


# ---------------------------------------------------------------------------
# evaluate_answer
# ---------------------------------------------------------------------------

class TestEvaluateAnswer:
    """Test answer evaluation against ground truth."""

    def test_correct_answer(self):
        result = evaluate_answer("B", 1)  # B = index 1
        assert result.correct is True
        assert result.extracted_answer == "B"
        assert result.ground_truth_letter == "B"

    def test_incorrect_answer(self):
        result = evaluate_answer("A", 2)  # Expected C
        assert result.correct is False
        assert result.extracted_answer == "A"
        assert result.ground_truth_letter == "C"

    def test_no_answer_extracted(self):
        result = evaluate_answer("I don't know", 0)
        assert result.correct is False
        assert result.extracted_answer is None

    def test_all_indices(self):
        for i, letter in enumerate("ABCD"):
            result = evaluate_answer(letter, i)
            assert result.correct is True
            assert result.ground_truth_letter == letter


# ---------------------------------------------------------------------------
# MMLUItem
# ---------------------------------------------------------------------------

class TestMMLUItem:
    """Test MMLUItem properties."""

    def test_answer_letter(self):
        item = MMLUItem(0, "Q?", "math", ["a", "b", "c", "d"], 2)
        assert item.answer_letter == "C"

    def test_category_known(self):
        item = MMLUItem(0, "Q?", "abstract_algebra", ["a", "b", "c", "d"], 0)
        assert item.category == "STEM"

    def test_category_unknown(self):
        item = MMLUItem(0, "Q?", "unknown_subject", ["a", "b", "c", "d"], 0)
        assert item.category == "Other"

    def test_to_dict(self):
        item = MMLUItem(5, "What is?", "virology", ["a", "b", "c", "d"], 3)
        d = item.to_dict()
        assert d["index"] == 5
        assert d["answer_letter"] == "D"
        assert d["subject"] == "virology"


# ---------------------------------------------------------------------------
# Subject categories
# ---------------------------------------------------------------------------

class TestSubjectCategories:
    """Test subject → category mapping."""

    def test_all_categories_present(self):
        assert "STEM" in CATEGORIES
        assert "Humanities" in CATEGORIES
        assert "Social Sciences" in CATEGORIES
        assert "Other" in CATEGORIES

    def test_57_subjects(self):
        assert len(SUBJECT_CATEGORIES) == 57

    def test_stem_subjects(self):
        stem = [s for s, c in SUBJECT_CATEGORIES.items() if c == "STEM"]
        assert "abstract_algebra" in stem
        assert "machine_learning" in stem
        assert len(stem) == 19

    def test_humanities_subjects(self):
        humanities = [s for s, c in SUBJECT_CATEGORIES.items() if c == "Humanities"]
        assert "philosophy" in humanities
        assert "world_religions" in humanities

    def test_social_sciences(self):
        ss = [s for s, c in SUBJECT_CATEGORIES.items() if c == "Social Sciences"]
        assert "sociology" in ss
        assert "econometrics" in ss


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildMessages:
    """Test prompt construction."""

    def test_zero_shot(self):
        item = MMLUItem(0, "What is 2+2?", "elementary_mathematics",
                        ["3", "4", "5", "6"], 1)
        msgs = build_messages(item, n_shots=0)
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "A." in msgs[1]["content"]
        assert "B." in msgs[1]["content"]

    def test_few_shot(self):
        item = MMLUItem(0, "Q?", "abstract_algebra", ["a", "b", "c", "d"], 0)
        examples = [
            MMLUItem(i, f"Ex {i}?", "abstract_algebra", ["x", "y", "z", "w"], i % 4)
            for i in range(5)
        ]
        msgs = build_messages(item, few_shot_examples=examples, n_shots=3)
        # system + 3*(user+assistant) + user = 8
        assert len(msgs) == 8
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_system_prompt_includes_subject(self):
        item = MMLUItem(0, "Q?", "machine_learning", ["a", "b", "c", "d"], 0)
        msgs = build_messages(item)
        assert "Machine Learning" in msgs[0]["content"]

    def test_no_few_shot_examples(self):
        item = MMLUItem(0, "Q?", "virology", ["a", "b", "c", "d"], 0)
        msgs = build_messages(item, few_shot_examples=None, n_shots=5)
        assert len(msgs) == 2  # system + user only


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

class TestRating:
    """Test rating function."""

    def test_excellent(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy
        assert "Excellent" in _rating_for_accuracy(90)

    def test_good(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy
        assert "Good" in _rating_for_accuracy(75)

    def test_poor(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy
        assert "Poor" in _rating_for_accuracy(20)
