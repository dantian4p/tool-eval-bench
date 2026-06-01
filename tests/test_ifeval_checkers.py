"""Tests for IFEval constraint checkers and evaluator."""

from __future__ import annotations

import pytest

from tool_eval_bench.plugins.ifeval.checkers import (
    available_checkers,
    check_instruction,
)
from tool_eval_bench.plugins.ifeval.evaluator import (
    evaluate_prompt,
)

# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------

class TestLengthConstraints:
    """Test word, sentence, and paragraph count checks."""

    def test_number_words_at_least_pass(self):
        response = " ".join(["word"] * 100)
        assert check_instruction("length_constraints:number_words", response,
                                {"num_words": 50, "relation": "at least"})

    def test_number_words_at_least_fail(self):
        response = "just three words"
        assert not check_instruction("length_constraints:number_words", response,
                                    {"num_words": 50, "relation": "at least"})

    def test_number_words_at_most(self):
        response = "only five words here now"
        assert check_instruction("length_constraints:number_words", response,
                                {"num_words": 10, "relation": "at most"})

    def test_number_words_exactly(self):
        response = "one two three"
        assert check_instruction("length_constraints:number_words", response,
                                {"num_words": 3, "relation": "exactly"})

    def test_number_sentences(self):
        response = "First sentence. Second sentence. Third sentence."
        assert check_instruction("length_constraints:number_sentences", response,
                                {"num_sentences": 3, "relation": "at least"})

    def test_number_paragraphs(self):
        response = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        assert check_instruction("length_constraints:number_paragraphs", response,
                                {"num_paragraphs": 3, "relation": "at least"})

    def test_number_paragraphs_fail(self):
        response = "Just one paragraph."
        assert not check_instruction("length_constraints:number_paragraphs", response,
                                    {"num_paragraphs": 3, "relation": "at least"})

    def test_nth_paragraph_first_word(self):
        response = "Hello world.\n\nWelcome back.\n\nGoodbye now."
        assert check_instruction("length_constraints:nth_paragraph_first_word", response,
                                {"nth_paragraph": 2, "first_word": "welcome"})


# ---------------------------------------------------------------------------
# Keyword constraints
# ---------------------------------------------------------------------------

class TestKeywordConstraints:
    """Test keyword existence, frequency, and forbidden words."""

    def test_keywords_existence_pass(self):
        response = "The cat sat on the mat"
        assert check_instruction("keywords:existence", response,
                                {"keywords": ["cat", "mat"]})

    def test_keywords_existence_fail(self):
        response = "The dog ran in the park"
        assert not check_instruction("keywords:existence", response,
                                    {"keywords": ["cat", "mat"]})

    def test_keywords_frequency(self):
        response = "hello hello hello world"
        assert check_instruction("keywords:frequency", response,
                                {"keyword": "hello", "frequency": 3, "relation": "at least"})

    def test_keywords_frequency_fail(self):
        response = "hello world"
        assert not check_instruction("keywords:frequency", response,
                                    {"keyword": "hello", "frequency": 3, "relation": "at least"})

    def test_forbidden_words_pass(self):
        response = "The cat sat on the mat"
        assert check_instruction("keywords:forbidden_words", response,
                                {"forbidden_words": ["dog", "bird"]})

    def test_forbidden_words_fail(self):
        response = "The dog ran in the park"
        assert not check_instruction("keywords:forbidden_words", response,
                                    {"forbidden_words": ["dog", "bird"]})

    def test_letter_frequency(self):
        response = "aaaaabbb"
        assert check_instruction("keywords:letter_frequency", response,
                                {"letter": "a", "let_frequency": 5, "let_relation": "at least"})

    def test_letter_frequency_fail(self):
        response = "aab"
        assert not check_instruction("keywords:letter_frequency", response,
                                    {"letter": "a", "let_frequency": 5, "let_relation": "at least"})


# ---------------------------------------------------------------------------
# Format constraints
# ---------------------------------------------------------------------------

class TestFormatConstraints:
    """Test format-related checks."""

    def test_highlighted_sections(self):
        response = "Here is *section one* and *section two* and *section three*."
        assert check_instruction("detectable_format:number_highlighted_sections",
                                response, {"num_highlights": 3})

    def test_highlighted_sections_fail(self):
        response = "Here is *section one* only."
        assert not check_instruction("detectable_format:number_highlighted_sections",
                                    response, {"num_highlights": 3})

    def test_bullet_lists(self):
        response = "Items:\n- item one\n- item two\n- item three"
        assert check_instruction("detectable_format:number_bullet_lists",
                                response, {"num_bullets": 3})

    def test_placeholders(self):
        response = "Name: [name], Address: [address], Phone: [phone]"
        assert check_instruction("detectable_format:number_placeholders",
                                response, {"num_placeholders": 3})

    def test_json_format_valid(self):
        response = '{"key": "value", "num": 42}'
        assert check_instruction("detectable_format:json_format", response, {})

    def test_json_format_with_fences(self):
        response = '```json\n{"key": "value"}\n```'
        assert check_instruction("detectable_format:json_format", response, {})

    def test_json_format_invalid(self):
        response = "This is not JSON at all"
        assert not check_instruction("detectable_format:json_format", response, {})

    def test_title(self):
        response = "# My Title\n\nContent here."
        assert check_instruction("detectable_format:title", response, {})

    def test_title_without_hash(self):
        response = "My Title\n\nContent here."
        assert check_instruction("detectable_format:title", response, {})

    def test_multiple_sections(self):
        response = "# Section 1\nContent.\n# Section 2\nMore content.\n# Section 3\nEnd."
        assert check_instruction("detectable_format:multiple_sections",
                                response, {"num_sections": 3})


# ---------------------------------------------------------------------------
# Punctuation constraints
# ---------------------------------------------------------------------------

class TestPunctuationConstraints:
    """Test punctuation checks."""

    def test_no_comma_pass(self):
        response = "This sentence has no commas at all."
        assert check_instruction("punctuation:no_comma", response, {})

    def test_no_comma_fail(self):
        response = "This sentence has commas, right here."
        assert not check_instruction("punctuation:no_comma", response, {})


# ---------------------------------------------------------------------------
# Start/end constraints
# ---------------------------------------------------------------------------

class TestStartEndConstraints:
    """Test start/end phrase checks."""

    def test_end_phrase_pass(self):
        response = "Some text. Is there anything else I can help with?"
        assert check_instruction("startend:end_checker", response,
                                {"end_phrase": "Is there anything else I can help with?"})

    def test_end_phrase_fail(self):
        response = "Some text here."
        assert not check_instruction("startend:end_checker", response,
                                    {"end_phrase": "The end."})

    def test_quotation_double(self):
        response = '"This is a quoted response"'
        assert check_instruction("startend:quotation", response, {})

    def test_quotation_single(self):
        response = "'This is a quoted response'"
        assert check_instruction("startend:quotation", response, {})

    def test_quotation_fail(self):
        response = "This is not quoted"
        assert not check_instruction("startend:quotation", response, {})


# ---------------------------------------------------------------------------
# Case constraints
# ---------------------------------------------------------------------------

class TestCaseConstraints:
    """Test case transformation checks."""

    def test_uppercase_pass(self):
        response = "THIS IS ALL UPPERCASE 123"
        assert check_instruction("change_case:english_uppercase", response, {})

    def test_uppercase_fail(self):
        response = "This has lowercase"
        assert not check_instruction("change_case:english_uppercase", response, {})

    def test_lowercase_pass(self):
        response = "this is all lowercase 123"
        assert check_instruction("change_case:english_lowercase", response, {})

    def test_lowercase_fail(self):
        response = "This Has Uppercase"
        assert not check_instruction("change_case:english_lowercase", response, {})

    def test_capitalize_pass(self):
        response = "Every Word Is Capitalized"
        assert check_instruction("change_case:english_capital", response, {})

    def test_capitalize_fail(self):
        response = "not every word is capitalized"
        assert not check_instruction("change_case:english_capital", response, {})


# ---------------------------------------------------------------------------
# Combination / misc constraints
# ---------------------------------------------------------------------------

class TestCombinationConstraints:
    """Test combination and misc checks."""

    def test_repeat_prompt_pass(self):
        prompt = "Tell me a story about a cat."
        response = f"You asked: {prompt} Here is my story..."
        assert check_instruction("combination:repeat_prompt", response,
                                {"prompt_to_repeat": prompt})

    def test_repeat_prompt_fail(self):
        response = "Here is a story about a dog."
        assert not check_instruction("combination:repeat_prompt", response,
                                    {"prompt_to_repeat": "Tell me a story about a cat."})

    def test_two_responses(self):
        response = "Response one here.\n******\nResponse two here."
        assert check_instruction("combination:two_responses", response, {})

    def test_two_responses_fail(self):
        response = "Just one response."
        assert not check_instruction("combination:two_responses", response, {})

    def test_postscript_pass(self):
        response = "Main content here.\n\nP.S. Don't forget to check."
        assert check_instruction("detectable_content:postscript", response, {})

    def test_postscript_fail(self):
        response = "Main content only."
        assert not check_instruction("detectable_content:postscript", response, {})

    def test_language_english(self):
        response = "This is a response in English."
        assert check_instruction("language:response_language", response,
                                {"language": "en"})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestCheckerRegistry:
    """Test the checker registry."""

    def test_all_checkers_registered(self):
        checkers = available_checkers()
        assert len(checkers) >= 20  # We have ~25 checkers

    def test_unknown_checker_raises(self):
        with pytest.raises(KeyError):
            check_instruction("nonexistent:checker", "test", {})


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluatePrompt:
    """Test prompt-level evaluation."""

    def test_all_pass(self):
        response = "THIS IS ALL UPPERCASE."
        result = evaluate_prompt(
            response,
            ["change_case:english_uppercase"],
            [{}],
        )
        assert result.prompt_pass is True
        assert result.instructions_passed == 1
        assert result.instructions_total == 1

    def test_partial_pass(self):
        response = "this has no commas but is lowercase."
        result = evaluate_prompt(
            response,
            ["punctuation:no_comma", "change_case:english_uppercase"],
            [{}, {}],
        )
        assert result.prompt_pass is False  # Not all passed
        assert result.instructions_passed == 1  # no_comma passed
        assert result.instructions_total == 2

    def test_all_fail(self):
        response = "this has commas, and is lowercase."
        result = evaluate_prompt(
            response,
            ["punctuation:no_comma", "change_case:english_uppercase"],
            [{}, {}],
        )
        assert result.prompt_pass is False
        assert result.instructions_passed == 0

    def test_unknown_instruction_passes(self):
        """Unknown instructions should pass (not penalize)."""
        response = "Any response."
        result = evaluate_prompt(
            response,
            ["unknown:instruction"],
            [{}],
        )
        assert result.prompt_pass is True
        assert result.instruction_results[0].error is not None

    def test_empty_instructions(self):
        result = evaluate_prompt("Any response.", [], [])
        assert result.prompt_pass is True
        assert result.instructions_total == 0

    def test_none_kwargs_filtered(self):
        """None values in kwargs should be filtered out."""
        response = " ".join(["word"] * 100)
        result = evaluate_prompt(
            response,
            ["length_constraints:number_words"],
            [{"num_words": 50, "relation": "at least", "num_sentences": None}],
        )
        assert result.prompt_pass is True


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

class TestIFEvalRating:
    """Test IFEval rating function."""

    def test_excellent(self):
        from tool_eval_bench.plugins.ifeval.plugin import _rating_for_accuracy
        assert "Excellent" in _rating_for_accuracy(90)

    def test_poor(self):
        from tool_eval_bench.plugins.ifeval.plugin import _rating_for_accuracy
        assert "Poor" in _rating_for_accuracy(20)
