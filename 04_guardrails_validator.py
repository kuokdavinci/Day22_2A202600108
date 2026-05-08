"""
Step 4 — Guardrails AI Validators
====================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone
     numbers, SSNs, and credit card numbers
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
  4. Run a full demo with 6 PII cases and 5 JSON cases

DELIVERABLE: All test cases pass (PII redacted, JSON repaired)
"""

import re
import json
from typing import Any, Dict, Optional, Callable

# --- Import Guardrails AI components ---
from guardrails import Guard, OnFailAction
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
    ValidationResult,
)


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).

    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def __init__(self, on_fail: Optional[Callable] = None, **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not isinstance(value, str):
            return FailResult(error_message="Input must be a string.", fix_value=None)

        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Redacted {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return FailResult(
                error_message=f"Found PII: {', '.join([p[0] for p in found_pii])}",
                fix_value=redacted_text,
            )
        return PassResult()

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        return self._validate(value, metadata)


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.

    Common repairs:
      - Strip markdown code fences (``` or ```json)
      - Replace single quotes with double quotes
      - Remove trailing commas before } or ]
      - Re-serialize with json.dumps for consistent formatting
    """

    def __init__(self, on_fail: Optional[Callable] = None, **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.
        """
        text = text.strip()

        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Single quotes -> double quotes
        text = text.replace("'", '"')

        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if not isinstance(value, str):
            return FailResult(error_message="Input must be a string.", fix_value=None)

        # 1. Try standard parse
        try:
            parsed  = json.loads(value)
            repaired = json.dumps(parsed, indent=2)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # 2. Try repair
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            repaired      = json.dumps(parsed, indent=2)
            print(f"  🔧 JSON repaired successfully")
            return FailResult(
                error_message="Invalid JSON, repaired successfully",
                fix_value=repaired,
            )
        except json.JSONDecodeError as e:
            # 3. Fallback error JSON when repair fails
            fallback_json = json.dumps({"error": "unparseable_json", "raw": value})
            print(f"  ❌ JSON repair failed, returning fallback JSON")
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=fallback_json,
            )

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        return self._validate(value, metadata)


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts.
    """
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings.
    """
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail (Repaired/Fallback)"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:60]}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    main()
