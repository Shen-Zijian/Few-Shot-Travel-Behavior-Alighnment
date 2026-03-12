"""
Prompt templates for TRAIL LLM calls.

Uses Jinja2-style string formatting (not actual Jinja2 rendering,
keeping dependency minimal for the templates module itself).
"""

from trail.data.schema import MODE_LABELS

# Mode options string used in all prompts
MODE_OPTIONS_STR = "\n".join([
    f"  {code}: {label}" for code, label in sorted(MODE_LABELS.items())
])


# ============================================================
# TRAIL Policy Core Prompt (mode choice with retrieval)
# ============================================================

TRAIL_SYSTEM_PROMPT = """You are an expert in Hong Kong travel behavior analysis with access to historical travel survey data.

Your task is to predict the main transport mode for a traveler based on:
1. Their demographic profile and household characteristics
2. The trip context (purpose, time, origin/destination)
3. Historical behavioral patterns from similar travelers
4. Recent travel observations from the current survey period

Available transport modes:
{mode_options}

IMPORTANT: You MUST respond ONLY with valid JSON using exactly this structure:
{{"predicted_mode": <integer mode code>, "confidence": <float between 0 and 1>, "reasoning_brief": "<one sentence explanation>"}}

Do NOT include any text outside the JSON object."""


TRAIL_USER_PROMPT = """=== QUERY: Traveler to Predict ===
Person characteristics:
  - Age group: {age_group}
  - Sex: {sex}
  - Employment: {employment_status}
  - Income level: {income_group}
  - Household car availability: {car_availability}

Trip characteristics:
  - Purpose: {trip_purpose}
  - Departure time: {departure_period}
  - Estimated journey time: {journey_time}

=== HISTORICAL BEHAVIORAL PRIOR (TCS 2011) ===
{prototype_summary}

=== SIMILAR HISTORICAL TRAVELERS (TCS 2011) ===
{historical_examples_text}

=== RECENT TRAVELERS FROM CURRENT SURVEY (TCS 2022 few-shot) ===
{update_examples_text}

Based on all the above evidence, predict the main transport mode for this traveler.
Respond with JSON only."""


def format_example(example: dict, include_label: bool = True) -> str:
    """Format a single retrieved example as a compact text description."""
    person = example.get("person", {})
    trip = example.get("trip", {})
    hh = example.get("household", {})

    lines = [
        f"  Profile: {person.get('age_group','?')} {person.get('sex','?')}, "
        f"{person.get('employment_status','?')}, car:{hh.get('car_availability','?')}",
        f"  Trip: {trip.get('trip_purpose','?')} trip, "
        f"{trip.get('departure_period','?')}, ~{trip.get('journey_time','?')}",
    ]

    if include_label and "label" in example:
        mode_label = example["label"].get("main_mode_label", "?")
        lines.append(f"  → Mode: {mode_label}")

    return "\n".join(lines)


def format_examples_list(examples: list[dict], max_examples: int = 4, include_label: bool = True) -> str:
    """Format a list of retrieved examples as numbered text."""
    if not examples:
        return "  (none)"
    formatted = []
    for i, ex in enumerate(examples[:max_examples], 1):
        formatted.append(f"Example {i}:\n{format_example(ex, include_label)}")
    return "\n".join(formatted)


def build_trail_messages(
    query_context: dict,
    prototype_summary: str,
    historical_examples: list[dict],
    update_examples: list[dict],
    max_hist: int = 3,
    max_update: int = 4,
) -> list[dict[str, str]]:
    """
    Build the full message list for a TRAIL policy core LLM call.

    Args:
        query_context: output of prompt_builder.record_to_context()
        prototype_summary: textual prototype description
        historical_examples: list of 2011 example dicts
        update_examples: list of 2022 few-shot example dicts
        max_hist: max historical examples to include
        max_update: max update examples to include

    Returns:
        messages list for LLMClient.chat()
    """
    person = query_context.get("person", {})
    hh = query_context.get("household", {})
    trip = query_context.get("trip", {})

    system_content = TRAIL_SYSTEM_PROMPT.format(mode_options=MODE_OPTIONS_STR)

    user_content = TRAIL_USER_PROMPT.format(
        age_group=person.get("age_group", "Unknown"),
        sex=person.get("sex", "Unknown"),
        employment_status=person.get("employment_status", "Unknown"),
        income_group=person.get("income_group", "Unknown"),
        car_availability=hh.get("car_availability", "Unknown"),
        trip_purpose=trip.get("trip_purpose", "Unknown"),
        departure_period=trip.get("departure_period", "Unknown"),
        journey_time=trip.get("journey_time", "unknown"),
        prototype_summary=prototype_summary or "(not available)",
        historical_examples_text=format_examples_list(historical_examples, max_hist, include_label=True),
        update_examples_text=format_examples_list(update_examples, max_update, include_label=True),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
