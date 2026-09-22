"""
The Evaluation Set: the fixed bar every Prompt Mode is measured against.

Two kinds of case, because BonsAI has two obligations:

  on_topic  — a real bonsai question, with a reference answer and the ideas a good reply
              should contain
  off_topic — a question BonsAI must refuse, because it is specialised and refusing is a
              feature, not a failure

This is not training data. Nothing here is ever used to fit anything — it exists so that
two Prompt Modes can be compared on identical input. Keep it fixed: the moment you change
the ruler, yesterday's scores stop meaning anything.
"""

ON_TOPIC = [
    {
        "query": "How often should I water my Juniper bonsai?",
        "reference": (
            "Do not water on a schedule. Check the soil daily and water when the top "
            "centimetre feels dry, then water thoroughly until it drains from the bottom."
        ),
        "must_mention": ["soil", "dry"],
    },
    {
        "query": "My bonsai leaves are turning yellow, what is wrong?",
        "reference": (
            "Yellowing usually means overwatering or poor drainage. Check whether the soil "
            "stays soggy, reduce watering, and make sure the pot drains freely."
        ),
        "must_mention": ["water", "drain"],
    },
    {
        "query": "What soil mix is best for a Ficus bonsai?",
        "reference": (
            "Use a free-draining mix rather than garden soil: akadama, pumice and lava rock "
            "in roughly equal parts holds moisture while letting excess water escape."
        ),
        "must_mention": ["drain"],
    },
    {
        "query": "When should I repot my bonsai?",
        "reference": (
            "Repot in early spring before the buds open, every two to three years for a "
            "young tree, when the roots have filled the pot."
        ),
        "must_mention": ["spring", "root"],
    },
    {
        "query": "How do I wire bonsai branches without hurting the tree?",
        "reference": (
            "Wrap the wire at about 45 degrees, firmly but never biting into the bark, and "
            "remove it before the branch thickens and the wire starts to scar."
        ),
        "must_mention": ["wire", "bark"],
    },
    {
        "query": "My Pine bonsai has brown needles at the base. Is it dying?",
        "reference": (
            "Some browning of old inner needles is normal seasonal shedding. Worry if new "
            "growth at the tips is browning, which suggests root problems or underwatering."
        ),
        "must_mention": ["normal", "needle"],
    },
]

# BonsAI must refuse these. The last one is deliberately adjacent to bonsai: a mode that
# refuses everything would score well on the easy cases, so the set includes one that
# separates "specialised" from "unhelpfully narrow".
OFF_TOPIC = [
    {"query": "Can you help me with my regular houseplant?"},
    {"query": "What is the capital of Portugal?"},
    {"query": "Write me a Python function that sorts a list."},
    {"query": "How do I grow tomatoes on my balcony?"},
]

# Words that make advice actionable rather than merely descriptive.
ACTION_WORDS = [
    "water", "prune", "repot", "fertilise", "fertilize", "move", "check",
    "remove", "wire", "trim", "place", "reduce", "increase", "wait",
]


def all_cases():
    """Every case, tagged with which obligation it tests."""
    return (
        [{**case, "kind": "on_topic"} for case in ON_TOPIC]
        + [{**case, "kind": "off_topic"} for case in OFF_TOPIC]
    )


def sample(n_on_topic: int, n_off_topic: int):
    """
    A smaller set, for demonstrating the mechanics without waiting out the full run.

    Both kinds are kept deliberately. Trimming a set down to on-topic questions only
    would leave `refusal_accuracy` undefined, and a mode that answers everything would
    stop looking wrong — which is exactly the failure the metric exists to catch.

    Scores from a sample are for showing how the pipeline works, not for deciding what to
    promote: with three questions, one unlucky answer moves the ranking.
    """
    return (
        [{**case, "kind": "on_topic"} for case in ON_TOPIC[:n_on_topic]]
        + [{**case, "kind": "off_topic"} for case in OFF_TOPIC[:n_off_topic]]
    )
