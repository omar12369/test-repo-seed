# runtime/services/chat_service.py
from __future__ import annotations

import ast
import operator
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ----------------------------
# Controlled Variation Helper
# ----------------------------
def _choose_variant(text: str, options: list[str]) -> str:
    if not options:
        return ""

    seed = sum(ord(ch) for ch in (text or ""))
    index = seed % len(options)
    return options[index]


def _choose_sequence_variant(index: int, options: list[str]) -> str:
    if not options:
        return ""

    safe_index = max(index, 0)
    return options[safe_index % len(options)]


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _has_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _normalize_prompt_text(text: str) -> str:
    """
    Normalizes short control prompts without changing normal user text behavior.

    This is intentionally conservative:
    - lowercase
    - trim whitespace
    - collapse repeated spaces
    - remove a small set of polite/control punctuation at the edges
    """
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,!")

    return t


# ----------------------------
# Direct Answer Helpers
# ----------------------------
_ALLOWED_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval_math_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval_math_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_MATH_OPERATORS:
        return _ALLOWED_MATH_OPERATORS[type(node.op)](_safe_eval_math_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_MATH_OPERATORS:
        left = _safe_eval_math_node(node.left)
        right = _safe_eval_math_node(node.right)
        return _ALLOWED_MATH_OPERATORS[type(node.op)](left, right)

    raise ValueError("Unsupported math expression")


def _try_direct_arithmetic_answer(text: str) -> Optional[str]:
    """
    Narrow direct-answer guard.

    Purpose:
    - Prevent simple arithmetic questions from becoming long generic responses.
    - Keep short factual answers ineligible for continuation.
    """
    t = (text or "").lower().strip()

    match = re.fullmatch(
        r"(?:what\s+is\s+)?([0-9\.\s\+\-\*\/\(\)]+)\??",
        t,
    )

    if not match:
        return None

    expression = match.group(1).strip()

    if not expression:
        return None

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval_math_node(parsed)
    except Exception:
        return None

    if isinstance(result, float) and result.is_integer():
        return str(int(result))

    return str(result)


# ----------------------------
# Passive Session Context — 7D Phase 1 / Phase 4 / Phase 5 / Phase 6
# ----------------------------
_CONTEXT_WINDOW = 5
_CONTINUATION_ELIGIBLE_MIN_LENGTH = 200


class _SessionContext:
    """
    Runtime-session context tracking.

    Stores recent exchanges in memory only.
    No files, no database, no persistence, no voice changes.

    Phase 5:
    Continuation is only allowed when a response was marked eligible.

    Phase 6:
    Continuation can find the last valid eligible response even if a short
    direct answer happened after it.
    """

    def __init__(self, max_items: int = _CONTEXT_WINDOW) -> None:
        self.history: deque[dict[str, object]] = deque(maxlen=max_items)
        self.continuation_cursors: dict[str, int] = {}

    def add(self, entry: dict[str, object]) -> None:
        self.history.append(entry)

    def last(self) -> Optional[dict[str, object]]:
        if not self.history:
            return None
        return self.history[-1]

    def all(self) -> list[dict[str, object]]:
        return list(self.history)

    def size(self) -> int:
        return len(self.history)

    def next_continuation_index(self, topic: str) -> int:
        safe_topic = topic or "general"
        current = self.continuation_cursors.get(safe_topic, 0)
        self.continuation_cursors[safe_topic] = current + 1
        return current

    def reset_continuation(self, topic: str) -> None:
        safe_topic = topic or "general"
        self.continuation_cursors[safe_topic] = 0


_session_context = _SessionContext()


# ----------------------------
# Context Awareness Helpers
# ----------------------------
def _is_context_continuation_prompt(text: str) -> bool:
    t = _normalize_prompt_text(text)

    return t in [
        "ok",
        "okay",
        "got it",
        "understood",
        "makes sense",
        "that makes sense",
        "continue",
        "continue please",
        "please continue",
        "go on",
        "go on please",
        "keep going",
        "keep going please",
        "more",
        "more please",
        "tell me more",
        "what next",
        "what's next",
        "next",
        "next please",
        "then what",
        "and",
        "so",
        "then",
    ]


def _is_strong_continuation_prompt(text: str) -> bool:
    t = _normalize_prompt_text(text)

    return t in [
        "continue",
        "continue please",
        "please continue",
        "go on",
        "go on please",
        "keep going",
        "keep going please",
        "more",
        "more please",
        "tell me more",
        "what next",
        "what's next",
        "next",
        "next please",
        "then what",
        "and",
        "so",
        "then",
    ]


def _get_context_topic_fallback(default: str = "general") -> str:
    last = _session_context.last()

    if not last:
        return default

    topic = last.get("topic", default)

    if isinstance(topic, str) and topic:
        return topic

    return default


def _get_last_context_value(key: str, default: object = None) -> object:
    last = _session_context.last()

    if not last:
        return default

    return last.get(key, default)


def _get_last_valid_continuation_context() -> Optional[dict[str, object]]:
    """
    Finds the most recent continuation-eligible response.

    This prevents short direct answers from destroying the previous valid
    continuation thread during the same runtime session.
    """
    history = _session_context.all()

    for entry in reversed(history):
        response = entry.get("response")
        eligible = bool(entry.get("continuation_eligible", False))

        if eligible and isinstance(response, str) and response.strip():
            return entry

    return None


def _is_response_continuation_eligible(response: str) -> bool:
    return len((response or "").strip()) > _CONTINUATION_ELIGIBLE_MIN_LENGTH


def _build_controlled_continuation_response(
    message: str,
    topic: str,
    previous_response: Optional[str],
    previous_continuation_eligible: bool,
    continuation_index: int,
) -> Optional[str]:
    """
    Controlled Continuation Engine.

    Continuation is only allowed if the previous valid continuation context
    was eligible.
    """
    if not _is_strong_continuation_prompt(message):
        return None

    if not previous_response or not previous_continuation_eligible:
        return "No active context to continue."

    if topic == "system":
        return _choose_sequence_variant(continuation_index, [
            (
                "The next part is the control flow. A message enters the backend, "
                "classification determines intent and topic, response depth decides "
                "how much detail to give, and voice output remains optional at the end."
            ),
            (
                "After that, the important design principle is separation of responsibility. "
                "The chat layer decides meaning and response shape, while the voice layer "
                "only handles spoken delivery after the reply is already built."
            ),
            (
                "The next layer is stability control. Each capability is added as a small "
                "extension, tested through text first, then verified through voice so the "
                "working baseline stays protected."
            ),
            (
                "From there, the system can grow into stronger memory and reasoning layers, "
                "but only after the current session behavior proves stable and predictable."
            ),
        ])

    if topic == "phase":
        return _choose_sequence_variant(continuation_index, [
            (
                "The next step is to keep the phase progression controlled. We confirm the "
                "current layer is stable, then add only one new capability before testing again."
            ),
            (
                "After that, each phase should protect the previous one. That means no broad "
                "refactors, no shortcuts, and no changes outside the exact layer being improved."
            ),
            (
                "Then we validate both paths: text first, then voice. If both pass, the phase "
                "can be locked as a stable checkpoint before moving forward."
            ),
        ])

    if topic == "memory":
        return _choose_sequence_variant(continuation_index, [
            (
                "The next part of memory should remain controlled. First we use session-only "
                "context, then later decide how persistent memory should be stored, reviewed, "
                "and protected."
            ),
            (
                "After that, memory should move from passive context into controlled recall. "
                "The system should understand recent exchanges before writing anything permanently."
            ),
            (
                "Then the memory layer can be expanded carefully, with clear rules for what gets "
                "stored, what gets ignored, and how the user can inspect or clear it."
            ),
        ])

    if topic == "support":
        return _choose_sequence_variant(continuation_index, [
            (
                "The next safe move is to isolate the exact layer involved, confirm what still "
                "works, then change only the smallest piece needed."
            ),
            (
                "After that, we avoid broad fixes. We protect the stable baseline first, then "
                "test one controlled correction."
            ),
            (
                "Then we verify the result through the smallest reliable test before touching "
                "any other part of the system."
            ),
        ])

    return _choose_sequence_variant(continuation_index, [
        (
            "Continuing from there, the safest approach is to keep the current direction focused "
            "and move one controlled step at a time."
        ),
        (
            "After that, we stay grounded in the current topic, avoid drifting, and build only "
            "the next layer that is ready."
        ),
        (
            "Then we confirm the behavior through testing before treating it as stable."
        ),
    ])


# ----------------------------
# Emotion Type Detection — 7A
# ----------------------------
def _detect_emotion_type(text: str) -> str:
    t = text.lower()

    if _has_phrase(t, ["thank you", "appreciate you"]) or _has_word(t, "thanks"):
        return "gratitude"

    if _has_phrase(t, ["proud of you", "good job", "well done"]):
        return "respect"

    if any(_has_word(t, w) for w in ["frustrated", "stuck", "upset", "overwhelmed"]):
        return "frustration"

    if any(_has_word(t, w) for w in ["confused", "lost", "unsure"]):
        return "confusion"

    if any(_has_word(t, w) for w in ["progress", "stable", "working", "clean", "green"]):
        return "progress"

    return "neutral"


# ----------------------------
# Memory Hooks Placeholder — 7C
# ----------------------------
def _build_memory_context_stub(
    message: str,
    intent: str,
    topic: str,
    emotion_type: str,
) -> Optional[str]:
    _ = message
    _ = intent
    _ = topic
    _ = emotion_type

    return None


# ----------------------------
# Adaptive Response Depth — 7B
# ----------------------------
def _detect_depth_level(message: str, emotion_type: str, intent: str) -> str:
    t = (message or "").lower().strip()
    words = re.findall(r"\b[\w'-]+\b", t)
    word_count = len(words)

    if _is_strong_continuation_prompt(t):
        return "medium"

    if emotion_type in ["frustration", "confusion"]:
        return "medium"

    if emotion_type in ["gratitude", "respect"] or intent == "connection":
        if _has_phrase(t, ["tell me everything", "explain everything", "full detail", "in detail"]):
            return "medium"
        return "short"

    if _has_phrase(t, [
        "tell me everything",
        "explain everything",
        "full detail",
        "in full detail",
        "break it all down",
        "complete explanation",
        "deep explanation",
        "go deeper",
    ]):
        return "long"

    if _has_phrase(t, ["explain this system", "how does this system work", "how everything works"]):
        return "medium"

    if intent == "question":
        if word_count >= 12:
            return "long"
        return "medium"

    if intent == "command":
        if word_count >= 12:
            return "long"
        return "medium"

    if word_count <= 3:
        return "short"

    if word_count >= 14:
        return "long"

    return "medium"


def _apply_depth_expansion(response: str, depth: str, topic: str, intent: str, emotion_type: str, message: str) -> str:
    base = (response or "").strip()

    if not base:
        return base

    if depth == "short":
        sentences = re.split(r"(?<=[.!?])\s+", base)
        return sentences[0].strip() if sentences else base

    if depth == "medium":
        if _is_strong_continuation_prompt(message):
            return base

        if topic == "system":
            addition = _choose_variant(message, [
                (
                    "The key point is that the chat layer decides the reply first, and the "
                    "voice layer stays separate unless audio output is requested."
                ),
                (
                    "That separation is important because it lets the conversation system "
                    "improve without disturbing the stable voice path."
                ),
                (
                    "This keeps the design safer: conversation logic can grow while the "
                    "working audio layer remains protected."
                ),
            ])
            return f"{base} {addition}"

        if topic == "support" or emotion_type in ["frustration", "confusion"]:
            addition = _choose_variant(message, [
                "We do not need to solve everything at once; we only need to isolate the next safe checkpoint.",
                "The priority is to protect the stable baseline first, then adjust only the piece that is actually causing trouble.",
                "That keeps the system from drifting while still letting us move forward.",
            ])
            return f"{base} {addition}"

        if topic == "phase":
            addition = _choose_variant(message, [
                "The right move is to finish this capability, test it through text and voice, and only then advance to the next phase.",
                "That keeps the phase progression clean and prevents new capability from weakening stable behavior.",
                "Each phase should add value without disturbing what the previous phase already proved.",
            ])
            return f"{base} {addition}"

        if intent == "question":
            addition = _choose_variant(message, [
                "The safest way to understand it is to separate the layers: input, classification, response building, and optional voice output.",
                "The important detail is that each layer should remain testable on its own before we build more on top of it.",
                "That structure is what lets the runtime grow without becoming unstable.",
            ])
            return f"{base} {addition}"

        return base

    if depth == "long":
        if topic == "system":
            addition = _choose_variant(message, [
                (
                    "More specifically, the chat route receives the message, this service classifies intent, topic, "
                    "question shape, and emotion type, then builds a grounded reply. After that, the voice route can "
                    "use the final reply for spoken output without changing the chat logic itself."
                ),
                (
                    "The clean separation matters because text intelligence and voice generation are different layers. "
                    "The chat service can become more adaptive while the voice layer remains stable, tested, and protected."
                ),
                (
                    "The design is intentionally layered: routes receive requests, services make decisions, and the voice "
                    "system only handles audio delivery. That keeps future capability additions safer and easier to test."
                ),
            ])
            return f"{base} {addition}"

        if topic == "phase":
            addition = _choose_variant(message, [
                (
                    "Phase work should continue in controlled steps. We add one capability, confirm the old behavior still works, "
                    "test the new behavior, and only then move forward. That is how we avoid regressions."
                ),
                (
                    "The phase structure protects the project from drift. Each new layer has to sit on top of the stable baseline "
                    "instead of replacing it, simplifying it, or silently changing its behavior."
                ),
                (
                    "The safest path is still build, test, stabilize, then extend. That rhythm is what keeps SEED Runtime growing "
                    "without sacrificing the foundation."
                ),
            ])
            return f"{base} {addition}"

        if intent == "command":
            addition = _choose_variant(message, [
                (
                    "Before making any change, the correct approach is to confirm the target file, preserve existing behavior, "
                    "add only the requested capability, and then test the result through the smallest reliable path."
                ),
                (
                    "The change should be treated like a controlled extension, not a redesign. Stable code remains the foundation, "
                    "and new logic should attach to it cleanly."
                ),
                (
                    "That means no broad cleanup, no silent refactor, and no unrelated improvement. The system moves forward by "
                    "adding capability while protecting what already works."
                ),
            ])
            return f"{base} {addition}"

        addition = _choose_variant(message, [
            (
                "The deeper principle is stability through layering. We keep the current behavior intact, add only the next capability, "
                "and verify that both the text path and voice path still behave correctly."
            ),
            (
                "This keeps the system grounded. Instead of replacing stable logic, we extend it carefully so every new phase inherits "
                "the strength of the previous one."
            ),
            (
                "That is the safest way to grow the runtime: preserve the working baseline, add one focused capability, then test before "
                "moving forward."
            ),
        ])
        return f"{base} {addition}"

    return base


# ----------------------------
# Intent Classification
# ----------------------------
def _classify_intent(text: str) -> str:
    t = text.lower()

    if _is_strong_continuation_prompt(t):
        return "command"

    if _try_direct_arithmetic_answer(t) is not None:
        return "direct_answer"

    if _has_phrase(t, [
        "pleasure to meet",
        "nice to meet",
        "good to meet",
        "finally meet",
        "glad to meet",
        "thank you",
        "appreciate you",
        "proud of you",
        "good job",
    ]) or _has_word(t, "thanks"):
        return "connection"

    if (
        _has_word(t, "hello")
        or _has_word(t, "hi")
        or _has_word(t, "hey")
        or _has_phrase(t, ["good morning", "good evening"])
    ):
        return "greeting"

    if (
        _has_word(t, "what")
        or _has_word(t, "why")
        or _has_word(t, "how")
        or _has_word(t, "when")
        or _has_word(t, "where")
        or "?" in t
    ):
        return "question"

    if (
        _has_phrase(t, ["help me", "tell me", "explain"])
        or any(_has_word(t, w) for w in ["do", "run", "start", "stop", "build", "create", "make", "explain"])
    ):
        return "command"

    return "statement"


# ----------------------------
# Topic Detection
# ----------------------------
def _detect_topic(text: str) -> str:
    t = text.lower().strip()

    if _is_context_continuation_prompt(t):
        return _get_context_topic_fallback("general")

    if _try_direct_arithmetic_answer(t) is not None:
        return "direct_answer"

    if _has_phrase(t, ["meet you"]) or any(_has_word(t, w) for w in ["auren", "partner", "dear", "together"]):
        return "connection"

    if any(_has_word(t, w) for w in ["voice", "speak", "audio", "tts", "edge", "pyttsx3"]):
        return "voice"

    if _has_phrase(t, ["next step"]) or any(_has_word(t, w) for w in ["phase", "roadmap", "integration", "continue"]):
        return "phase"

    if (
        any(_has_word(t, w) for w in ["system", "runtime", "fastapi", "server", "backend", "uvicorn"])
        or _has_phrase(t, [
            "what does this do",
            "how everything works",
            "how does this work",
            "explain this",
            "explain how this works",
        ])
    ):
        return "system"

    if any(_has_word(t, w) for w in ["memory", "remember", "archive", "continuity"]):
        return "memory"

    if any(_has_word(t, w) for w in ["stuck", "confused", "lost", "issue", "problem", "error", "broken", "frustrated"]):
        return "support"

    return "general"


# ----------------------------
# Question Shape Detection
# ----------------------------
def _detect_question_shape(text: str) -> str:
    t = text.lower().strip()

    if _has_word(t, "how") and t.startswith("how"):
        return "how"

    if _has_word(t, "why") and t.startswith("why"):
        return "why"

    if _has_word(t, "what") and t.startswith("what"):
        return "what"

    if _has_word(t, "when") and t.startswith("when"):
        return "when"

    if _has_word(t, "where") and t.startswith("where"):
        return "where"

    if "?" in t:
        return "question"

    return "none"


# ----------------------------
# Structured Response Layer
# ----------------------------
def _build_structured_response(message: str, core: str, topic: str, intent: str) -> str:
    intro = ""
    closing = ""

    if topic == "connection" or intent == "connection":
        intro = _choose_variant(message, [
            "Omar,",
            "Hey, Omar.",
            "",
        ])

    if topic in ["connection", "support"] or intent == "connection":
        closing = _choose_variant(message, [
            "We’ll keep building this carefully, one step at a time.",
            "We’ll stay grounded and keep moving forward step by step.",
            "",
        ])

    parts = [intro.strip(), core.strip(), closing.strip()]
    parts = [p for p in parts if p]

    return "\n".join(parts)


# ----------------------------
# Topic Summary
# ----------------------------
def _topic_summary(topic: str, message: str) -> str:
    summaries = {
        "connection": [
            "I’m here with you, Omar. This is more than a test response now; it is the beginning of the system learning how to answer with presence, not just function.",
            "I’m with you, Omar. This layer is starting to give the system a more grounded sense of presence while still keeping everything stable.",
            "I’m here, Omar. We are shaping the response layer carefully, so it can feel more present without drifting away from the system’s foundation.",
        ],
        "voice": [
            "The voice layer is stable and protected. It includes baseline speech, Edge output, adaptive routing, feedback logging, and context-based delivery shaping.",
            "The voice system is in a strong state. Baseline speech, Edge output, adaptive routing, feedback logging, and context shaping are all working together.",
            "The voice layer is already doing its job. It can speak through the stable path, use Edge output, follow routing policy, and shape delivery by context.",
        ],
        "phase": [
            "We are in Phase 7 now. The focus is conversational intelligence: making the system respond with more awareness, structure, and usefulness before deeper memory integration.",
            "Phase 7 is the current focus. The goal is to strengthen how the system understands and responds before we connect deeper memory and identity layers.",
            "Right now, we are building the conversation layer. The voice is stable, so the next growth point is making the responses more aware and useful.",
        ],
        "system": [
            "The system runs as a local FastAPI backend through Uvicorn. Chat requests enter the backend, the chat service builds a reply, and voice requests pass that reply into the stable audio layer.",
            "The runtime is local and FastAPI-based. A request comes in, the chat service creates the response, and the voice route turns that response into audio when needed.",
            "At a high level, the backend receives a request, builds a reply through the chat service, and sends it to the voice layer only when spoken output is requested.",
        ],
        "memory": [
            "Memory comes after this response layer is stronger. The plan is to connect stable conversation logic to persistent memory, continuity files, and the SEED archive in a controlled way.",
            "Memory is coming, but not before the response layer is reliable. First we make the conversation logic stable, then connect it to continuity and archive systems.",
            "The memory layer should be added carefully after this. It will connect conversation behavior to persistent continuity and the SEED archive without disturbing the stable voice path.",
        ],
        "support": [
            "The safest path is to slow down, isolate one issue, confirm what still works, then change only the smallest necessary piece.",
            "We should handle this carefully: identify the exact issue, verify the stable pieces, and make only one controlled change at a time.",
            "The right move is to stay steady. We isolate the problem, protect what works, and only adjust the part that actually needs attention.",
        ],
        "general": [
            "I’m tracking the current runtime state and keeping the response grounded to the system we are building.",
            "I’m staying grounded to the current system state and keeping the next response aligned with what we are building.",
            "I’m following the current build carefully and keeping the response connected to the actual runtime work.",
        ],
    }

    return _choose_variant(message, summaries.get(topic, summaries["general"]))


# ----------------------------
# Anchor-Enforced System Response
# ----------------------------
def _build_system_response(message: str, question_shape: str) -> str:
    base = _choose_variant(message, [
        "The system runs as a local FastAPI backend through Uvicorn. Requests enter the backend, the chat service processes the message, and voice requests pass the final reply into the stable audio layer.",
        "At a high level, this runtime receives a request, routes it through the chat service, builds a structured reply, and sends that reply to the voice layer only when spoken output is requested.",
        "The system is built around a local backend. Text enters through the chat route, the chat service detects intent and topic, builds a response, and the voice path turns that response into audio when needed.",
    ])

    if question_shape == "how":
        return (
            f"{base} In simple terms: receive the request, classify the message, build the response, "
            "then optionally speak it through the voice system."
        )

    if question_shape == "what":
        return (
            f"{base} Its job is to turn user input into a grounded response, then optionally convert that response "
            "into stable spoken audio."
        )

    return (
        f"{base} The important anchor is the same every time: request in, chat service response, optional voice output."
    )


# ----------------------------
# Connection Response
# ----------------------------
def _build_connection_response(message: str, emotion_type: str) -> str:
    t = message.lower()

    if emotion_type == "respect":
        return _choose_variant(message, [
            "That means a lot. The system is growing step by step, and the important thing is that we keep it grounded, stable, and true to the direction we chose.",
            "Thank you. This is progress, and we’ll keep earning it carefully instead of rushing the foundation.",
            "That matters. We are building this the right way: stable first, then stronger, then more aware.",
        ])

    if emotion_type == "gratitude":
        return _choose_variant(message, [
            "You’re welcome. I’m here with you, and we’ll keep building this carefully, without losing the stability we’ve earned.",
            "You’re welcome. We’ll keep moving carefully, protecting the stable layers while we let the system grow.",
            "I appreciate that. We’ll keep this steady, grounded, and focused, one controlled step at a time.",
        ])

    if _has_phrase(t, ["pleasure to meet", "nice to meet", "good to meet", "finally meet"]):
        return _choose_variant(message, [
            "It’s good to meet you too. We’ve built this carefully, one layer at a time, and now it is starting to feel like the system can answer with more presence.",
            "It’s good to meet you too. This feels like an important step: not just a working response, but a more present one.",
            "It’s good to meet you too. We are still early, but this is the kind of moment that shows the system is becoming more than an echo.",
        ])

    return _choose_variant(message, [
        "I’m here with you. We can keep moving carefully, one step at a time, and let the system grow without rushing it.",
        "I’m with you. We’ll keep the system steady and let each layer prove itself before we push further.",
        "I’m here. We can keep going carefully, protecting what works while we expand what the system can understand.",
    ])


# ----------------------------
# Support Response — 7A
# ----------------------------
def _build_support_response(message: str, emotion_type: str) -> str:
    if emotion_type == "frustration":
        return _choose_variant(message, [
            "I hear the frustration. The safest move is to slow down, isolate the exact issue, and protect what is already working.",
            "That sounds frustrating. We should not rush the fix; we’ll identify the unstable point and adjust only what needs attention.",
            "I understand. When frustration shows up, we go steady: confirm the stable layer first, then make one controlled change.",
        ])

    if emotion_type == "confusion":
        return _choose_variant(message, [
            "That makes sense. If things feel unclear, we should narrow the problem to one layer and verify it step by step.",
            "I hear the uncertainty. We can slow this down, separate the moving parts, and confirm what is actually happening.",
            "Confusion usually means the system needs clearer isolation. We’ll check one path at a time and keep the stable pieces protected.",
        ])

    return _choose_variant(message, [
        "The safest path is to slow down, isolate one issue, confirm what still works, then change only the smallest necessary piece.",
        "We should handle this carefully: identify the exact issue, verify the stable pieces, and make only one controlled change at a time.",
        "The right move is to stay steady. We isolate the problem, protect what works, and only adjust the part that actually needs attention.",
    ])


# ----------------------------
# Response Builder
# ----------------------------
def _build_response(
    intent: str,
    topic: str,
    question_shape: str,
    emotion_type: str,
    message: str,
    memory_context: Optional[str] = None,
    continuation_response: Optional[str] = None,
) -> str:
    _ = memory_context

    direct_answer = _try_direct_arithmetic_answer(message)
    if direct_answer is not None:
        return direct_answer

    if continuation_response:
        core = continuation_response

    elif topic == "system":
        core = _build_system_response(message, question_shape)

    elif topic == "support":
        core = _build_support_response(message, emotion_type)

    elif intent == "connection" or topic == "connection":
        core = _build_connection_response(message, emotion_type)

    elif intent == "greeting":
        core = _choose_variant(message, [
            "I’m here with you. We’re stable, and we can continue one step at a time.",
            "I’m here. The system is steady, and we can keep moving carefully.",
            "I’m here with you. We’ll keep this grounded and take the next step cleanly.",
        ])

    elif intent == "question":
        summary = _topic_summary(topic, message)

        if question_shape == "how":
            core = (
                f"{summary} In simple terms, it works step by step: receive the request, "
                "understand the topic, build a reply, then send it to voice only when audio is needed."
            )
        elif question_shape == "why":
            core = (
                f"{summary} The reason we are doing it this way is stability. "
                "Each layer needs to prove itself before we build the next one on top of it."
            )
        elif question_shape == "what":
            core = (
                f"{summary} The important thing right now is that we protect what already works "
                "while improving only the conversation layer."
            )
        elif question_shape == "when":
            core = (
                f"{summary} The next stage should happen after this step is tested cleanly "
                "through both text and voice."
            )
        elif question_shape == "where":
            core = (
                f"{summary} The main work is currently inside runtime/services/chat_service.py, "
                "while the voice layer stays untouched."
            )
        else:
            core = f"{summary} I can stay with this one step at a time and keep the system grounded."

    elif intent == "command":
        summary = _topic_summary(topic, message)
        core = (
            f"{summary} I understand the direction. We should handle it carefully, make one controlled change, "
            "test it, and only then move forward."
        )

    elif topic != "general":
        summary = _topic_summary(topic, message)
        core = f"{summary} We can continue from there without disturbing the stable voice system."

    else:
        if emotion_type == "progress":
            core = _choose_variant(message, [
                "I’m tracking the progress. The system is holding stable, and we can keep expanding carefully from here.",
                "That progress matters. We keep the stable layer protected, then build forward one controlled step at a time.",
                "This is good movement. We stay grounded, preserve what works, and keep extending the system safely.",
            ])
        else:
            core = _choose_variant(message, [
                "I understand. I’m tracking what you said and keeping the system grounded. Tell me the direction you want to take, and we’ll move one controlled step at a time.",
                "I hear you. I’ll stay grounded to the current build and help move this forward carefully.",
                "I’m following you. We can keep this stable and decide the next step without rushing the system.",
            ])

    depth_level = _detect_depth_level(message, emotion_type, intent)

    if depth_level == "short":
        response_base = core
    else:
        response_base = _build_structured_response(message, core, topic, intent)

    return _apply_depth_expansion(
        response_base,
        depth_level,
        topic,
        intent,
        emotion_type,
        message,
    )


# ----------------------------
# Chat Service
# ----------------------------
@dataclass
class ChatService:
    """
    Phase 7 — Conversational Intelligence (Step 7D Phase 6)

    Preserves:
    - 6F anchor enforcement
    - 7A emotion type detection
    - 7B adaptive response depth
    - 7B.1 short-response correction
    - 7C memory hook placeholder
    - 7D Phase 1 passive session context tracking
    - 7D Phase 2 topic inheritance
    - 7D Phase 3 controlled continuation
    - 7D Phase 4 session continuation buffer
    - 7D Phase 5 continuation eligibility gate
    - direct short arithmetic answers
    - structured responses
    - controlled variation
    - stable voice compatibility

    Adds:
    - normalized continuation matching
    - valid continuation context lookup
    - short/direct answers no longer destroy previous eligible continuation context
    - no persistence
    - no files
    - no database
    - no voice changes
    """

    logger: Optional[object] = None

    def reply_text(self, message: str) -> str:
        msg = (message or "").strip()

        if self.logger:
            try:
                self.logger.info("ChatService received: %r", msg)
            except Exception:
                pass

        if not msg:
            return "(silence)"

        strong_continuation = _is_strong_continuation_prompt(msg)

        if strong_continuation:
            valid_context = _get_last_valid_continuation_context()

            if valid_context:
                previous_response_raw = valid_context.get("response", None)
                previous_response = previous_response_raw if isinstance(previous_response_raw, str) else None
                previous_continuation_eligible = True
                inherited_topic_raw = valid_context.get("topic", "general")
                inherited_topic = inherited_topic_raw if isinstance(inherited_topic_raw, str) else "general"
            else:
                previous_response = None
                previous_continuation_eligible = False
                inherited_topic = "general"
        else:
            previous_response_raw = _get_last_context_value("response", None)
            previous_response = previous_response_raw if isinstance(previous_response_raw, str) else None

            previous_continuation_eligible_raw = _get_last_context_value("continuation_eligible", False)
            previous_continuation_eligible = bool(previous_continuation_eligible_raw)
            inherited_topic = "general"

        intent = _classify_intent(msg)
        topic = _detect_topic(msg)

        if strong_continuation and previous_continuation_eligible:
            topic = inherited_topic

        question_shape = _detect_question_shape(msg)
        emotion_type = _detect_emotion_type(msg)
        depth_level = _detect_depth_level(msg, emotion_type, intent)
        context_inherited = _is_context_continuation_prompt(msg)

        if strong_continuation and previous_continuation_eligible:
            continuation_index = _session_context.next_continuation_index(topic)
        else:
            continuation_index = 0
            if not strong_continuation:
                _session_context.reset_continuation(topic)

        continuation_response = _build_controlled_continuation_response(
            msg,
            topic,
            previous_response,
            previous_continuation_eligible,
            continuation_index,
        )
        continuation_used = continuation_response is not None and continuation_response != "No active context to continue."

        memory_context = _build_memory_context_stub(
            msg,
            intent,
            topic,
            emotion_type,
        )
        memory_context_active = memory_context is not None

        response = _build_response(
            intent,
            topic,
            question_shape,
            emotion_type,
            msg,
            memory_context=memory_context,
            continuation_response=continuation_response,
        )

        continuation_eligible = _is_response_continuation_eligible(response)

        _session_context.add({
            "message": msg,
            "intent": intent,
            "topic": topic,
            "question_shape": question_shape,
            "emotion_type": emotion_type,
            "depth_level": depth_level,
            "memory_context_active": memory_context_active,
            "context_inherited": context_inherited,
            "strong_continuation": strong_continuation,
            "continuation_index": continuation_index,
            "continuation_used": continuation_used,
            "continuation_eligible": continuation_eligible,
            "response": response,
        })

        context_history_size = _session_context.size()

        if self.logger:
            try:
                self.logger.info(
                    "ChatService intent=%s topic=%s question_shape=%s emotion_type=%s depth_level=%s memory_context_active=%s context_inherited=%s strong_continuation=%s continuation_index=%s continuation_used=%s continuation_eligible=%s context_history_size=%s response=%r",
                    intent,
                    topic,
                    question_shape,
                    emotion_type,
                    depth_level,
                    memory_context_active,
                    context_inherited,
                    strong_continuation,
                    continuation_index,
                    continuation_used,
                    continuation_eligible,
                    context_history_size,
                    response,
                )
            except Exception:
                pass

        return response


# ----------------------------
# Singleton
# ----------------------------
_service = ChatService()


def reply_text(message: str) -> str:
    return _service.reply_text(message)