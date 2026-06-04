"""
LLM-based node annotator for text-attributed graphs.

Wraps OpenAI API (GPT-4o-mini by default) for zero-shot node classification.
Supports disk-based caching and loading pre-cached LoCLE annotations.
Tracks budget (total API calls + token usage) across all rounds.
"""

import json
import logging
import os
import os.path as osp
import re
import time
from collections import Counter
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Prompt templates — matches LoCLE / GNN-as-Judge style
# ──────────────────────────────────────────────────────────────

_PROMPT_TEMPLATES = {
    "cora": (
        "Which of the following sub-categories of AI does this paper belong to? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above."
    ),
    "citeseer": (
        "Which of the following themes does this paper belong to? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above."
    ),
    "pubmed": (
        "Which of the following topics does this scientific publication discuss? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above."
    ),
    "wikics": (
        "Which of the following branches of Computer Science does this article belong to? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above."
    ),
    "dblp": (
        "Which of the following research areas does this paper belong to? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above."
    ),
    "arxiv": (
        "Which of the following arXiv CS sub-categories does this paper belong to? "
        "Here are the {num_classes} categories: {class_list}. "
        "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
        "Only use category names from the list above (e.g., cs.LG, cs.CV)."
    ),
}

_DEFAULT_TEMPLATE = (
    "Classify the following text into exactly one of these {num_classes} categories: "
    "{class_list}. "
    "Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
    "Only use category names from the list above."
)


def _build_prompt_llmgnn_arxiv(text):
    # type: (str) -> str
    """
    Bit-exact port of LLM-GNN's arxiv consistency prompt.

    Reconstructs:
        no_task_question = zero_shot_prompt([text], label_names, need_tasks=False,
                                            object_cat="Paper",
                                            question="Which arxiv CS-subcategory does this paper belong to?",
                                            answer_format=<unused>)[0]
        final = exp.topk_confidence(no_task_question, 3, 'arxiv')

    which yields (see baselines/llmgnn/src/llm.py lines 557-571 and 256-258):

        "Paper: Paper: \n<text>\nTask: \nWhich arxiv CS-subcategory does this paper belong to?\n. Which arxiv CS-subcategory does this paper belong to? Output your answer in the form of arXiv CS sub-categories like \"cs.XX\" together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}, ...]\n"

    The double "Paper: " at the start is intentional — it is what LLM-GNN's own
    pipeline actually sends to the API, and is part of what produced the
    reported 72.92% annotation accuracy on ogbn-arxiv.
    """
    inner = "Paper: \n" + text + "\n" + "Task: \n" + \
            "Which arxiv CS-subcategory does this paper belong to?\n"
    # topk_confidence(question=inner, k=3, name='arxiv'):
    final = (
        "Paper: {}. Which arxiv CS-subcategory does this paper belong to? "
        "Output your answer in the form of arXiv CS sub-categories like \"cs.XX\" "
        "together with a confidence ranging from 0 to 100, in the form of a list "
        "of python dicts like [{{\"answer\":<answer_here>, \"confidence\": "
        "<confidence_here>}}, ...]\n"
    ).format(inner)
    return final


def _build_prompt_llmgnn_generic(text, class_names, dataset_name):
    # type: (str, List[str], str) -> str
    """
    Port of LLM-GNN's Cora/CiteSeer/PubMed/WikiCS top-k consistency prompt.

    Reconstructs (from baselines/llmgnn/src/llm.py lines 557-571 and 256-263):

        no_task_question = zero_shot_prompt([text], label_names, need_tasks=False,
                                            object_cat="Paper",
                                            question="What's the category of this paper",
                                            answer_format=<unused>)[0]
        final = exp.topk_confidence(no_task_question, 3, '<dataset>')

    Which yields (non-arxiv topk_confidence branch):

        Question: Paper: \n<text>\nTask: \nThere are following categories: \n
        [<label_names>]\nWhat's the category of this paper\n. Provide your 3
        best guesses and a confidence number that each is correct (0 to 100)
        for the following question from most probable to least. The sum of all
        confidence should be 100. For example,
        [{"answer": <your_first_answer>, "confidence": <confidence_for_first_answer>}, ...]

    Per baselines/llmgnn/src/openail/config.py, Cora/CiteSeer/PubMed use
    object_cat="Paper"; WikiCS uses object_cat="Wiki Article"; all use the
    same question string "What's the category of this paper" (with "category"
    meaning paper/article category). For WikiCS, LLM-GNN's config.py actually
    uses "What's the wiki category of this article" — we honour that.
    """
    ds = dataset_name.lower()
    if ds == "wikics":
        object_cat = "Wiki Article"
        question = "What's the wiki category of this article"
    else:
        object_cat = "Paper"
        question = "What's the category of this paper"
    # Step 1 — zero_shot_prompt with need_tasks=False.
    no_task = "{}: \n{}\nTask: \n".format(object_cat, text)
    # Non-arxiv branch: include the category list before the question.
    no_task += "There are following categories: \n"
    no_task += "[" + ", ".join(class_names) + "]\n"
    no_task += question + "\n"
    # Step 2 — topk_confidence(q, 3, name) non-arxiv branch.
    final = (
        "Question: {}. Provide your 3 best guesses and a confidence number "
        "that each is correct (0 to 100) for the following question from "
        "most probable to least. The sum of all confidence should be 100. "
        "For example,  [{{\"answer\": <your_first_answer>, \"confidence\": "
        "<confidence_for_first_answer>}}, ...]        "
    ).format(no_task)
    return final


def _build_prompt(text, class_names, dataset_name, prompt_style="ours"):
    # type: (str, List[str], str, str) -> str
    """Build the full classification prompt for a single node."""
    if prompt_style == "llmgnn" and dataset_name.lower() == "arxiv":
        return _build_prompt_llmgnn_arxiv(text)
    if prompt_style == "llmgnn" and dataset_name.lower() in (
        "cora", "citeseer", "pubmed", "wikics", "dblp", "products"
    ):
        return _build_prompt_llmgnn_generic(text, class_names, dataset_name)
    template = _PROMPT_TEMPLATES.get(dataset_name.lower(), _DEFAULT_TEMPLATE)
    class_list = ", ".join(class_names)
    instruction = template.format(
        num_classes=len(class_names),
        class_list=class_list,
    )
    return "Text: {}\n\n{}".format(text, instruction)


# ──────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────

# Regex for the generic placeholder produced by tag_dataset.py when a TAG
# dataset (e.g. ogbn-arxiv) is loaded without per-node text. When the arxiv
# unrestricted-pool path is active, untitled nodes carry these placeholders;
# they carry zero semantic content so we skip the API call rather than waste
# budget on them.
_PLACEHOLDER_RE = re.compile(r"^\s*node_\d+\s*$")


def _is_placeholder_text(text):
    # type: (str) -> bool
    """Return True if ``text`` is empty/whitespace or a ``node_{i}`` placeholder."""
    if text is None:
        return True
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_RE.match(stripped):
        return True
    return False


def _parse_response(response_text, class_names):
    # type: (str, List[str]) -> Tuple[int, float]
    """
    Parse LLM response into (label_index, confidence).

    Tries JSON first, then falls back to fuzzy text matching (like LoCLE).
    Returns (-1, 0.0) if parsing completely fails.
    """
    # Try JSON extraction
    json_match = re.search(r"\{[^}]+\}", response_text)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            label_str = obj.get("label", "")
            confidence = float(obj.get("confidence", 50.0))

            class_names_lower = [c.lower() for c in class_names]
            if label_str.lower() in class_names_lower:
                idx = class_names_lower.index(label_str.lower())
                return idx, min(max(confidence, 0.0), 100.0)

            matches = get_close_matches(
                label_str.lower(), class_names_lower, n=1, cutoff=0.4
            )
            if matches:
                idx = class_names_lower.index(matches[0])
                return idx, min(max(confidence / 2, 0.0), 100.0)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Fallback: find the best matching class name in raw text
    class_names_lower = [c.lower() for c in class_names]
    response_lower = response_text.lower()

    best_idx, best_pos = -1, len(response_lower) + 1
    for i, name in enumerate(class_names_lower):
        pos = response_lower.find(name)
        if pos != -1 and pos < best_pos:
            best_idx = i
            best_pos = pos

    if best_idx == -1:
        matches = get_close_matches(response_lower, class_names_lower, n=1, cutoff=0.3)
        if matches:
            best_idx = class_names_lower.index(matches[0])

    # Extract confidence from text (largest integer found)
    confidence = 50.0
    numbers = re.findall(r"\b(\d{1,3})\b", response_text)
    if numbers:
        candidates = [int(n) for n in numbers if 0 <= int(n) <= 100]
        if candidates:
            confidence = float(max(candidates))

    if best_idx == -1:
        return -1, 0.0

    return best_idx, confidence


def _parse_response_llmgnn(response_text, class_names):
    # type: (str, List[str]) -> Tuple[int, float]
    """
    Parse an LLM-GNN-style top-k response into (label_index, confidence).

    Expected format (from the arxiv consistency prompt):
        [{"answer": "cs.LG", "confidence": 60},
         {"answer": "cs.CV", "confidence": 30},
         {"answer": "cs.AI", "confidence": 10}]

    Strategy (mirrors LLM-GNN's ``retrieve_multiple_answers`` / ``retrieve_answer``):
      1. Find the list-of-dicts literal in the text and ``ast.literal_eval`` it.
      2. Take the FIRST dict ("most likely" — LLM-GNN orders most→least likely).
      3. ``answer`` is expected to be a ``cs.XX`` string; match case-insensitively
         against class_names. If not found, fall back to fuzzy match.
      4. ``confidence`` is 0-100 (or "NN%"); clip.

    Falls back to ``_parse_response`` (the "ours" parser) on any failure so that
    occasional malformed responses still produce a usable label.
    """
    import ast

    text = response_text.strip()
    # Find the outer list [...] substring robustly.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        list_str = text[start:end + 1]
        # LLM-GNN lowercases before parsing in retrieve_multiple_answers; we do
        # the same so "cs.LG" matches class_names case-insensitively.
        parsed = None
        for candidate in (list_str, list_str.lower()):
            try:
                parsed = ast.literal_eval(candidate)
                break
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            first = parsed[0]
            # Handle both "answer" and "answer:" (the training example in
            # llm.py line 558 literally has a stray colon in the key).
            answer = (
                first.get("answer")
                or first.get("answer:")
                or first.get("Answer")
                or ""
            )
            confidence = first.get("confidence", 50)
            if isinstance(confidence, str):
                confidence = confidence.replace("%", "").strip()
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 50.0
            confidence = float(confidence)
            confidence = min(max(confidence, 0.0), 100.0)

            answer_str = str(answer).strip().lower()
            class_names_lower = [c.lower() for c in class_names]
            if answer_str in class_names_lower:
                return class_names_lower.index(answer_str), confidence
            # Try a "cs.xx" substring match — sometimes the model writes
            # "cs.lg (machine learning)".
            for i, c in enumerate(class_names_lower):
                if c in answer_str:
                    return i, confidence
            matches = get_close_matches(answer_str, class_names_lower, n=1, cutoff=0.5)
            if matches:
                return class_names_lower.index(matches[0]), confidence / 2.0

    # Fallback to the generic parser.
    return _parse_response(response_text, class_names)


# ──────────────────────────────────────────────────────────────
# LoCLE cache loading
# ──────────────────────────────────────────────────────────────

def _load_locle_cache(
    cache_dir,   # type: str
    dataset_name,  # type: str
    temperature=0.7,  # type: float
    n=3,         # type: int
    seed=42,     # type: int
):
    # type: (...) -> Dict[int, Tuple[int, float]]
    """
    Load pre-cached LoCLE annotations.

    LoCLE cache format:  {str(node_id): [label_int, confidence_float], ...}
    Filename pattern:    {dataset}_temperature_{t}_n_{n}_output_seed_{s}_save_zero.json
    """
    file_dataset = "citeseer2" if dataset_name.lower() == "citeseer" else dataset_name.lower()
    filename = "{}_temperature_{}_n_{}_output_seed_{}_save_zero.json".format(
        file_dataset, temperature, n, seed
    )
    path = osp.join(cache_dir, filename)

    if not osp.exists(path):
        logger.warning("LoCLE cache not found: %s", path)
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    result = {}
    for node_str, (label, conf) in raw.items():
        result[int(node_str)] = (int(label), float(conf))

    logger.info("Loaded %d LoCLE cached annotations from %s", len(result), path)
    return result


# ──────────────────────────────────────────────────────────────
# Main Annotator
# ──────────────────────────────────────────────────────────────

class LLMAnnotator(object):
    """
    LLM-based zero-shot node annotator with caching and budget tracking.

    Usage:
        annotator = LLMAnnotator(
            dataset_name="cora",
            class_names=["Rule_Learning", ...],
            cache_dir="data/annotations/",
            budget=280,
        )
        results = annotator.annotate_nodes(node_ids, texts)
        results = annotator.load_locle_cache()
    """

    def __init__(
        self,
        dataset_name,       # type: str
        class_names,        # type: List[str]
        cache_dir="data/annotations/",  # type: str
        model="gpt-4o-mini",  # type: str
        budget=None,        # type: Optional[int]
        api_key=None,       # type: Optional[str]
        temperature=0.0,    # type: float
        max_retries=3,      # type: int
        retry_delay=1.0,    # type: float
        n_samples=1,        # type: int
        sampling_temperature=None,  # type: Optional[float]
        budget_per_node=False,  # type: bool
        prompt_style="ours",  # type: str
        force_fresh=False,  # type: bool
        cache_suffix="",    # type: str
    ):
        self.dataset_name = dataset_name.lower()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.cache_dir = cache_dir
        self.model = model
        self.budget = budget
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # LLM-GNN style consistency voting config.
        # n_samples=1 (default) preserves single-shot behaviour.
        # n_samples>1 samples the API n times at sampling_temperature and takes
        # the majority vote; confidence = (majority_count / n_samples) * 100.
        self.n_samples = int(n_samples)
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1, got {}".format(self.n_samples))
        # sampling_temperature defaults to self.temperature to keep default
        # path bit-identical. When n_samples>1 callers typically pass 0.7.
        self.sampling_temperature = (
            float(sampling_temperature) if sampling_temperature is not None else self.temperature
        )
        # budget_per_node=False (default): budget counts individual API calls.
        # budget_per_node=True: budget counts n-sample sets (one set per node),
        # so budget=800 with n_samples=3 yields 800 unique nodes / 2400 calls.
        self.budget_per_node = bool(budget_per_node)

        # Prompt style — "ours" (default, legacy) or "llmgnn" (bit-exact port of
        # LLM-GNN's arxiv top-k consistency prompt, which yielded 72.92%
        # annotation accuracy on ogbn-arxiv in their reference impl).
        # Only affects arxiv dataset currently; other datasets fall through to
        # the legacy "ours" templates regardless.
        if prompt_style not in ("ours", "llmgnn"):
            raise ValueError(
                "prompt_style must be 'ours' or 'llmgnn', got {!r}".format(prompt_style)
            )
        self.prompt_style = prompt_style

        # Budget tracking
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Disk cache: {node_id: (label, confidence)}
        # Cache path rules:
        #   - Default (n_samples=1, sampling_temperature == self.temperature):
        #     use the legacy path "{dataset}_{model}.json" so existing caches
        #     (e.g. arxiv_gpt-3.5-turbo.json with 4371 entries) remain usable
        #     with no filename change.
        #   - Otherwise (n_samples > 1, or sampling_temperature differs from
        #     self.temperature): bucket by sampling params so single-shot and
        #     voting runs never collide. Different (n, t) pairs -> different
        #     cache files. Filename example: arxiv_gpt-3.5-turbo_n3_t0.7.json.
        self._cache = {}  # type: Dict[int, Tuple[int, float]]
        safe_model = self.model.replace("/", "_")
        use_default_path = (
            self.n_samples == 1
            and self.sampling_temperature == self.temperature
        )
        # Prompt-style suffix. When prompt_style != "ours" AND the dataset
        # actually has a custom path for that style (currently only arxiv),
        # append a suffix so caches never mix.
        style_affects_prompt = (
            self.prompt_style == "llmgnn"
            and self.dataset_name in (
                "arxiv", "cora", "citeseer", "pubmed", "wikics", "dblp",
            )
        )
        style_suffix = "_{}".format(self.prompt_style) if style_affects_prompt else ""

        # Optional user-supplied suffix for caches. When non-empty this appends
        # "_<suffix>" before .json so runs that alter the effective prompt (e.g.
        # switching the arxiv text source from title-only to TAPE title+abstract)
        # never stomp the baseline cache. Default "" preserves legacy filenames
        # bit-identically.
        self.cache_suffix = str(cache_suffix) if cache_suffix else ""
        user_suffix = "_{}".format(self.cache_suffix) if self.cache_suffix else ""

        # force_fresh: when True, annotate_nodes() ignores the in-memory / disk
        # cache when deciding what to query (every requested non-placeholder
        # node triggers a fresh API call) but NEW results are still written to
        # self._cache and persisted. Overwrites any prior entries for queried
        # nodes. For safety, callers that enable force_fresh should ALSO pass
        # cache_suffix so the overwritten cache lands in a new file.
        self.force_fresh = bool(force_fresh)

        if use_default_path:
            cache_filename = "{}_{}{}{}.json".format(
                self.dataset_name, safe_model, style_suffix, user_suffix,
            )
        else:
            cache_filename = "{}_{}_n{}_t{:.1f}{}{}.json".format(
                self.dataset_name,
                safe_model,
                self.n_samples,
                self.sampling_temperature,
                style_suffix,
                user_suffix,
            )
        self._cache_path = osp.join(cache_dir, cache_filename)
        self._load_disk_cache()

        # OpenAI client (lazy)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    # ── Client ──────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            if not self._api_key:
                raise ValueError(
                    "No OpenAI API key. Set OPENAI_API_KEY env var or pass api_key."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    # ── Disk cache management ───────────────────────────────

    def _load_disk_cache(self):
        # type: () -> None
        """Load our own annotation cache from disk."""
        if osp.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                raw = json.load(f)
            for node_str, (label, conf) in raw.items():
                self._cache[int(node_str)] = (int(label), float(conf))
            logger.info(
                "Loaded %d cached annotations from %s",
                len(self._cache),
                self._cache_path,
            )

    def _save_disk_cache(self):
        # type: () -> None
        """Persist annotation cache to disk."""
        os.makedirs(osp.dirname(self._cache_path), exist_ok=True)
        serializable = {
            str(k): [v[0], v[1]] for k, v in self._cache.items()
        }
        with open(self._cache_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # ── LoCLE cache loading ─────────────────────────────────

    def load_locle_cache(self, temperature=0.7, n=3, seed=42):
        # type: (float, int, int) -> Dict[int, Tuple[int, float]]
        """
        Load all annotations from LoCLE's pre-cached files.
        Merges into internal cache (LoCLE entries won't overwrite fresh annotations).
        """
        locle_data = _load_locle_cache(
            self.cache_dir, self.dataset_name, temperature, n, seed
        )
        added = 0
        for nid, (label, conf) in locle_data.items():
            if nid not in self._cache:
                self._cache[nid] = (label, conf)
                added += 1

        if added > 0:
            self._save_disk_cache()
            logger.info("Merged %d new entries from LoCLE cache", added)

        return dict(locle_data)

    # ── Response parsing dispatch ───────────────────────────

    def _parse(self, response_text):
        # type: (str) -> Tuple[int, float]
        """Dispatch to the right parser based on prompt_style."""
        if self.prompt_style == "llmgnn" and self.dataset_name in (
            "arxiv", "cora", "citeseer", "pubmed", "wikics", "dblp",
        ):
            # _parse_response_llmgnn expects a list-of-dicts topk response,
            # which is exactly what the generic non-arxiv topk prompt returns.
            return _parse_response_llmgnn(response_text, self.class_names)
        return _parse_response(response_text, self.class_names)

    # ── Single node query ───────────────────────────────────

    def _query_single(self, text, temperature=None):
        # type: (str, Optional[float]) -> Tuple[str, int, int]
        """Query the LLM for a single text. Returns (response, prompt_tokens, completion_tokens).

        If ``temperature`` is None, uses ``self.temperature`` (preserves existing
        behaviour when called with the default single argument).
        """
        client = self._get_client()
        prompt = _build_prompt(
            text, self.class_names, self.dataset_name, prompt_style=self.prompt_style,
        )
        temp = self.temperature if temperature is None else float(temperature)

        # LLM-GNN style: no structured JSON system prompt, no max_tokens=64.
        # Their reference impl sends ONLY the user prompt and expects a list
        # of python dicts back — truncating at 64 tokens would cut off the
        # list and break parsing.
        use_llmgnn = (
            self.prompt_style == "llmgnn"
            and self.dataset_name in (
                "arxiv", "cora", "citeseer", "pubmed", "wikics", "dblp",
            )
        )
        if use_llmgnn:
            messages = [{"role": "user", "content": prompt}]
            max_tokens = 256
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise text classifier. "
                        "Always respond with a valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            max_tokens = 64

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                usage = response.usage
                return (
                    response.choices[0].message.content or "",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("API call failed after %d retries: %s", self.max_retries, e)
                    raise

    def _query_n_samples(self, text, n, temperature):
        # type: (str, int, float) -> Tuple[List[str], int, int]
        """Query the LLM ``n`` times independently. Returns (responses, total_prompt_tokens, total_completion_tokens).

        For n=1 this is equivalent to a single ``_query_single`` call at the
        given temperature. For n>1 each call is a fresh API request; OpenAI's
        ``n`` parameter is deliberately NOT used because (a) LLM-GNN's reference
        implementation makes ``n`` independent calls via its async_api wrapper,
        and (b) counting separate calls keeps budget accounting transparent.
        """
        responses = []  # type: List[str]
        total_p = 0
        total_c = 0
        for _ in range(n):
            resp, p, c = self._query_single(text, temperature=temperature)
            responses.append(resp)
            total_p += p
            total_c += c
        return responses, total_p, total_c

    # ── Main annotation interface ───────────────────────────

    def annotate_nodes(self, node_ids, texts, class_names=None, force=False):
        # type: (List[int], List[str], Optional[List[str]], bool) -> Dict[int, Tuple[int, float]]
        """
        Annotate a batch of nodes via LLM.

        Args:
            node_ids:    Node indices to annotate.
            texts:       Corresponding raw texts (same length as node_ids).
            class_names: Override class names (uses self.class_names if None).
            force:       If True, re-annotate even if cached.

        Returns:
            {node_id: (label_index, confidence)} for all requested nodes.

        Raises:
            RuntimeError: If budget is exhausted.
        """
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)

        assert len(node_ids) == len(texts), (
            "node_ids ({}) and texts ({}) must match".format(len(node_ids), len(texts))
        )

        results = {}  # type: Dict[int, Tuple[int, float]]
        to_query = []  # type: List[Tuple[int, str]]
        placeholder_skipped = 0

        # self.force_fresh and the per-call `force` arg have the same effect:
        # bypass the cache lookup and force a fresh API call. New results are
        # still stored in self._cache (and overwrite any existing entries).
        effective_force = bool(force) or bool(self.force_fresh)
        for nid, text in zip(node_ids, texts):
            if not effective_force and nid in self._cache:
                results[nid] = self._cache[nid]
                continue
            # Skip empty / "node_{i}" placeholder rows: no real text means no
            # informative label and the API call would be wasted budget. These
            # nodes simply get no annotation (caller filters out label == -1).
            if _is_placeholder_text(text):
                placeholder_skipped += 1
                results[nid] = (-1, 0.0)
                # Do NOT write to self._cache so that if a real title is
                # supplied later the node can still be annotated.
                continue
            to_query.append((nid, text))

        if placeholder_skipped:
            logger.info(
                "Skipped %d placeholder/empty-text nodes (no API call, no cache write)",
                placeholder_skipped,
            )

        if not to_query:
            logger.info(
                "All %d nodes resolved without API calls (%d cached, %d placeholder)",
                len(node_ids), len(results) - placeholder_skipped, placeholder_skipped,
            )
            return results

        # Budget check. Two accounting modes:
        #   budget_per_node=False (default): budget counts individual API calls.
        #     With n_samples=k, each node costs k calls, so a budget of B allows
        #     floor(B / k) unique nodes.
        #   budget_per_node=True: budget counts n-sample sets (one per node),
        #     so budget=B allows B unique nodes = B*k API calls.
        n_samples = self.n_samples
        if self.budget is not None:
            if self.budget_per_node:
                # "units" here means nodes (n-sample sets). total_api_calls is still
                # in call units; we derive node-units as floor(calls / n_samples).
                nodes_used = self.total_api_calls // max(n_samples, 1)
                remaining_units = self.budget - nodes_used
                if remaining_units <= 0:
                    raise RuntimeError(
                        "LLM budget exhausted: {} nodes used / {} budget".format(
                            nodes_used, self.budget
                        )
                    )
                if len(to_query) > remaining_units:
                    logger.warning(
                        "Trimming query batch from %d to %d (budget limit, per-node mode)",
                        len(to_query),
                        remaining_units,
                    )
                    to_query = to_query[:remaining_units]
            else:
                remaining = self.budget - self.total_api_calls
                if remaining <= 0:
                    raise RuntimeError(
                        "LLM budget exhausted: {}/{} calls used".format(
                            self.total_api_calls, self.budget
                        )
                    )
                # Each node costs n_samples fresh calls; cap nodes so we don't
                # overflow the remaining call budget mid-node.
                max_nodes = remaining // max(n_samples, 1)
                if len(to_query) > max_nodes:
                    logger.warning(
                        "Trimming query batch from %d to %d (budget limit, per-call mode, "
                        "n_samples=%d)",
                        len(to_query),
                        max_nodes,
                        n_samples,
                    )
                    to_query = to_query[:max_nodes]

        logger.info(
            "Annotating %d nodes (%d from cache, %d to query, n_samples=%d, temp=%.2f)",
            len(node_ids),
            len(results),
            len(to_query),
            n_samples,
            self.sampling_temperature,
        )

        # Incremental-save interval so partial progress survives quota/rate-limit
        # crashes. Cache is written to disk every N nodes and on any exception.
        save_every = int(os.environ.get("NOISEPREF_CACHE_SAVE_EVERY", "10"))
        for i_nid, (nid, text) in enumerate(to_query):
            try:
                if n_samples == 1:
                    # Fast path — bit-identical to original behaviour.
                    response_text, p_tokens, c_tokens = self._query_single(
                        text, temperature=self.sampling_temperature
                    )
                    self.total_api_calls += 1
                    self.total_prompt_tokens += p_tokens
                    self.total_completion_tokens += c_tokens

                    label_idx, confidence = self._parse(response_text)

                    if label_idx == -1:
                        logger.warning(
                            "Failed to parse label for node %d, response: '%s'",
                            nid,
                            response_text[:200],
                        )
                        label_idx = 0
                        confidence = 0.0
                else:
                    # LLM-GNN consistency-voting path.
                    responses, p_tokens, c_tokens = self._query_n_samples(
                        text, n_samples, self.sampling_temperature
                    )
                    self.total_api_calls += n_samples
                    self.total_prompt_tokens += p_tokens
                    self.total_completion_tokens += c_tokens

                    votes = []
                    for resp in responses:
                        lbl, _ = self._parse(resp)
                        if lbl != -1:
                            votes.append(lbl)

                    if not votes:
                        logger.warning(
                            "All %d samples failed to parse for node %d; last='%s'",
                            n_samples,
                            nid,
                            responses[-1][:200] if responses else "",
                        )
                        label_idx = 0
                        confidence = 0.0
                    else:
                        counter = Counter(votes)
                        # most_common breaks ties by insertion order, which is
                        # first-occurrence order — deterministic for identical inputs.
                        label_idx, majority_count = counter.most_common(1)[0]
                        # Voting ratio is out of n_samples (NOT len(votes)) so that
                        # parse failures correctly reduce confidence.
                        confidence = (majority_count / float(n_samples)) * 100.0

                self._cache[nid] = (label_idx, confidence)
                results[nid] = (label_idx, confidence)

                # Periodic checkpoint — survives crashes.
                if (i_nid + 1) % save_every == 0:
                    self._save_disk_cache()
            except Exception as exc:
                # On ANY error (rate limit, quota, network), save partial cache
                # so a resumed run can skip already-annotated nodes. Re-raise so
                # the pipeline fails loudly.
                logger.error(
                    "annotate_nodes failed at i=%d/%d (node=%d): %s. "
                    "Saving partial cache (%d entries) and re-raising.",
                    i_nid, len(to_query), nid, exc, len(self._cache),
                )
                try:
                    self._save_disk_cache()
                except Exception:
                    pass
                raise

        self._save_disk_cache()

        logger.info(
            "Budget used: %d/%s calls | tokens: %d prompt + %d completion",
            self.total_api_calls,
            self.budget or "unlimited",
            self.total_prompt_tokens,
            self.total_completion_tokens,
        )

        return results

    # ── Cache-only retrieval ────────────────────────────────

    def get_cached(self, node_ids):
        # type: (List[int]) -> Dict[int, Tuple[int, float]]
        """Return cached annotations for given node_ids (no API calls)."""
        return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    def get_all_cached(self):
        # type: () -> Dict[int, Tuple[int, float]]
        """Return all cached annotations."""
        return dict(self._cache)

    # ── Named cache save/load (dataset x model keyed) ──────

    def save_annotations(self, dataset_name, model_name, annotations):
        # type: (str, str, Dict[int, Dict]) -> str
        """
        Save annotations to a named cache file.

        Args:
            dataset_name: Dataset identifier (e.g. 'cora').
            model_name:   Model identifier (e.g. 'gpt-4o-mini').
            annotations:  {node_id: {'label': int, 'confidence': float}}

        Returns:
            Path to the saved cache file.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_model = model_name.replace("/", "_")
        path = osp.join(self.cache_dir, "{}_{}.json".format(dataset_name, safe_model))

        serializable = {
            str(nid): [ann["label"], ann["confidence"]]
            for nid, ann in annotations.items()
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        # Also merge into internal cache
        for nid, ann in annotations.items():
            self._cache[int(nid)] = (int(ann["label"]), float(ann["confidence"]))

        logger.info("Saved %d annotations to %s", len(annotations), path)
        return path

    def load_cache(self, dataset_name, model_name):
        # type: (str, str) -> Dict[int, Dict]
        """
        Load annotations from a named cache file.

        Args:
            dataset_name: Dataset identifier.
            model_name:   Model identifier.

        Returns:
            {node_id: {'label': int, 'confidence': float}}
        """
        safe_model = model_name.replace("/", "_")
        path = osp.join(self.cache_dir, "{}_{}.json".format(dataset_name, safe_model))

        if not osp.exists(path):
            logger.warning("Cache file not found: %s", path)
            return {}

        with open(path, "r") as f:
            raw = json.load(f)

        result = {}
        for node_str, (label, conf) in raw.items():
            nid = int(node_str)
            result[nid] = {"label": int(label), "confidence": float(conf)}

        logger.info("Loaded %d annotations from %s", len(result), path)
        return result

    # ── Budget info ─────────────────────────────────────────

    @property
    def budget_remaining(self):
        # type: () -> Optional[int]
        if self.budget is None:
            return None
        return max(0, self.budget - self.total_api_calls)

    def budget_summary(self):
        # type: () -> Dict
        return {
            "total_api_calls": self.total_api_calls,
            "budget": self.budget,
            "budget_remaining": self.budget_remaining,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cached_annotations": len(self._cache),
        }

    def __repr__(self):
        return (
            "LLMAnnotator(dataset={!r}, model={!r}, "
            "calls={}/{}, cached={})".format(
                self.dataset_name,
                self.model,
                self.total_api_calls,
                self.budget or "inf",
                len(self._cache),
            )
        )


# ──────────────────────────────────────────────────────────────
# Self-test on Cora
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset

    data_dir = osp.join(repo_root, "data")
    locle_data_dir = osp.join(repo_root, "baselines", "locle", "data")
    data_path = data_dir if osp.exists(osp.join(data_dir, "cora_random_sbert.pt")) else locle_data_dir
    cache_dir = osp.join(repo_root, "data", "annotations")

    data = load_dataset("cora", data_path=data_path)
    print("Loaded Cora: {} nodes, {} classes".format(data.x.size(0), data.num_classes))

    # --- Test 1: Load LoCLE cache ---
    annotator = LLMAnnotator(
        dataset_name="cora",
        class_names=data.class_names,
        cache_dir=cache_dir,
        budget=280,
    )

    locle_results = annotator.load_locle_cache()
    if locle_results:
        print("\n[Test 1] LoCLE cache: {} annotations loaded".format(len(locle_results)))
        sample_ids = list(locle_results.keys())[:5]
        for nid in sample_ids:
            label, conf = locle_results[nid]
            gt = data.y[nid].item()
            match = "OK" if label == gt else "WRONG"
            print("  Node {}: predicted={} (conf={:.0f}), gt={}, {}".format(
                nid, data.class_names[label], conf, data.class_names[gt], match
            ))
        correct = sum(1 for nid, (l, _) in locle_results.items() if l == data.y[nid].item())
        print("  LoCLE cache accuracy: {}/{} = {:.3f}".format(
            correct, len(locle_results), correct / len(locle_results)
        ))
    else:
        print("\n[Test 1] No LoCLE cache found -- skipping")

    # --- Test 1.2: Cache round-trip ---
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ann2 = LLMAnnotator(dataset_name="test", class_names=["A", "B"], cache_dir=tmpdir)
        ann2.save_annotations("test_dataset", "test_model", {0: {"label": 1, "confidence": 0.9}})
        loaded = ann2.load_cache("test_dataset", "test_model")
        assert loaded[0]["label"] == 1
        assert loaded[0]["confidence"] == 0.9
        print("\n[Test 1.2] Cache round-trip: PASS")

    # --- Test 2: Cache-only annotation ---
    test_ids = list(range(10))
    cached = annotator.get_cached(test_ids)
    print("\n[Test 2] Cache lookup for nodes 0-9: {} found".format(len(cached)))

    # --- Test 3: Response parser ---
    print("\n[Test 3] Response parser tests:")
    test_cases = [
        ('{"label": "Neural_Networks", "confidence": 85}', 1, 85.0),
        ('{"label": "neural_networks", "confidence": 90}', 1, 90.0),
        ("The answer is Rule_Learning with 75% confidence", 0, 75.0),
        ("Probabilistic_Methods", 6, 50.0),
        ("completely irrelevant nonsense xyz", -1, 0.0),
    ]
    for text, expected_label, expected_conf in test_cases:
        label, conf = _parse_response(text, data.class_names)
        status = "PASS" if label == expected_label else "FAIL"
        print("  [{}] parse('{}...') -> label={} (expected {}), conf={}".format(
            status, text[:50], label, expected_label, conf
        ))

    # --- Test 3.1: Placeholder detection ---
    print("\n[Test 3.1] Placeholder detector tests:")
    placeholder_cases = [
        ("node_0", True),
        ("node_12345", True),
        ("  node_42  ", True),
        ("", True),
        ("   ", True),
        (None, True),
        ("Modeling distributed data exchange", False),
        ("node-42 is a real paper on graph theory", False),
    ]
    for text, expected in placeholder_cases:
        got = _is_placeholder_text(text)
        status = "PASS" if got == expected else "FAIL"
        print("  [{}] _is_placeholder_text({!r}) -> {} (expected {})".format(
            status, text, got, expected,
        ))

    # --- Test 4: Budget tracking ---
    print("\n[Test 4] Budget: {}".format(annotator.budget_summary()))
    print("  repr: {}".format(annotator))

    # --- Test 5: Fresh API annotation ---
    if os.environ.get("OPENAI_API_KEY"):
        print("\n[Test 5] Live API annotation (2 nodes)...")
        test_nodes = [0, 1]
        test_texts = [data.raw_texts[i] for i in test_nodes]
        results = annotator.annotate_nodes(test_nodes, test_texts, force=True)
        for nid, (label, conf) in results.items():
            gt = data.y[nid].item()
            print("  Node {}: predicted={} (conf={:.0f}), gt={}".format(
                nid, data.class_names[label], conf, data.class_names[gt]
            ))
        print("  Budget after: {}".format(annotator.budget_summary()))
    else:
        print("\n[Test 5] OPENAI_API_KEY not set -- skipping live API test")

    print("\nAll tests complete.")
