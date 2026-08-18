import json, os, random, argparse
from typing import Dict, Any, List, Optional, Tuple
import requests

FACT_RULES = (
    "FACT RULES (must follow):\n"
    "- span_m is the typical / representative span length. Mention it only as span length.\n"
    "- total_length_m is the overall bridge length. Mention it only as total length.\n"
    "- These two are independent. NEVER add or multiply spans to explain total length "
    "(do not write 'N spans of X m leading to / for a total length of Y m'). "
    "If both appear, state them as separate facts.\n"
    "- num_spans is the count of spans; do not compute length from it.\n"
    "- bridge_type is only 'slab' or 'girder'. slab = flat soffit, no discrete "
    "longitudinal members. girder = discrete beams/girders under the deck. "
    "Never write box girder, beam-slab, box_girder, or beam_slab.\n"
    "- Piers: has_piers says whether supports exist between the abutments. "
    "number_of_piers_along_length = pier lines along the bridge; "
    "number_of_piers_across_width = columns in a line; total_piers = total columns. "
    "Do not name pier styles (hammer-head, multicolumn, solid) or cross-sections "
    "(circular, rectangular).\n"
    "- Do not invent materials (no concrete, steel, etc.).\n"
    "- Do not invent purpose, abutments, railings, or other parts unless they are in the metadata.\n"
    "- Use only numbers and categories from the metadata. If a field is missing, skip it.\n"
    "- Lengths are already rounded to 0.1 m. Copy them as written "
    "(13.6 not 13.60; 5.2 not 5.237). Whole metres have no decimal (30 not 30.0).\n"
)


# === Build LLM prompt for simple descriptions ===
def build_simple_prompt(meta: Dict[str, Any]) -> str:
    """
    Asks LLM to generate a simple description conversation.
    """
    meta_llm = meta.copy()
    meta_llm.pop("id", None)
    meta_llm.pop("domain", None)

    brief_questions = [
        "Summarize the 3D point cloud object briefly.",
        "What kind of object is depicted by this point cloud?",
        "Provide a short explanation of this 3D structure.",
        "What does this collection of points represent?",
        "Offer a succinct summary of this 3D object.",
        "Can you give a brief overview of this point cloud?",
        "Characterize the object this point cloud is illustrating.",
        "Share a brief interpretation of this 3D point cloud.",
        "Provide an outline of this 3D shape's characteristics.",
        "What object is this point cloud rendering?",
        "Deliver a quick description of the object represented here.",
        "How would you describe the 3D form shown in this point cloud?",
        "What is the nature of the object this point cloud is representing?",
        "Present a compact account of this 3D object's key features.",
        "What can you infer about the object from this point cloud?",
        "Offer a clear and concise description of this point cloud object.",
        "How would you summarize this 3D data set?",
        "Give a brief explanation of the object that this cloud of points forms.",
        "What kind of structure does this 3D point cloud depict?",
        "Could you delineate the form indicated by this point cloud?",
        "Express in brief, what this point cloud is representing.",
        "Give a quick overview of the object represented by this 3D cloud.",
        "Convey a summary of the 3D structure represented in this point cloud.",
        "What kind of object is illustrated by this collection of points?",
        "Describe the object that this point cloud forms.",
        "How would you interpret this 3D point cloud?",
        "Can you briefly outline the shape represented by these points?",
        "Give a concise interpretation of the 3D data presented here.",
        "Explain the object this point cloud depicts succinctly.",
        "Offer a summary of the 3D object illustrated by this cloud."
    ]
    
    return (
        "You are a PointLLM-style assistant creating training conversations for 3D point cloud data of bridges.\n\n"
        "Based on the bridge metadata provided, generate a simple description in this EXACT JSON format:\n\n"
        "{\n"
        '  "question": "' + random.choice(brief_questions) + '",\n'
        '  "answer": "A concise 2-3 sentence description of the bridge point cloud data"\n'
        "}\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- Keep the answer concise (2-3 sentences, 40-60 words)\n"
        "- Mention key components the metadata supports (deck, spans, piers, slab vs girder)\n"
        "- Focus on what the point cloud captures, not on data quality\n"
        "- Use professional, clear language\n"
        "- Mention that it's a 3D representation/digital twin/LiDAR scan\n"
        "- Do not mention the words synthetic, real, generated, CadQuery, or HELIOS\n"
        f"{FACT_RULES}\n"
        f"BRIDGE METADATA:\n{json.dumps(meta_llm, indent=2, ensure_ascii=False)}"
    )

# === Build LLM prompt for complex instructions ===
def build_complex_prompt(meta: Dict[str, Any]) -> str:
    """
    Asks LLM to generate complex conversations in the final format directly.
    """
    tone = random.choice([
        "write in a factual yet vivid tone",
        "describe the object precisely but avoid redundancy",
        "use clear, professional phrasing appropriate for academic datasets"
    ])

    detailed_questions = [
        "Can you tell me more about this?",
        "What does this represent?",
        "Can you describe this in more detail?",
        "I'm interested in this, can you explain?",
        "Could you provide more info about this?",
        "What exactly am I looking at here?",
        "What is this?",
        "Could you describe the detailed structure of this?",
        "This looks interesting, can you expand on it?",
        "Can you explain more about this form?",
        "What can you tell me about the shape of this object?",
        "Could you delve deeper into this?",
        "I want to know more about this, can you help?",
        "Can you walk me through the details of this object?",
        "Can you provide a comprehensive account of this object?",
        "Offer a detailed interpretation of this point cloud.",
        "Please elucidate on the characteristics of this form.",
        "Could you provide an in-depth description of this structure?",
        "What does this cloud represent in its entirety?",
        "Elaborate on the details of this point cloud, please.",
        "Kindly furnish me with more information about this object.",
        "Please expand on the intricate structure of this form.",
        "Provide a meticulous explanation of what these points represent.",
        "I request a detailed breakdown of this structure.",
        "Give a thorough rundown of this point cloud.",
        "Can you offer a complete analysis of this object?",
        "I would like a comprehensive explanation of this form.",
        "Please detail the specific features of this point cloud.",
        "Could you elaborate extensively on what this represents?"
    ]

    meta_llm = meta.copy()
    meta_llm.pop("id", None)  # Remove id from metadata sent to LLM
    meta_llm.pop("domain", None)  # Remove domain from metadata sent to LLM

    return (
        "You are a PointLLM-style assistant creating training conversations for 3D point cloud data of bridges.\n\n"
        "Based on the bridge metadata provided, generate conversations in this EXACT JSON format:\n\n"
        "{\n"
        '  "detailed_description": {\n'
        '    "question": "' + random.choice(detailed_questions) + '",\n'
        '    "answer": "A detailed 50-100 word description of the bridge covering geometry, structure, and dimensions"\n'
        '  },\n'
        '  "single_round": [\n'
        '    {\n'
        '      "question": "A specific question about one aspect of the bridge",\n'
        '      "answer": "A focused answer to that question"\n'
        '    }\n'
        '    // Generate 3 different single-round Q&A pairs\n'
        '  ],\n'
        '  "multi_round": {\n'
        '    "rounds": [\n'
        '      {\n'
        '        "question": "Initial question starting the conversation",\n'
        '        "answer": "Answer to the first question"\n'
        '      },\n'
        '      {\n'
        '        "question": "Follow-up question building on the previous answer",\n'
        '        "answer": "Answer continuing the conversation"\n'
        '      },\n'
        '      {\n'
        '        "question": "Final question deepening the conversation",\n'
        '        "answer": "Comprehensive final answer"\n'
        '      }\n'
        '    ]\n'
        '  }\n'
        "}\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- DO NOT mention data quality issues (occlusion, sparsity, missing scan parts)\n"
        "- Describe only the bridge itself — geometry, structure, and dimensions from the metadata\n"
        "- For single_round: Generate 3 different Q&A pairs about geometry, spans, total length, "
        "pier counts, width, and slab vs girder. Keep span length and total length in separate questions "
        "when both are discussed. Do not ask about pier style, box cells, or materials.\n"
        "- For multi_round: Generate 3 rounds with questions showing logical progression and depth. "
        "Do not ask about materials unless the metadata lists them.\n"
        "- Use clear, professional technical language\n"
        "- Include specific numeric values from metadata in answers\n"
        "- Do not mention the words synthetic, real, generated, CadQuery, or HELIOS\n"
        f"- {tone}\n"
        f"{FACT_RULES}\n"
        f"BRIDGE METADATA:\n{json.dumps(meta_llm, indent=2, ensure_ascii=False)}"
    )

DEFAULT_SYNTH_SUMMARY = (
    "/media/syedsalman/salmandrive/SyntheticBridgeDatasetGeneration/"
    "MultiModalDatasetBridges/Dataset_23.7.26/bridge_summary.json"
)
DEFAULT_REAL_DIR = "/media/syedsalman/salmandrive/Dataset/RealPointClouds/measurements/out"
DEFAULT_OUTPUT = "/media/syedsalman/salmandrive/Dataset/MixedPointCloudv2/annotations"
ENV_CANDIDATES = [
    "/media/syedsalman/salmandrive/BridgeMLLM/.env",
]

# Mixed-training fields only — what a scan can show, same keys for real and synth.
# CAD classes (box_girder, hammer_head, circular/rectangular) stay out.
CAPTION_FIELDS = (
    "bridge_type",
    "num_spans",
    "span_m",
    "total_length_m",
    "width_m",
    "lanes",
    "depth_of_girder",
    "clearance_height_m",
    "has_piers",
    "number_of_piers_along_length",
    "number_of_piers_across_width",
    "total_piers",
)
CAPTION_ALIASES = {
    "depth_of_girder": ("depth_of_girder", "deck_structural_depth_m"),
    "clearance_height_m": ("clearance_height_m", "bridge_clearance_height"),
}

# Generator / heuristic names → observable soffit class.
_TYPE_TO_GIRDER = {"beam_slab", "box_girder", "girder", "plattenbalken"}
_TYPE_TO_SLAB = {"slab", "plattenbruecke", "plattenbrücke"}

# === Model pricing (per 1M tokens) ===
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "google/gemini-2.0-pro-exp": {"input": 1.25, "output": 5.00},
}

total_cost = 0.0
total_input_tokens = 0
total_output_tokens = 0
SELECTED_MODELS = ["openai/gpt-4o-mini", "openai/gpt-4o"]


def load_env() -> None:
    """Load OPENROUTER_API_KEY from BridgeMLLM/.env if it is not already set."""
    if os.getenv("OPENROUTER_API_KEY"):
        return
    for path in ENV_CANDIDATES:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val
        break


COUNT_FIELDS = {
    "num_spans",
    "lanes",
    "number_of_piers_along_length",
    "number_of_piers_across_width",
    "total_piers",
}


def _round(value: Any, field: Optional[str] = None) -> Any:
    """One decimal for metres, ints for counts — same look for real and synth."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, list):
        return [_round(v, field) for v in value]
    if field in COUNT_FIELDS:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    if isinstance(value, (int, float)):
        rounded = round(float(value), 1)
        return int(rounded) if rounded == int(rounded) else rounded
    return value


def _pick(src: Dict[str, Any], field: str) -> Any:
    for key in CAPTION_ALIASES.get(field, (field,)):
        if key in src and src[key] is not None:
            return src[key]
    return None


GIRDER_SHARE_MIN = 0.30


def caption_bridge_type(src: Dict[str, Any]) -> Optional[str]:
    """slab vs girder from what the cloud can show, not CAD hollowness.

    Real: a girder share above GIRDER_SHARE_MIN means discrete beams are really
    there. A few stray girder-labelled points are not enough — japan_05 sits at
    17% girder against 66% deck with a 0.22 m depth over a 12 m span, which is a
    slab whose soffit got mislabelled, so it falls through to the measured type.
    Synth: beam_slab and box_girder both → girder.
    """
    meta = src.get("_meta") if isinstance(src.get("_meta"), dict) else {}
    counts = meta.get("class_counts") if isinstance(meta.get("class_counts"), dict) else {}
    structure = sum(int(v or 0) for k, v in counts.items() if k != "other")
    if structure > 0:
        share = int(counts.get("girder") or 0) / structure
        if share >= GIRDER_SHARE_MIN:
            return "girder"
    raw = str(src.get("bridge_type") or "").strip().lower()
    if raw in _TYPE_TO_GIRDER:
        return "girder"
    if raw in _TYPE_TO_SLAB:
        return "slab"
    return None


def to_caption_meta(object_id: str, src: Dict[str, Any]) -> Dict[str, Any]:
    """One schema for both domains. Drops nulls so missing facts are skipped."""
    meta: Dict[str, Any] = {"id": str(object_id), "domain": "bridges"}
    src = dict(src)
    if src.get("has_piers") is None:
        src["has_piers"] = int(src.get("total_piers") or 0) > 0
    mapped_type = caption_bridge_type(src)
    if mapped_type is not None:
        src["bridge_type"] = mapped_type
    for field in CAPTION_FIELDS:
        value = _pick(src, field)
        if value is None or value == []:
            continue
        meta[field] = _round(value, field)
    if mapped_type is not None:
        meta["bridge_type"] = mapped_type
    return meta


def meta_from_synthetic(bridge_data: Dict[str, Any]) -> Dict[str, Any]:
    """Map Dataset_23.7.26 bridge_summary.json rows to caption metadata."""
    return to_caption_meta(str(bridge_data.get("bridge_id")), bridge_data)


def meta_from_real(bridge_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Map BridgeBank measurements/out/<id>/parameters.json to the same fields."""
    return to_caption_meta(bridge_id, params)


def load_real_bridges(measurements_dir: str) -> List[Dict[str, Any]]:
    """Load Tier A/B metadata from measurements/out/<id>/parameters.json."""
    records = []
    if not os.path.isdir(measurements_dir):
        raise FileNotFoundError(f"Real measurements dir not found: {measurements_dir}")
    for name in sorted(os.listdir(measurements_dir)):
        path = os.path.join(measurements_dir, name, "parameters.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        records.append(meta_from_real(name, params))
    return records


# === OpenRouter API Call ===
def call_openrouter(prompt: str) -> str:
    """
    Call OpenRouter API with best models for technical dataset generation.
    """
    global total_cost, total_input_tokens, total_output_tokens
    
    load_env()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    models = list(dict.fromkeys(SELECTED_MODELS))
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    last_err = None
    for model in models:
        try:
            print(f"Using model: {model}")
            
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "PointLLM Bridge Dataset Generator"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 3000,  # Generous limit for complete responses
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                
                if text and text.strip():
                    # Print usage statistics and calculate cost
                    if "usage" in result:
                        usage = result["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        
                        # Calculate cost for this call
                        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
                        call_cost = (prompt_tokens / 1_000_000 * pricing["input"]) + \
                                   (completion_tokens / 1_000_000 * pricing["output"])
                        
                        # Update global tracking
                        total_input_tokens += prompt_tokens
                        total_output_tokens += completion_tokens
                        total_cost += call_cost
                        
                        print(f"✓ Success: {prompt_tokens} in + {completion_tokens} out = {prompt_tokens + completion_tokens} tokens | Cost: ${call_cost:.4f}")
                    return text
                else:
                    print(f"⚠️ Empty response from {model}")
                    last_err = RuntimeError(f"Empty response from {model}")
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("error", {}).get("message", response.text[:200])
                print(f"⚠️ HTTP {response.status_code} from {model}: {error_msg}")
                last_err = RuntimeError(f"{model} failed with status {response.status_code}: {error_msg}")
                
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout after 120s for {model}, trying next model...")
            last_err = RuntimeError(f"Timeout for {model}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error for {model}: {e}")
            last_err = e
        except Exception as e:
            print(f"⚠️ Unexpected error for {model}: {e}")
            last_err = e
    
    raise RuntimeError(f"All models failed. Last error: {last_err}")

# === Generate simple description ===
def generate_simple_description(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a simple description conversation.
    """
    raw = call_openrouter(build_simple_prompt(meta))
    
    # Parse JSON from response
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except json.JSONDecodeError:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                start = raw.find("```json") + 7
                end = raw.find("```", start)
                data = json.loads(raw[start:end].strip())
            else:
                debug_file = f"debug_simple_{meta['id']}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(raw)
                raise RuntimeError(f"JSON parse failed. Raw saved to {debug_file}")
    
    # Validate structure
    if "question" not in data or "answer" not in data:
        raise RuntimeError(f"Missing 'question' or 'answer' in LLM output. Got keys: {list(data.keys())}")
    
    # Return in final format
    return {
        "object_id": meta["id"],
        "conversation_type": "simple_description",
        "conversations": [
            {
                "from": "human",
                "value": (
                    data["question"]
                    if str(data["question"]).startswith("<point>")
                    else f"<point>\n{data['question']}"
                )
            },
            {
                "from": "gpt",
                "value": data["answer"]
            }
        ]
    }

# === Main pipeline for complex instructions ===
def generate_complex_instructions(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate complex instruction conversations.
    """
    raw = call_openrouter(build_complex_prompt(meta))
    
    # Parse JSON from response
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except json.JSONDecodeError:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                start = raw.find("```json") + 7
                end = raw.find("```", start)
                data = json.loads(raw[start:end].strip())
            else:
                debug_file = f"debug_response_{meta['id']}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(raw)
                raise RuntimeError(f"JSON parse failed. Raw saved to {debug_file}")
    
    # Validate structure
    required_keys = ["detailed_description", "single_round", "multi_round"]
    for k in required_keys:
        if k not in data:
            raise RuntimeError(f"Missing '{k}' in LLM output. Got keys: {list(data.keys())}")
    
    # Transform to final format
    conversations = []
    object_id = meta["id"]
    
    # 1. Detailed description
    desc = data["detailed_description"]
    conversations.append({
        "object_id": object_id,
        "conversation_type": "detailed_description",
        "conversations": [
            {
                "from": "human",
                "value": f"<point>\n{desc.get('question', 'Can you describe what this point cloud represents?')}"
            },
            {
                "from": "gpt",
                "value": desc["answer"]
            }
        ]
    })
    
    # 2. Single-round conversations
    rounds = data["single_round"]
    if isinstance(rounds, dict):
        rounds = [rounds]
    for qa in rounds:
        conversations.append({
            "object_id": object_id,
            "conversation_type": "single_round",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{qa['question']}"
                },
                {
                    "from": "gpt",
                    "value": qa["answer"]
                }
            ]
        })
    
    # 3. Multi-round conversation
    multi = data["multi_round"]
    multi_conv = []
    for i, round_data in enumerate(multi.get("rounds", [])):
        prefix = "<point>\n" if i == 0 else ""  # Only first question has <point>
        multi_conv.append({
            "from": "human",
            "value": f"{prefix}{round_data['question']}"
        })
        multi_conv.append({
            "from": "gpt",
            "value": round_data["answer"]
        })
    
    if multi_conv:
        conversations.append({
            "object_id": object_id,
            "conversation_type": "multi_round",
            "conversations": multi_conv
        })
    
    return conversations

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simple descriptions and complex instructions for bridge point clouds"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_SYNTH_SUMMARY,
        help="Synthetic bridge_summary.json (default: Dataset_23.7.26)",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=DEFAULT_REAL_DIR,
        help="BridgeBank measurements/out folder with <id>/parameters.json",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "real", "all"],
        default="all",
        help="Which bridges to caption (default: all = synth + real)",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=None,
        help="Number of bridges to process (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory for the two annotation JSON files",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter model id (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing annotations instead of resuming from them",
    )
    args = parser.parse_args()

    SELECTED_MODELS[:] = [args.model] + [m for m in SELECTED_MODELS if m != args.model]

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    simple_output = os.path.join(out_dir, "bridge_simple_descriptions.json")
    complex_output = os.path.join(out_dir, "bridge_complex_instructions.json")

    metas: List[Dict[str, Any]] = []
    if args.source in ("synthetic", "all"):
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            raise SystemExit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            synth_rows = json.load(f)
        metas.extend(meta_from_synthetic(row) for row in synth_rows)
    if args.source in ("real", "all"):
        metas.extend(load_real_bridges(args.real_dir))

    if args.count is not None:
        metas = metas[: args.count]
        print(f"Processing {len(metas)} bridges (limited by --count)...")
    else:
        print(f"Processing all {len(metas)} bridges...")

    def load_existing(path: str) -> List[Dict[str, Any]]:
        """Reuse finished bridges so a restart never re-pays for them."""
        if args.overwrite or not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Ignoring unreadable {path}, starting that file fresh")
            return []
        return data if isinstance(data, list) else []

    simple_descriptions: List[Dict[str, Any]] = load_existing(simple_output)
    complex_instructions: List[Dict[str, Any]] = load_existing(complex_output)
    done_simple = {r.get("object_id") for r in simple_descriptions}
    done_complex = {r.get("object_id") for r in complex_instructions}
    if done_simple or done_complex:
        print(
            f"Resuming: {len(done_simple)} simple and {len(done_complex)} complex "
            "bridges already done"
        )
    errors: List[str] = []

    def save_outputs() -> None:
        # tmp + replace, because an interrupt during the write would leave a
        # truncated file that the next run could not resume from
        for path, payload in ((simple_output, simple_descriptions),
                              (complex_output, complex_instructions)):
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
            os.replace(tmp, path)

    for meta in metas:
        need_simple = meta["id"] not in done_simple
        need_complex = meta["id"] not in done_complex
        if not need_simple and not need_complex:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {meta['id']}...")
        print(f"{'='*60}")

        if need_simple:
            try:
                print("Generating simple description...")
                simple_descriptions.append(generate_simple_description(meta))
                print(f"Generated simple description for {meta['id']}")
            except Exception as e:
                print(f"Failed to generate simple description: {e}")
                errors.append(f"[bridge {meta['id']}] simple description: {e}")

        if need_complex:
            try:
                print("Generating complex instructions...")
                complex_convs = generate_complex_instructions(meta)
                complex_instructions.extend(complex_convs)
                print(f"Generated {len(complex_convs)} complex conversations for {meta['id']}")
            except Exception as e:
                print(f"Failed to generate complex instructions: {e}")
                errors.append(f"[bridge {meta['id']}] complex instructions: {e}")

        save_outputs()

    if errors:
        error_log_path = os.path.join(out_dir, "caption_generation_errors.txt")
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"Logged {len(errors)} error(s) to {error_log_path}")

    print(f"\n{'='*60}")
    print(f"Saved {len(simple_descriptions)} simple descriptions to {simple_output}")
    print(f"Saved {len(complex_instructions)} complex instructions to {complex_output}")
    print(f"{'='*60}")
    print(f"\nComplete! Total conversations: {len(simple_descriptions) + len(complex_instructions)}")
    print("COST SUMMARY")
    print(f"Total Input Tokens:  {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total Cost:          ${total_cost:.4f}")
   