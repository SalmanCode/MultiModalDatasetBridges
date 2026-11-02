"""PointLLM bridge captions.

scale_mode=none: questions are unmodified.
scale_mode=prompt: every <point> question gets the scan's normalisation radius:

    The scan was centered and divided by a radius of {m:.1f} metres.
"""
import json, os, random, argparse, re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
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
    "longitudinal members. girder = discrete beams/girders under the deck.\n"
    "- Piers: has_piers says whether supports exist between the abutments. "
    "number_of_piers_along_length = pier lines along the bridge; "
    "number_of_piers_across_width = columns in a line; total_piers = total columns. "
    "pier_type is multicolumn or wall; column_shape is cylindrical, rectangular "
    "or trapezoidal. Name them only when the metadata carries them.\n"
    "- Do not invent materials (no concrete, steel, etc.).\n"
    "- Do not invent purpose, abutments, railings, or other parts unless they are in the metadata.\n"
    "- Use only numbers and categories from the metadata. If a field is missing, skip it.\n"
    "- Lengths are already rounded to 0.1 m. Copy them as written "
    "(13.6 not 13.60; 5.2 not 5.237; 30.0 not 30).\n"
    "- Do not mention a normalization radius, unit sphere, or scan scaling. "
    "Those are added to the question separately and are not bridge dimensions.\n"
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SYNTH_SUMMARY = os.path.join(REPO_ROOT, "Dataset", "bridge_summary.json")
DEFAULT_SCANS_DIR = os.path.join(REPO_ROOT, "Dataset", "PointCloudScans")
DEFAULT_REAL_DIR = os.environ.get("REAL_BRIDGE_DIR")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "Dataset", "annotations")
# Identical wording for synthetic and real bridges.
SCALE_SENTENCE = (
    "The scan was centered and divided by a radius of {m:.1f} metres."
)
ENV_CANDIDATES = [
    os.path.join(REPO_ROOT, ".env"),
]

# fields used in captions, same for real and synthetic (lanes left out)
CAPTION_FIELDS = (
    "bridge_type",
    "num_spans",
    "span_m",
    "total_length_m",
    "width_m",
    "depth_of_girder",
    "clearance_height_m",
    "has_piers",
    "number_of_piers_along_length",
    "number_of_piers_across_width",
    "total_piers",
    "pier_type",
    "column_shape",
)
# caption field -> possible key names in real and synthetic records
CAPTION_ALIASES = {
    "depth_of_girder": ("depth_of_girder", "deck_structural_depth_m"),
    "clearance_height_m": (
        "clearance_height_m", "bridge_clearance_height", "bridge_clearance_height_m",
    ),
    # interior span
    "span_m": ("span_m", "pier_span_m"),
    "pier_type": ("pier_type", "type_of_piers"),
    "column_shape": ("column_shape", "pier_cross_section"),
    "number_of_piers_along_length": (
        "number_of_piers_along_length", "number_of_piers_across_length",
    ),
    "number_of_piers_across_width": (
        "number_of_piers_across_width", "piercolumns_across_width",
    ),
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
    """Load OPENROUTER_API_KEY from the repo's .env if it is not already set."""
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
        return round(float(value), 1)
    return value


def _pick(src: Dict[str, Any], field: str) -> Any:
    for key in CAPTION_ALIASES.get(field, (field,)):
        if key in src and src[key] is not None:
            return src[key]
    return None


GIRDER_SHARE_MIN = 0.30


def caption_bridge_type(src: Dict[str, Any]) -> Optional[str]:
    """slab or girder deck type."""
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
        if field == "clearance_height_m" and float(value) < 0:
            # Negative clearance is a measurement artefact, not a bridge dimension.
            continue
        meta[field] = _round(value, field)
    if mapped_type is not None:
        meta["bridge_type"] = mapped_type
    return meta


def meta_from_synthetic(bridge_data: Dict[str, Any]) -> Dict[str, Any]:
    """Map Dataset/bridge_summary.json rows to caption metadata."""
    return to_caption_meta(str(bridge_data.get("bridge_id")), bridge_data)


def meta_from_real(bridge_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Map BridgeBank measurements/out/<id>/parameters.json to the same fields."""
    return to_caption_meta(bridge_id, params)


# fixed caption sentences and questions
RATIO_EVAL_QUESTION = "What does this cloud represent in its entirety?"

Q_SIMPLE = "Provide a short description of this 3D structure."
Q_LENGTH = "What is the total length of the bridge?"
Q_WIDTH = "What is the width of the bridge?"
Q_SPAN = "What is the span length of the bridge?"
Q_NUM_SPANS = "How many spans does the bridge have?"
Q_PIERS = "How many piers does the bridge have?"
Q_PIER_TYPE = "What type of piers does the bridge have?"
Q_DIMENSIONS = "Can you describe the overall dimensions of the bridge?"


def _m(value: Any) -> str:
    """One decimal, always 'meters'."""
    return f"{float(value):.1f} meters"


def _n(value: Any) -> int:
    return int(round(float(value)))


def _sentences(meta: Dict[str, Any]) -> Dict[str, str]:
    """The canonical sentence for each quantity this bridge actually has."""
    out: Dict[str, str] = {}
    btype = meta.get("bridge_type")
    if btype in ("slab", "girder"):
        out["type"] = f"This is a {btype} bridge."
    if meta.get("total_length_m") is not None:
        out["length"] = f"The total length of the bridge is {_m(meta['total_length_m'])}."
    if meta.get("width_m") is not None:
        out["width"] = f"The width of the bridge is {_m(meta['width_m'])}."
    if meta.get("span_m") is not None:
        out["span"] = f"The span length of the bridge is {_m(meta['span_m'])}."
    # Plural agreement is the one thing allowed to vary.
    if meta.get("num_spans") is not None:
        n = _n(meta["num_spans"])
        out["num_spans"] = f"The bridge has {n} span{'' if n == 1 else 's'}."
    piers = _n(meta.get("total_piers") or 0)
    out["piers"] = f"The bridge has {piers} pier{'' if piers == 1 else 's'}."
    if piers:
        if meta.get("pier_type"):
            out["pier_type"] = f"The piers are {meta['pier_type']} piers."
        if meta.get("column_shape"):
            out["column_shape"] = f"The pier columns are {meta['column_shape']} in shape."
        if meta.get("number_of_piers_along_length") is not None:
            rows = _n(meta["number_of_piers_along_length"])
            out["pier_rows"] = f"The bridge has {rows} row{'' if rows == 1 else 's'} of piers."
        if meta.get("number_of_piers_across_width") is not None:
            cols = _n(meta["number_of_piers_across_width"])
            out["pier_columns"] = f"Each row has {cols} column{'' if cols == 1 else 's'}."
    if meta.get("clearance_height_m") is not None:
        out["clearance"] = (
            f"The clearance height of the bridge is {_m(meta['clearance_height_m'])}."
        )
    return out


def _qa(object_id: str, ctype: str, turns: List[Tuple[str, str]]) -> Dict[str, Any]:
    convs: List[Dict[str, str]] = []
    for i, (q, a) in enumerate(turns):
        convs.append({"from": "human", "value": (f"<point>\n{q}" if i == 0 else q)})
        convs.append({"from": "gpt", "value": a})
    return {"object_id": object_id, "conversation_type": ctype, "conversations": convs}


def template_caption_records(
    meta: Dict[str, Any], radius_m: Optional[float] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Templated captions with fixed questions and answers, no LLM."""
    bid = str(meta["id"])
    s = _sentences(meta)

    def joined(keys: Tuple[str, ...]) -> str:
        return " ".join(s[k] for k in keys if k in s)

    simple = _qa(bid, "simple_description", [(
        Q_SIMPLE,
        joined(("type", "length", "width", "num_spans", "piers")),
    )])

    complex_records = [
        _qa(bid, "detailed_description", [(
            RATIO_EVAL_QUESTION,
            joined((
                "type", "length", "width", "span", "num_spans", "piers",
                "pier_type", "column_shape", "pier_rows", "pier_columns",
                "clearance",
            )),
        )]),
    ]

    for question, key in (
        (Q_LENGTH, "length"),
        (Q_WIDTH, "width"),
        (Q_SPAN, "span"),
        (Q_NUM_SPANS, "num_spans"),
        (Q_PIERS, "piers"),
    ):
        if key in s:
            complex_records.append(_qa(bid, "single_round", [(question, s[key])]))

    if "pier_type" in s:
        complex_records.append(_qa(bid, "single_round", [(
            Q_PIER_TYPE, joined(("pier_type", "column_shape")),
        )]))

    multi: List[Tuple[str, str]] = [(Q_DIMENSIONS, joined(("length", "width")))]
    if "piers" in s:
        multi.append((Q_PIERS, joined(("piers", "pier_type", "column_shape"))))
    if "num_spans" in s:
        multi.append((Q_NUM_SPANS, joined(("num_spans", "span"))))
    complex_records.append(_qa(bid, "multi_round", multi))

    if radius_m is not None:
        simple = apply_scale_to_record(simple, float(radius_m), "both")
        complex_records = [
            apply_scale_to_record(rec, float(radius_m), "both") for rec in complex_records
        ]
    return simple, complex_records


def real_record(measurements_dir: str, bridge_id: str) -> Optional[Dict[str, Any]]:
    """One real bridge from primary_parameter.json (depth from parameters.json)."""
    folder = os.path.join(measurements_dir, bridge_id)
    hand_path = os.path.join(folder, "primary_parameter.json")
    if not os.path.isfile(hand_path):
        return None
    with open(hand_path, encoding="utf-8") as f:
        hand = json.load(f)

    src: Dict[str, Any] = dict(hand)

    # girder: true marks a T-beam deck
    src["bridge_type"] = "girder" if hand.get("girder") else "slab"

    # total piers from the layout
    lines = hand.get("number_of_piers_across_length") or 0
    across = hand.get("piercolumns_across_width")
    src["total_piers"] = lines if hand.get("type_of_piers") == "wall" else lines * (across or 0)

    measured_path = os.path.join(folder, "parameters.json")
    if os.path.isfile(measured_path):
        with open(measured_path, encoding="utf-8") as f:
            measured = json.load(f)
        if measured.get("deck_structural_depth_m") is not None:
            src.setdefault("deck_structural_depth_m", measured["deck_structural_depth_m"])

    return meta_from_real(bridge_id, src)


def load_real_bridges(measurements_dir: str) -> List[Dict[str, Any]]:
    """Every real bridge that carries a hand annotation."""
    if not os.path.isdir(measurements_dir):
        raise FileNotFoundError(f"Real measurements dir not found: {measurements_dir}")
    records = []
    for name in sorted(os.listdir(measurements_dir)):
        if not os.path.isdir(os.path.join(measurements_dir, name)):
            continue
        rec = real_record(measurements_dir, name)
        if rec is not None:
            records.append(rec)
    return records


def _merged_xyz_path(scans_dir: str, object_id: str) -> Optional[str]:
    """PointCloudScans/merged/TLS_<id>/<id>_complete.xyz"""
    if not scans_dir:
        return None
    path = os.path.join(scans_dir, "merged", f"TLS_{object_id}", f"{object_id}_complete.xyz")
    return path if os.path.isfile(path) else None


def pc_norm_radius(xyz: np.ndarray) -> Tuple[float, List[float]]:
    """Same radius convert_to_npy.pc_norm divides by: centroid, then max ||p||."""
    centroid = xyz.mean(axis=0)
    radius = float(np.max(np.linalg.norm(xyz - centroid, axis=1)))
    return max(radius, 1e-6), [float(v) for v in centroid]


def compute_norm_radius_from_xyz(xyz_path: str) -> Tuple[float, List[float]]:
    """Load metric XYZ (HELIOS x y z ...) and return (norm_radius_m, centroid_m)."""
    xyz = np.loadtxt(xyz_path, usecols=(0, 1, 2), dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz.reshape(1, 3)
    if xyz.size == 0:
        raise ValueError(f"No points in {xyz_path}")
    return pc_norm_radius(xyz)


def load_scales_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_scales_json(path: str, scales: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scales, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def collect_scales(
    object_ids: List[str],
    scans_dir: Optional[str],
    real_dir: Optional[str],
    cache_paths: List[str],
) -> Dict[str, Dict[str, Any]]:
    """norm_radius_m per object: cache → real parameters.json → merged XYZ."""
    scales: Dict[str, Dict[str, Any]] = {}
    for path in cache_paths:
        scales.update(load_scales_json(path))

    # A real bridge's radius always comes from its parameters.json, cache or no
    # cache. The cache exists to avoid re-reading multi-GB merged .xyz scans for
    # the synthetic side; re-reading 21 small JSON files costs nothing. Letting a
    # stale cache win here is how a corrected measurement would silently fail to
    # reach the captions -- which is exactly what would have happened after the
    # ground-included norm_radius_m fix.
    n_refreshed = 0
    if real_dir:
        for object_id in object_ids:
            params_path = os.path.join(real_dir, object_id, "parameters.json")
            if not os.path.isfile(params_path):
                continue
            with open(params_path, encoding="utf-8") as f:
                params = json.load(f)
            radius = params.get("norm_radius_m")
            if radius is None:
                continue
            cached = (scales.get(object_id) or {}).get("norm_radius_m")
            if cached is not None and abs(float(cached) - float(radius)) > 1e-6:
                print(f"  {object_id}: cached norm_radius_m {float(cached):.3f} m is stale, "
                      f"using {float(radius):.3f} m from parameters.json")
                n_refreshed += 1
            scales[object_id] = {
                "norm_radius_m": float(radius),
                "centroid_m": params.get("centroid_m"),
                "norm_basis": params.get("norm_basis"),
                "source": "parameters.json",
            }
    if n_refreshed:
        print(f"Refreshed {n_refreshed} stale real radius/radii from parameters.json")

    missing = [oid for oid in object_ids if oid not in scales or scales[oid].get("norm_radius_m") is None]
    n_compute = 0
    for object_id in missing:
        record: Optional[Dict[str, Any]] = None
        if record is None and scans_dir:
            xyz_path = _merged_xyz_path(scans_dir, object_id)
            if xyz_path:
                print(f"Computing norm_radius_m from {xyz_path} ...")
                radius, centroid = compute_norm_radius_from_xyz(xyz_path)
                record = {
                    "norm_radius_m": radius,
                    "centroid_m": centroid,
                    "source": xyz_path,
                }
                n_compute += 1
        if record is not None:
            scales[object_id] = record

    if n_compute:
        print(f"Computed norm_radius_m from {n_compute} merged XYZ scan(s)")
    return scales


def scale_sentence(radius_m: float) -> str:
    return SCALE_SENTENCE.format(m=float(radius_m))


def inject_scale_sentence(text: str, radius_m: float) -> str:
    """Insert the L1 sentence after <point>\\n. Idempotent. Leaves answers alone."""
    if not text or radius_m is None:
        return text
    sentence = scale_sentence(radius_m)
    if sentence in text:
        return text
    match = re.match(r"^(<point>\s*)", text)
    if not match:
        return text
    rest = text[match.end():]
    if rest.startswith(sentence):
        return text
    spacer = "" if rest.startswith("\n") else " "
    return f"{match.group(1)}{sentence}{spacer}{rest}"


def apply_scale_to_record(
    record: Dict[str, Any],
    radius_m: float,
    apply_to: str,
) -> Dict[str, Any]:
    """Prefix human <point> turns. apply_to: both | complex | simple."""
    conv_type = record.get("conversation_type")
    if apply_to == "simple" and conv_type != "simple_description":
        return record
    if apply_to == "complex" and conv_type == "simple_description":
        return record
    for turn in record.get("conversations") or []:
        if turn.get("from") != "human":
            continue
        turn["value"] = inject_scale_sentence(str(turn.get("value", "")), radius_m)
    return record


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
        description=(
            "Generate PointLLM captions for synthetic/real bridges. "
            "scale_mode=prompt adds the L1 radius sentence to human <point> turns."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_SYNTH_SUMMARY,
        help="Synthetic bridge_summary.json (default: Dataset/bridge_summary.json)",
    )
    parser.add_argument(
        "--scans-dir",
        type=str,
        default=DEFAULT_SCANS_DIR,
        help="PointCloudScans folder with merged/TLS_<id>/<id>_complete.xyz",
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
        default="synthetic",
        help="Which bridges to caption (default: synthetic)",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=None,
        help="Number of bridges to process (default: all with a known scale)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated object ids to process (e.g. bridge_1,bridge_2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory for annotation JSON + scales.json",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["none", "prompt"],
        default="prompt",
        help="none = L0 captions; prompt = L1 radius sentence in questions (default)",
    )
    parser.add_argument(
        "--apply-scale-to",
        choices=["both", "complex", "simple"],
        default="both",
        help="Which files get the L1 sentence (default: both). Use complex for Stage-2-only L1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metadata + radius + example L1 question; do not call the LLM",
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
    parser.add_argument(
        "--template",
        action="store_true",
        help="Write deterministic captions (length, width, pier count) instead of calling the LLM",
    )
    args = parser.parse_args()

    SELECTED_MODELS[:] = [args.model] + [m for m in SELECTED_MODELS if m != args.model]

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    simple_output = os.path.join(out_dir, "bridge_simple_descriptions.json")
    complex_output = os.path.join(out_dir, "bridge_complex_instructions.json")
    scales_output = os.path.join(out_dir, "scales.json")
    scans_scales = os.path.join(os.path.abspath(args.scans_dir), "scales.json") if args.scans_dir else ""

    metas: List[Dict[str, Any]] = []
    if args.source in ("synthetic", "all"):
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            raise SystemExit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            synth_rows = json.load(f)
        metas.extend(meta_from_synthetic(row) for row in synth_rows)
    if args.source in ("real", "all"):
        if not args.real_dir or not os.path.isdir(args.real_dir):
            print("Error: --source real/all needs --real-dir (or REAL_BRIDGE_DIR)")
            raise SystemExit(1)
        metas.extend(load_real_bridges(args.real_dir))

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        metas = [m for m in metas if m["id"] in wanted]
        missing_ids = wanted - {m["id"] for m in metas}
        if missing_ids:
            print(f"Warning: ids not found in metadata: {sorted(missing_ids)}")

    cache_paths = [p for p in (scales_output, scans_scales) if p]
    scales_cache: Dict[str, Dict[str, Any]] = {}
    for path in cache_paths:
        scales_cache.update(load_scales_json(path))

    def has_scale_source(object_id: str) -> bool:
        rec = scales_cache.get(object_id)
        if rec and rec.get("norm_radius_m") is not None:
            return True
        if args.source in ("real", "all"):
            if os.path.isfile(os.path.join(args.real_dir, object_id, "parameters.json")):
                return True
        if args.source in ("synthetic", "all") and args.scans_dir:
            if _merged_xyz_path(args.scans_dir, object_id):
                return True
        return False

    selected: List[Dict[str, Any]] = []
    skipped_no_scale: List[str] = []
    for meta in metas:
        if not has_scale_source(meta["id"]):
            skipped_no_scale.append(meta["id"])
            if args.scale_mode == "prompt":
                continue
        selected.append(meta)
        if args.count is not None and len(selected) >= args.count:
            break
    metas = selected

    scales = collect_scales(
        [m["id"] for m in metas],
        scans_dir=args.scans_dir if args.source in ("synthetic", "all") else None,
        real_dir=args.real_dir if args.source in ("real", "all") else None,
        cache_paths=cache_paths,
    )
    used_scales = {oid: scales[oid] for oid in [m["id"] for m in metas] if oid in scales}
    if used_scales:
        save_scales_json(scales_output, {**load_scales_json(scales_output), **used_scales})
        if scans_scales and args.source in ("synthetic", "all"):
            save_scales_json(scans_scales, {**load_scales_json(scans_scales), **{
                oid: rec for oid, rec in used_scales.items()
                if rec.get("source") != "parameters.json"
            }})

    if skipped_no_scale:
        preview = ", ".join(skipped_no_scale[:8])
        extra = f" (+{len(skipped_no_scale) - 8} more)" if len(skipped_no_scale) > 8 else ""
        print(
            f"No norm_radius_m for {len(skipped_no_scale)} bridge(s) "
            f"(no merged XYZ / parameters.json): {preview}{extra}"
        )
        if args.scale_mode == "prompt":
            print("Those were skipped because --scale-mode prompt requires a radius.")

    if args.count is not None:
        print(f"Processing {len(metas)} bridges (limited by --count)...")
    else:
        print(f"Processing {len(metas)} bridges...")

    if not metas:
        print("Nothing to process.")
        raise SystemExit(1)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN  scale_mode={args.scale_mode}  apply_scale_to={args.apply_scale_to}")
        print(f"{'='*60}")
        for meta in metas:
            radius = (scales.get(meta["id"]) or {}).get("norm_radius_m")
            print(f"\n{meta['id']}")
            print(f"  metadata: { {k: v for k, v in meta.items() if k not in ('id', 'domain')} }")
            if radius is None:
                print("  norm_radius_m: MISSING")
                continue
            print(f"  norm_radius_m: {float(radius):.3f} m  → caption {float(radius):.1f} m")
            example = inject_scale_sentence(
                "<point>\nWhat is the total length of the bridge?",
                float(radius),
            )
            print(f"  L1 human turn:\n    {example.replace(chr(10), chr(10) + '    ')}")
        print(f"\nWould write captions to {out_dir}")
        print(f"Would write scales to {scales_output}")
        raise SystemExit(0)

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

    def maybe_scale(record: Dict[str, Any]) -> Dict[str, Any]:
        if args.scale_mode != "prompt":
            return record
        radius = (scales.get(record["object_id"]) or {}).get("norm_radius_m")
        if radius is None:
            raise RuntimeError(f"No norm_radius_m for {record['object_id']}")
        return apply_scale_to_record(record, float(radius), args.apply_scale_to)

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
        radius = (scales.get(meta["id"]) or {}).get("norm_radius_m")
        if radius is not None:
            print(f"norm_radius_m = {float(radius):.3f} m")
        print(f"{'='*60}")

        if need_simple:
            try:
                print("Generating simple description...")
                if args.template:
                    simple_rec, _ = template_caption_records(meta, radius)
                    simple_descriptions.append(simple_rec)
                else:
                    simple_descriptions.append(maybe_scale(generate_simple_description(meta)))
                print(f"Generated simple description for {meta['id']}")
            except Exception as e:
                print(f"Failed to generate simple description: {e}")
                errors.append(f"[bridge {meta['id']}] simple description: {e}")

        if need_complex:
            try:
                print("Generating complex instructions...")
                if args.template:
                    _, complex_convs = template_caption_records(meta, radius)
                    complex_instructions.extend(complex_convs)
                else:
                    complex_convs = [
                        maybe_scale(conv) for conv in generate_complex_instructions(meta)
                    ]
                    complex_instructions.extend(complex_convs)
                print(f"Generated complex conversations for {meta['id']}")
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
    print(f"Saved scales to {scales_output}")
    print(f"{'='*60}")
    print(f"\nComplete! Total conversations: {len(simple_descriptions) + len(complex_instructions)}")
    print("COST SUMMARY")
    print(f"Total Input Tokens:  {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total Cost:          ${total_cost:.4f}")
   