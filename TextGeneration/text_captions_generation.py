import json, os, random, argparse
from typing import Dict, Any, List, Tuple
import requests

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
        "- Mention key components: railings, abutments, deck, surrounding environment\n"
        "- Focus on what the point cloud captures, not on data quality\n"
        "- Use professional, clear language\n"
        "- Mention that it's a 3D representation/digital twin/LiDAR scan\n\n"
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
        "What is this object made of?",
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
        '    "answer": "A detailed 50-100 word description of the bridge covering geometry, structure, dimensions, and purpose"\n'
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
        "- Describe only the bridge itself — geometry, structure, materials, dimensions\n"
        "- Use specific measurements and technical details from the metadata\n"
        "- For single_round: Generate 3 different Q&A pairs with varied questions about geometry, materials, purpose, structural details, dimensions, etc.\n"
        "- For multi_round: Generate 3 rounds with questions showing logical progression and depth\n"
        "- Use clear, professional technical language\n"
        "- Include specific numeric values from metadata in answers\n"
        f"- {tone}\n\n"
        f"BRIDGE METADATA:\n{json.dumps(meta_llm, indent=2, ensure_ascii=False)}"
    )

# === Model pricing (per 1M tokens) ===
MODEL_PRICING = {
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "google/gemini-2.0-pro-exp": {"input": 1.25, "output": 5.00},
}

# === Global cost tracking ===
total_cost = 0.0
total_input_tokens = 0
total_output_tokens = 0

# === OpenRouter API Call ===
def call_openrouter(prompt: str) -> str:
    """
    Call OpenRouter API with best models for technical dataset generation.
    """
    global total_cost, total_input_tokens, total_output_tokens
    
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    
    # Best models for technical dataset generation (ordered by preference)
    # Claude 3.5 Sonnet: Best for structured technical descriptions
    # GPT-4o: Excellent backup with strong reasoning
    # Gemini 2.0 Pro: Good for technical content
    models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-2.0-pro-exp"
    ]
    
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
                "value": data["question"]
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
    for qa in data["single_round"]:
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate simple descriptions and complex instructions for bridge point cloud data"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="bridge_summary.json",
        help="Path to the bridge_summary.json file (default: bridge_summary.json)"
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=None,
        help="Number of bridges to process (default: all bridges in the file)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="Dataset",
        help="Output directory for JSON and error files (default: Dataset)"
    )
    args = parser.parse_args()
    
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    
    # Read all bridges from JSON file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        exit(1)
    
    with open(args.input, "r", encoding="utf-8") as f:
        bridges = json.load(f)
    
    # Limit number of bridges if specified
    if args.count is not None:
        bridges = bridges[:args.count]
        print(f"Processing {len(bridges)} of {len(bridges)} bridges (limited by --count)...")
    else:
        print(f"Processing all {len(bridges)} bridges...")
    
    # Collect conversations and errors
    simple_descriptions = []
    complex_instructions = []
    errors = []
    
    # Process each bridge
    for bridge_data in bridges:
        # Specify fields you DON'T want in the caption
        EXCLUDE_FIELDS = {
            "bridge_id",           # Already handled separately as 'id'
            "include_sidewalks",   # Maybe too minor?
            "top_slab_thk",
            "bottom_slab_thk",
            "web_thk",
            "wing_wall_thickness"
        }
        
        # Take all fields except excluded ones
        meta = {k: v for k, v in bridge_data.items() if k not in EXCLUDE_FIELDS}
        meta["id"] = str(bridge_data.get("bridge_id"))
        meta["domain"] = "bridges"
        
        print(f"\n{'='*60}")
        print(f"Processing {meta['id']}...")
        print(f"{'='*60}")
        
        # Generate simple description
        try:
            print("Generating simple description...")
            simple_conv = generate_simple_description(meta)
            simple_descriptions.append(simple_conv)
            print(f"Generated simple description for {meta['id']}")
        except Exception as e:
            print(f"Failed to generate simple description: {e}")
            errors.append(f"[bridge {meta['id']}] simple description: {e}")
        
        # Generate complex instructions
        try:
            print("Generating complex instructions...")
            complex_convs = generate_complex_instructions(meta)
            complex_instructions.extend(complex_convs)
            print(f"Generated {len(complex_convs)} complex conversations for {meta['id']}")
        except Exception as e:
            print(f"Failed to generate complex instructions: {e}")
            errors.append(f"[bridge {meta['id']}] complex instructions: {e}")
    
    # Write single error log if any errors occurred
    if errors:
        error_log_path = os.path.join(out_dir, "caption_generation_errors.txt")
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"Logged {len(errors)} error(s) to {error_log_path}")
    
    # Save outputs
    simple_output = os.path.join(out_dir, "bridge_simple_descriptions.json")
    complex_output = os.path.join(out_dir, "bridge_complex_instructions.json")
    
    with open(simple_output, "w", encoding="utf-8") as file:
        json.dump(simple_descriptions, file, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(simple_descriptions)} simple descriptions to {simple_output}")
    
    with open(complex_output, "w", encoding="utf-8") as file:
        json.dump(complex_instructions, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(complex_instructions)} complex instructions to {complex_output}")
    print(f"{'='*60}")
    print(f"\nComplete! Total conversations: {len(simple_descriptions) + len(complex_instructions)}")
    
    # Print cost summary
    
    print(f"COST SUMMARY")
    print(f"Total Input Tokens:  {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total Cost:          ${total_cost:.4f}")
   