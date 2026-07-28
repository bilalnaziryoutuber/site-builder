#!/usr/bin/env python3
"""
generate_tool.py

Updated for the Next.js micro‑SaaS platform. Instead of generating a full
HTML page, this script asks Gemini to create a tool definition (system prompt,
JSON schema, SEO metadata) and appends it to config/tools.json.

The registry is consumed by the Next.js app to dynamically render tool pages
and power the Gemini API handler.

Requirements:
    pip install google-genai

Environment:
    GEMINI_API_KEY - your Gemini API key
"""

import os
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai

# Use a model that is currently available to all users
MODEL_NAME = "gemini-1.5-flash"   # <-- changed from gemini-2.5-flash
CONFIG_DIR = Path("config")
REGISTRY_FILE = CONFIG_DIR / "tools.json"

# Hardcoded pool of tool ideas – you can extend this or let Gemini generate
# a completely novel idea later.
TOOL_IDEAS = [
    "Freelance Hourly Rate Calculator",
    "SaaS Churn Calculator",
    "Simple Pomodoro Timer",
    "Tip Splitter Calculator",
    "Unit Converter (Length, Weight, Temperature)",
    "Password Strength Checker",
    "Word & Character Counter",
    "Countdown Timer for Events",
    "Loan Payoff Calculator",
    "Habit Streak Tracker",
]

# Predefined categories (match your app's sidebar grouping)
CATEGORIES = [
    "document-ai",
    "productivity",
    "developer-tools",
    "analytics",
    "finance",
    "conversion",
]


def build_tool_prompt(tool_idea: str) -> str:
    """Build the instruction prompt sent to Gemini to generate a tool definition."""
    return f"""
You are an expert AI architect. Your task is to define a new micro‑tool for a platform
that offers many AI‑powered utilities.

The tool is called "{tool_idea}".

Generate a complete tool definition as a JSON object with the following structure:

{{
  "id": "a unique kebab-case identifier based on the tool name (e.g., 'freelance-rate')",
  "slug": "same as id (for URL path)",
  "subdomain": "a short kebab-case subdomain for this tool (e.g., 'rate')",
  "category": "one of: {', '.join(CATEGORIES)} – pick the most appropriate",
  "title": "the tool title (may be same as {tool_idea})",
  "description": "a concise, compelling one‑line description of what the tool does",
  "systemPrompt": "the complete system prompt to be used when calling Gemini for this tool. Include role, instructions, output format, and any constraints.",
  "jsonSchema": {{
    "type": "object",
    "properties": {{
      // define the input parameters the user must provide.
      // For example:
      // "hours": {{ "type": "number", "description": "Number of hours worked" }},
      // "rate": {{ "type": "number", "description": "Hourly rate in USD" }}
    }},
    "required": ["list", "of", "required", "fields"]
  }},
  "seo": {{
    "title": "SEO title (max 60 chars)",
    "description": "SEO meta description (max 160 chars)",
    "keywords": ["keyword1", "keyword2"]
  }}
}}

Requirements:
- The `systemPrompt` must instruct Gemini to perform the tool's function given the input parameters defined in `jsonSchema`.
- The output of the tool should be a structured JSON object (the API will enforce this via `response_schema`).
- Ensure the schema is clear, with proper descriptions and types.
- Choose an appropriate category from the list.
- The `subdomain` should be unique and short (2‑5 characters if possible, but descriptive).

Return ONLY the JSON object, without any Markdown fences or extra commentary.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """Strip Markdown fences and parse JSON from the model's response."""
    text = text.strip()
    # Remove possible Markdown code fences
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # If it still contains surrounding text, try to extract a JSON block
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    return json.loads(text)


def load_existing_registry() -> List[Dict[str, Any]]:
    """Load the current tools.json; create an empty list if it doesn't exist."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def write_registry(registry: List[Dict[str, Any]]) -> None:
    """Write the registry back to tools.json with pretty formatting."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")


def generate_tool_definition(tool_idea: str, client: genai.Client) -> Dict[str, Any]:
    """Ask Gemini to generate a complete tool definition."""
    prompt = build_tool_prompt(tool_idea)
    print(f"Requesting {MODEL_NAME} to define tool: {tool_idea}")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Empty response from Gemini.")

    try:
        definition = extract_json(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        # Save the raw response for debugging, then abort.
        raw_file = REGISTRY_FILE.parent / "error_response.txt"
        raw_file.write_text(response.text, encoding="utf-8")
        raise RuntimeError(f"Failed to parse Gemini response as JSON. Raw response saved to {raw_file}") from e

    # Validate required fields
    required = ["id", "slug", "subdomain", "category", "title", "description",
                "systemPrompt", "jsonSchema", "seo"]
    for field in required:
        if field not in definition:
            raise ValueError(f"Missing required field '{field}' in Gemini response.")

    return definition


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # 1. Pick a tool idea (could be randomized from list, or let Gemini suggest one)
    tool_idea = random.choice(TOOL_IDEAS)
    print(f"Selected tool idea: {tool_idea}")

    client = genai.Client(api_key=api_key)

    # 2. Generate tool definition from Gemini
    try:
        new_tool = generate_tool_definition(tool_idea, client)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Load existing registry, check for duplicate id/slug/subdomain
    registry = load_existing_registry()
    existing_ids = {tool["id"] for tool in registry}
    existing_slugs = {tool["slug"] for tool in registry}
    existing_subdomains = {tool["subdomain"] for tool in registry}

    if new_tool["id"] in existing_ids:
        print(f"WARNING: Tool with id '{new_tool['id']}' already exists. Skipping.", file=sys.stderr)
        sys.exit(0)
    if new_tool["slug"] in existing_slugs:
        print(f"WARNING: Tool with slug '{new_tool['slug']}' already exists. Skipping.", file=sys.stderr)
        sys.exit(0)
    if new_tool["subdomain"] in existing_subdomains:
        print(f"WARNING: Tool with subdomain '{new_tool['subdomain']}' already exists. Skipping.", file=sys.stderr)
        sys.exit(0)

    # 4. Append and save
    registry.append(new_tool)
    write_registry(registry)
    print(f"Appended new tool '{new_tool['title']}' (id: {new_tool['id']}) to {REGISTRY_FILE}")

    # 5. Set GitHub outputs for the PR title/body
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"tool_id={new_tool['id']}\n")
            f.write(f"tool_title={new_tool['title']}\n")


if __name__ == "__main__":
    main()
