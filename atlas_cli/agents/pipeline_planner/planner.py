"""
Pipeline Planner orchestrator — loads Phase 2 artifacts, calls the LLM,
validates the response, and persists the ExecutionPlan.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from atlas_cli.agents.pipeline_planner import llm_client
from atlas_cli.agents.pipeline_planner.prompts import build_system_prompt, build_user_prompt
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan
from atlas_cli.core.config import settings
from atlas_cli.core.logger import logger


def _load_json(path: Path, label: str) -> dict:
    """Load a JSON artifact file, raising a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at: {path}\n"
            "Run 'atlas analyze <file>' first to generate dataset intelligence artifacts."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from LLM output, stripping markdown fences
    or any surrounding text the model may have added despite instructions.
    """
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


def run_planner(
    goal: str,
    run_dir: Path,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> ExecutionPlan:
    """
    Orchestrate the Pipeline Planner:
      1. Validate API keys.
      2. Load Phase 2 artifacts from run_dir.
      3. Build prompts and call the LLM.
      4. Parse and validate the response into ExecutionPlan.
      5. Persist execution_plan.json to run_dir.

    Args:
        goal: User's natural-language prediction goal.
        run_dir: Path to the run directory containing Phase 2 artifact JSON files.
        model: Optional LiteLLM model string override.
        temperature: LLM sampling temperature.

    Returns:
        Validated ExecutionPlan instance.

    Raises:
        RuntimeError: On API key issues, LLM errors, or JSON parse failures.
    """

    provider = llm_client.validate_api_keys()
    logger.info(f"Using LLM provider: {provider}")

    dataset_summary = _load_json(run_dir / "dataset_summary.json", "Dataset summary")
    quality_report  = _load_json(run_dir / "quality_report.json",  "Quality report")
    risk_assessment = _load_json(run_dir / "risk_assessment.json",  "Risk assessment")

    messages = [
        {"role": "system",  "content": build_system_prompt()},
        {"role": "user",    "content": build_user_prompt(goal, dataset_summary, quality_report, risk_assessment)},
    ]

    logger.info("Calling LLM planner...")
    raw_response = llm_client.call(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=2048,
    )
    logger.debug(f"Raw LLM response:\n{raw_response}")

    json_str = _extract_json(raw_response)
    try:
        plan_dict = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response:\n{raw_response[:800]}"
        ) from exc

    try:
        plan = ExecutionPlan.model_validate(plan_dict)
    except ValidationError as exc:
        raise RuntimeError(
            f"LLM response did not match the ExecutionPlan schema.\n"
            f"Validation errors:\n{exc}\n"
            f"Parsed dict:\n{json.dumps(plan_dict, indent=2)[:800]}"
        ) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "execution_plan.json"
    plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, default=str),
        encoding="utf-8"
    )
    logger.info(f"Execution plan saved to: {plan_path}")

    return plan
