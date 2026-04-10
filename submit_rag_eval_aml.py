"""Submit scripts/run_rag_eval_batch.py as an Azure ML command job."""

from __future__ import annotations

import argparse
import os
import shlex
import time
import webbrowser
from pathlib import Path
from typing import Dict, List

from azure.ai.ml import Input, MLClient, Output, command
from azure.identity import DefaultAzureCredential
from dotenv import dotenv_values


DEFAULT_SUBSCRIPTION_ID = "5c9e4789-4852-4ffe-8551-d682affcbd74"
DEFAULT_RESOURCE_GROUP = "playground-rg"
DEFAULT_WORKSPACE_NAME = "as-playground-w3-ws"
DEFAULT_COMPUTE = "a100x4"
DEFAULT_EXPERIMENT_NAME = "drugbank-rag-eval"
DEFAULT_ENVIRONMENT_URI = (
    "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"
)
DEFAULT_AML_OUTPUT_NAME = "rag_results"


def _required(value: str, name: str) -> str:
    cleaned = value.strip() if value else ""
    if not cleaned:
        raise ValueError(f"{name} must be provided")
    return cleaned


def _load_job_env_vars(env_file: Path) -> Dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found: {env_file}")

    values = dotenv_values(env_file)
    env_vars: Dict[str, str] = {}
    for key, value in values.items():
        if not key or value is None:
            continue
        env_vars[key] = str(value)

    if not env_vars:
        raise RuntimeError(f"No key/value pairs found in {env_file}")
    return env_vars


def _build_job_name(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def _normalize_output_subdir(raw_value: str) -> str:
    cleaned = raw_value.strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("output-dir must not be empty")
    return cleaned


def _force_output_dir_arg(args: List[str], output_dir: str) -> List[str]:
    rewritten: List[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--output-dir":
            skip_next = True
            continue
        if token.startswith("--output-dir="):
            continue
        rewritten.append(token)
    rewritten.extend(["--output-dir", output_dir])
    return rewritten


def _default_rag_batch_args(output_dir: str) -> List[str]:
    return [
        "--questions-dir",
        "questions_final",
        "--output-dir",
        output_dir,
        "--db-path",
        "data/drugbank.db",
        "--table-index-name",
        "drug_bank_data_lake_tables",
        "--passage-index-name",
        "drug_bank_data_lake",
        "--limit-per-file",
        "0",
        "--workers",
        "5",
        "--gpu-ids",
        "0,1,2,3",
        "--question-workers",
        "16",
        "--eval-workers",
        "24",
        "--embed-batch-size",
        "256",
        "--save-every",
        "100",
        "--nl2sql-model",
        "openai/gpt-5",
        "--answer-model",
        "openai/gpt-5",
        "--eval-model",
        "openai/gpt-5",
    ]


def _build_rag_eval_command(
    extra_args: List[str],
    skip_bootstrap_install: bool,
    output_dir: str,
) -> str:
    normalized_args = list(extra_args)
    if normalized_args and normalized_args[0] == "--":
        normalized_args = normalized_args[1:]
    if not normalized_args:
        normalized_args = _default_rag_batch_args(output_dir)
    else:
        normalized_args = _force_output_dir_arg(normalized_args, output_dir)

    # Ensure expected local `data/` assets exist inside AML working directory even when
    # `.gitignore` excludes them from the code artifact.
    prep_cmd = (
        "mkdir -p data "
        "&& ln -sfn '${{inputs.data_dir}}/Pharma' data/Pharma "
        "&& if [ -f '${{inputs.data_dir}}/drugbank.db' ]; then "
        "cp '${{inputs.data_dir}}/drugbank.db' data/drugbank.db; "
        "else "
        "python scripts/load_csv_to_sqlite.py --tables-dir data/Pharma/drugbank-tables --db-path data/drugbank.db; "
        "fi"
    )
    run_cmd = shlex.join(["python", "scripts/run_rag_eval_batch.py", *normalized_args])
    full_cmd = f"{prep_cmd} && {run_cmd}"
    if skip_bootstrap_install:
        return full_cmd

    bootstrap_cmd = (
        "python -m pip install --quiet --root-user-action=ignore --upgrade-strategy only-if-needed "
        "azure-search-documents openai pandas rouge-score python-dotenv tqdm transformers==4.53.3"
    )
    return f"{bootstrap_cmd} && {full_cmd}"


def submit_job(args: argparse.Namespace) -> None:
    subscription_id = _required(args.subscription_id, "subscription_id")
    resource_group = _required(args.resource_group, "resource_group")
    workspace_name = _required(args.workspace_name, "workspace_name")
    compute_name = _required(args.compute, "compute")
    environment_uri = _required(args.environment_uri, "environment_uri")

    env_path = Path(args.env_file).expanduser().resolve()
    job_env_vars = _load_job_env_vars(env_path)
    output_subdir = _normalize_output_subdir(
        args.output_dir or f"questions_final_test/aml_full_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = f"${{{{outputs.{DEFAULT_AML_OUTPUT_NAME}}}}}/{output_subdir}"
    command_line = _build_rag_eval_command(
        args.rag_eval_args,
        skip_bootstrap_install=args.skip_bootstrap_install,
        output_dir=output_dir,
    )
    code_dir = Path(__file__).resolve().parent
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    job_name = _build_job_name(args.job_name_prefix)
    job = command(
        name=job_name,
        display_name=job_name,
        experiment_name=args.experiment_name,
        code=str(code_dir),
        command=command_line,
        inputs={
            "data_dir": Input(
                type="uri_folder",
                path=str(code_dir / "data"),
            )
        },
        outputs={
            DEFAULT_AML_OUTPUT_NAME: Output(
                type="uri_folder",
                mode="rw_mount",
            )
        },
        environment=environment_uri,
        compute=compute_name,
        instance_count=1,
        environment_variables=job_env_vars,
    )

    submitted = ml_client.jobs.create_or_update(job)
    print(f"[AML] Submitted job: {submitted.name}")
    print(f"[AML] Experiment: {args.experiment_name}")
    print(f"[AML] Compute: {compute_name}")
    print(f"[AML] Command: {command_line}")
    print(f"[AML] Environment: {environment_uri}")
    print(f"[AML] Output subdir inside '{DEFAULT_AML_OUTPUT_NAME}': {output_subdir}")
    print(f"[AML] Loaded {len(job_env_vars)} variables from: {env_path}")
    try:
        output_path = submitted.outputs[DEFAULT_AML_OUTPUT_NAME].path
    except Exception:
        output_path = None
    if output_path:
        print(f"[AML] Output URI: {output_path}")
    if submitted.studio_url:
        print(f"[AML] Studio URL: {submitted.studio_url}")
        if not args.no_open_browser:
            try:
                webbrowser.open(submitted.studio_url)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit an Azure ML job that runs `python scripts/run_rag_eval_batch.py` "
            "and injects values from a local .env file."
        )
    )
    parser.add_argument(
        "--subscription-id",
        default=os.getenv("AML_SUBSCRIPTION_ID", DEFAULT_SUBSCRIPTION_ID),
    )
    parser.add_argument(
        "--resource-group",
        default=os.getenv("AML_RESOURCE_GROUP", DEFAULT_RESOURCE_GROUP),
    )
    parser.add_argument(
        "--workspace-name",
        default=os.getenv("AML_WORKSPACE_NAME", DEFAULT_WORKSPACE_NAME),
    )
    parser.add_argument("--compute", default=DEFAULT_COMPUTE)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--environment-uri", default=DEFAULT_ENVIRONMENT_URI)
    parser.add_argument("--job-name-prefix", default="drugbank-rag-eval")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory used when no pass-through args are provided. "
            "Defaults to questions_final_test/aml_full_eval_<timestamp>."
        ),
    )
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--skip-bootstrap-install", action="store_true")
    parser.add_argument(
        "rag_eval_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed through to `python scripts/run_rag_eval_batch.py`. "
            "Example: -- --workers 4 --gpu-ids 0,1,2,3"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    submit_job(parse_args())
