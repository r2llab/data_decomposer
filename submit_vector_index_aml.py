"""Submit scripts/vector_index.py as an Azure ML command job."""

from __future__ import annotations

import argparse
import os
import shlex
import time
import webbrowser
from pathlib import Path
from typing import Dict, List

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from dotenv import dotenv_values


DEFAULT_SUBSCRIPTION_ID = "5c9e4789-4852-4ffe-8551-d682affcbd74"
DEFAULT_RESOURCE_GROUP = "playground-rg"
DEFAULT_WORKSPACE_NAME = "as-playground-w3-ws"
DEFAULT_COMPUTE = "a100x4"
DEFAULT_EXPERIMENT_NAME = "drugbank-vector-index"
DEFAULT_ENVIRONMENT_URI = (
    "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"
)


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


def _build_vector_index_command(extra_args: List[str], skip_bootstrap_install: bool) -> str:
    normalized_args = list(extra_args)
    if normalized_args and normalized_args[0] == "--":
        normalized_args = normalized_args[1:]
    run_cmd = shlex.join(["python", "scripts/vector_index.py", *normalized_args])
    if skip_bootstrap_install:
        return run_cmd

    bootstrap_cmd = (
        "python -m pip install --quiet --root-user-action=ignore --upgrade-strategy only-if-needed "
        "azure-search-documents python-dotenv tqdm transformers==4.53.3"
    )
    return f"{bootstrap_cmd} && {run_cmd}"


def submit_job(args: argparse.Namespace) -> None:
    subscription_id = _required(args.subscription_id, "subscription_id")
    resource_group = _required(args.resource_group, "resource_group")
    workspace_name = _required(args.workspace_name, "workspace_name")
    compute_name = _required(args.compute, "compute")
    environment_uri = _required(args.environment_uri, "environment_uri")

    env_path = Path(args.env_file).expanduser().resolve()
    job_env_vars = _load_job_env_vars(env_path)
    command_line = _build_vector_index_command(
        args.vector_index_args,
        skip_bootstrap_install=args.skip_bootstrap_install,
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
    print(f"[AML] Loaded {len(job_env_vars)} variables from: {env_path}")
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
            "Submit an Azure ML job that runs `python scripts/vector_index.py` "
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
    parser.add_argument("--job-name-prefix", default="drugbank-vector-index")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--skip-bootstrap-install", action="store_true")
    parser.add_argument(
        "vector_index_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed through to `python scripts/vector_index.py`. "
            "Example: -- --sample-size 256 --sample-only"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    submit_job(parse_args())
