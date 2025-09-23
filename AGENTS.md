# Repository Guidelines for Agents

This project builds datasets of question-answer pairs over table and passage content. Most automation scripts live under `data_processing/`.

## Environment setup

Install Python dependencies before running scripts or tests:

```bash
pip install -r requirements.txt --quiet
```

## Test commands

Run the following command before submitting changes that touch Python sources:

```bash
python -m compileall data_processing/generate_passage_questions.py data_processing/process_passage_questions.py
```

## Code style

- Follow PEP 8 with 4 spaces for indentation.
- Prefer explicit, well-named functions over in-line scripts.
- Include type hints on public function signatures when practical.

## Documentation

- Update `README.md` when behaviour changes for the user-facing scripts.
- Keep docstrings up to date and add comments for non-obvious logic.

## Pull request message

Summarize notable changes and list all commands executed for testing.
