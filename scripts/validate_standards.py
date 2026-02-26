#!/usr/bin/env python3
"""
Validate files against Greene Lab coding standards.

Usage:
    python scripts/validate_standards.py <file1> <file2> ...
    python scripts/validate_standards.py --check-all
    python scripts/validate_standards.py --staged  # Check git staged files

Standards checked:
- No emojis in any files
- Python files: PEP 8 compliance (via flake8 if available)
- Python files: Comprehensive docstrings
- No prohibited statements (e.g., "Greene Lab standards:")
"""

import sys
import re
import argparse
from pathlib import Path
import subprocess
import ast


def has_emoji(text):
    """
    Check if text contains emojis.

    Parameters
    ----------
    text : str
        Text to check

    Returns
    -------
    bool
        True if emojis found
    list
        List of (line_num, emoji_char) tuples
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FAFF"  # extended symbols
        "]+",
        flags=re.UNICODE,
    )

    emojis_found = []
    for line_num, line in enumerate(text.split("\n"), 1):
        matches = emoji_pattern.findall(line)
        if matches:
            for emoji in matches:
                emojis_found.append((line_num, emoji))

    return len(emojis_found) > 0, emojis_found


def has_prohibited_statements(text):
    """
    Check for prohibited statements.

    Parameters
    ----------
    text : str
        Text to check

    Returns
    -------
    bool
        True if prohibited statements found
    list
        List of (line_num, statement) tuples
    """
    prohibited = [
        "Greene Lab standards:",
        "Greene Lab standards: - No emojis",
    ]

    found = []
    for line_num, line in enumerate(text.split("\n"), 1):
        for statement in prohibited:
            if statement in line:
                found.append((line_num, statement))

    return len(found) > 0, found


def check_python_docstrings(filepath):
    """
    Check if Python file has comprehensive docstrings.

    Parameters
    ----------
    filepath : Path
        Path to Python file

    Returns
    -------
    bool
        True if issues found
    list
        List of missing docstring locations
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError as e:
        return True, [f"Syntax error: {e}"]

    missing = []

    # Check module docstring
    if not ast.get_docstring(tree):
        missing.append("Module-level docstring missing")

    # Check functions and classes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                missing.append(f"Function '{node.name}' (line {node.lineno})")
        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                missing.append(f"Class '{node.name}' (line {node.lineno})")

    return len(missing) > 0, missing


def check_pep8(filepath):
    """
    Check PEP 8 compliance using flake8.

    Parameters
    ----------
    filepath : Path
        Path to Python file

    Returns
    -------
    bool
        True if violations found
    list
        List of violation messages
    """
    try:
        result = subprocess.run(
            ["flake8", str(filepath)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return False, []
        else:
            violations = result.stdout.strip().split("\n")
            return True, violations
    except FileNotFoundError:
        return False, ["flake8 not installed (skipping PEP 8 check)"]
    except subprocess.TimeoutExpired:
        return True, ["flake8 timeout"]


def validate_file(filepath):
    """
    Validate a single file against all standards.

    Parameters
    ----------
    filepath : Path
        Path to file

    Returns
    -------
    dict
        Validation results
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return {
            "filepath": str(filepath),
            "exists": False,
            "errors": ["File does not exist"],
            "valid": False,
        }

    results = {
        "filepath": str(filepath),
        "exists": True,
        "errors": [],
        "warnings": [],
        "valid": True,
    }

    # Read file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Binary file, skip
        results["warnings"].append("Binary file (skipped)")
        return results

    # Check for emojis
    has_emojis, emoji_list = has_emoji(content)
    if has_emojis:
        results["valid"] = False
        for line_num, emoji in emoji_list:
            results["errors"].append(f"Line {line_num}: Emoji found '{emoji}'")

    # Check for prohibited statements
    has_prohibited, prohibited_list = has_prohibited_statements(content)
    if has_prohibited:
        results["valid"] = False
        for line_num, statement in prohibited_list:
            results["errors"].append(
                f"Line {line_num}: Prohibited statement '{statement}'"
            )

    # Python-specific checks
    if filepath.suffix == ".py":
        # Check docstrings
        has_missing, missing_list = check_python_docstrings(filepath)
        if has_missing:
            results["warnings"].append("Missing docstrings:")
            for item in missing_list:
                results["warnings"].append(f"  - {item}")

        # Check PEP 8
        has_violations, violations = check_pep8(filepath)
        if has_violations:
            for violation in violations:
                if "not installed" in violation:
                    results["warnings"].append(violation)
                else:
                    results["warnings"].append(f"PEP 8: {violation}")

    return results


def get_staged_files():
    """
    Get list of staged files in git.

    Returns
    -------
    list
        List of file paths
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
        return files
    except subprocess.CalledProcessError:
        return []


def main():
    """
    Main validation function.
    """
    parser = argparse.ArgumentParser(
        description="Validate files against Greene Lab coding standards"
    )
    parser.add_argument("files", nargs="*", help="Files to validate")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all Python and Markdown files",
    )
    parser.add_argument(
        "--staged", action="store_true", help="Check git staged files only"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print errors"
    )

    args = parser.parse_args()

    # Determine files to check
    if args.staged:
        files = get_staged_files()
        if not files:
            print("No staged files found")
            return 0
    elif args.check_all:
        repo_root = Path(__file__).parent.parent
        files = list(repo_root.glob("**/*.py")) + list(
            repo_root.glob("**/*.md")
        )
        # Exclude certain directories
        exclude_dirs = {".git", "__pycache__", ".ipynb_checkpoints", "venv"}
        files = [
            f
            for f in files
            if not any(excl in f.parts for excl in exclude_dirs)
        ]
    else:
        files = args.files

    if not files:
        parser.print_help()
        return 1

    # Validate files
    all_valid = True
    error_count = 0
    warning_count = 0

    for filepath in files:
        results = validate_file(filepath)

        if not results["exists"]:
            print(f"ERROR: {filepath} - File does not exist")
            all_valid = False
            continue

        # Print results
        if not results["valid"]:
            all_valid = False
            print(f"\nFAILED: {filepath}")
            for error in results["errors"]:
                print(f"  ERROR: {error}")
                error_count += 1
        elif results["warnings"] and not args.quiet:
            print(f"\nWARNING: {filepath}")
            for warning in results["warnings"]:
                print(f"  WARNING: {warning}")
                warning_count += 1
        elif not args.quiet:
            print(f"PASSED: {filepath}")

    # Summary
    print("\n" + "=" * 80)
    if all_valid:
        print("Document checked to maintain coding standards")
        print(f"All {len(files)} files passed validation")
        if warning_count > 0:
            print(f"  ({warning_count} warnings)")
        return 0
    else:
        print(f"VALIDATION FAILED")
        print(f"  Errors: {error_count}")
        print(f"  Warnings: {warning_count}")
        print(f"  Files with errors: {len([f for f in files if not validate_file(f)['valid']])}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
