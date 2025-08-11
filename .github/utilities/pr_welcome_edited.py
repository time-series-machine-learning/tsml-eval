"""Labels PRs based on bot comment checkboxes."""

import json
import os
import sys

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
issue = repo.get_issue(number=issue_number)
comment = context_dict["event"]["comment"]
comment_body = context_dict["event"]["comment"]["body"]
comment_user = context_dict["event"]["comment"]["user"]["login"]
labels = [label.name for label in issue.get_labels()]

if (
    issue.pull_request is None
    or comment_user != "tsml-actions-bot[bot]"
    or "## Thank you for contributing to `tsml-eval`" not in comment_body
):
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=false", file=fh)  # noqa: T201
    sys.exit(0)
pr = issue.as_pull_request()

label_options = [
    ("full pre-commit", "Run `pre-commit` checks for all files"),
    ("full examples run", "Run all notebook example tests"),
    ("full pytest actions", "Run all `pytest` tests and configurations"),
    (
        "stop pre-commit fixes",
        "Stop automatic `pre-commit` fixes (always disabled for drafts)",
    ),
]

for option in label_options:
    if f"- [x] {option[1]}" in comment_body and option[0] not in labels:
        pr.add_to_labels(option[0])
    elif f"- [ ] {option[1]}" in comment_body and option[0] in labels:
        pr.remove_from_labels(option[0])

repo_name = pr.head.repo.full_name
branch_name = pr.head.ref
with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    print(f"repo={repo_name}", file=fh)  # noqa: T201
    print(f"branch={branch_name}", file=fh)  # noqa: T201

if "- [x] Push an empty commit to re-run CI checks" in comment_body:
    comment.edit(
        comment_body.replace(
            "- [x] Push an empty commit to re-run CI checks",
            "- [ ] Push an empty commit to re-run CI checks",
        )
    )

    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=true", file=fh)  # noqa: T201
else:
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=false", file=fh)  # noqa: T201
