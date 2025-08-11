"""Labels PRs based on bot comment checkboxes."""

import json
import os
import sys

from _commons import label_options
from github import Github

if __name__ == "__main__":
    context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

    repo = context_dict["repository"]
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo)
    pr_number = context_dict["event"]["number"]
    pr = repo.get_pull(number=pr_number)
    labels = [label.name for label in pr.get_labels()]

    comment = None
    for c in pr.get_comments():
        if (
            c.user.login == "tsml-actions-bot[bot]"
            and "## Thank you for contributing to `tsml-eval`" in c.body
        ):
            comment = c
            break

    if comment is None:
        sys.exit(0)

    comment_body = comment.body
    for option in label_options:
        if f"- [x] {option[1]}" in comment_body and option[0] not in labels:
            comment_body = comment_body.replace(
                f"- [x] {option[1]}",
                f"- [ ] {option[1]}",
            )
        elif f"- [ ] {option[1]}" in comment_body and option[0] in labels:
            comment_body = comment_body.replace(
                f"- [ ] {option[1]}",
                f"- [x] {option[1]}",
            )

    print("Comment body:")  # noqa: T201
    print(comment_body)  # noqa: T201
    print(f"type(comment_body): {type(comment_body)}")  # noqa: T201

    comment.edit(comment_body)
