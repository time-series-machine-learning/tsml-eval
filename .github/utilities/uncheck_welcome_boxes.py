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
comment_body = context_dict["event"]["comment"]["body"]
comment_user = context_dict["event"]["comment"]["user"]["login"]

if (
    issue.pull_request is None
    or "## Thank you for contributing to `tsml-eval`" not in comment_body
):
    sys.exit(0)

pr = issue.as_pull_request()
comment = pr.get_issue_comment(context_dict["event"]["comment"]["id"])

if "- [x] Push an empty commit to re-run CI checks" in comment_body:
    comment.edit(
        comment_body.replace(
            "- [x] Push an empty commit to re-run CI checks",
            "- [ ] Push an empty commit to re-run CI checks",
        )
    )
