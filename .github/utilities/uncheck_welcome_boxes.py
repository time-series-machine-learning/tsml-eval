"""Labels PRs based on bot comment checkboxes."""

import json
import os

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
pr = repo.get_issue(number=issue_number).as_pull_request()
comment_body = context_dict["event"]["comment"]["body"]
comment = pr.get_issue_comment(context_dict["event"]["comment"]["id"])

if "- [x] Push an empty commit to re-run CI checks" in comment_body:
    comment.edit(
        comment_body.replace(
            "- [x] Push an empty commit to re-run CI checks",
            "- [ ] Push an empty commit to re-run CI checks",
        )
    )
