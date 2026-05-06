# Private Preview Access and Repository Workflow

This page is for maintainers preparing a private preview with customers and GBBs.

## Access model

GitHub read access is repo-scoped, not branch-scoped. A user with Read access can clone the repo, branches, tags, and visible history. Branch protection controls writes; it does not hide branches.

Therefore, do not use branches as the customer isolation boundary. Use a separate private preview repo that contains only customer-safe content.

| Audience | Repository | Permission |
|---|---|---|
| 3P customers | Private preview repo | Read |
| GBB readers | Private preview repo | Read |
| GBB contributors | Private preview repo | Prefer fork + PR; otherwise Write with branch protection |
| Engineering team | EMU/internal repo | Write/Maintain/Admin |

## What belongs in the preview repo

- README
- quickstart docs
- customer-safe target docs
- examples intended for preview
- config reference
- contribution instructions

## What stays internal

- engineering design debates
- LT review material
- TAM and OSS escalation docs
- roadmap and backlog
- private customer notes
- science comparison drafts
- security/threat-modeling details not intended for customers

## Repo workflow

Use the EMU/internal repo as the engineering source of truth. Promote only sanitized commits to the private preview repo.

Recommended flow:

```powershell
# Internal branch and PR first.
git switch -c docs/customer-preview origin/main
git push -u origin docs/customer-preview

# Then promote safe commits to the preview repo.
git fetch preview
git switch -c preview/docs-customer-preview preview/main
git cherry-pick <safe-commit-sha>
git diff --name-only preview/main...HEAD
git push -u preview preview/docs-customer-preview
```

Open a PR into the preview repo. Do not merge the whole internal branch or history into the preview repo.

## Branch protection

Protect `main` in the preview repo:

- require pull requests
- require at least one internal reviewer
- block force-push
- block branch deletion
- require status checks when available
- add CODEOWNERS for sensitive paths

## GBB PRs

Preferred: GBB contributors fork the preview repo and open PRs back. This avoids granting direct write access.

If Write access is necessary, keep `main` protected and require internal review before merge.

## EMU alias-resolution bug

If an engineer cannot be added to the EMU repo because of a GitHub alias-resolution issue, treat that as an internal identity/access problem.

Do not copy private EMU-only docs into the preview repo as a workaround. Instead:

1. fix org/team membership through the appropriate GitHub or identity admin path;
2. temporarily share internal docs through a Microsoft-controlled internal location; or
3. create a separate internal docs repo with the correct audience.

The private preview repo should remain safe for customers to clone.
