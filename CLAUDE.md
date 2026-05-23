# autoalpha — Agent Instructions

## Editing PLAN.md

**Never rewrite a section from scratch.** Always edit incrementally (add/modify lines, mark items complete). Rewriting loses detail silently.

Before committing any change to PLAN.md, run:

```bash
git diff PLAN.md | grep "^-" | grep -E "^\-\s*- \[" 
```

If that outputs any lines, a checklist item was removed. Either restore it or explicitly note in the commit message that the removal was intentional and why.

**Rule:** a `- [ ]` item may only be removed if it is either (a) marked `- [x]` (complete) or (b) explicitly called out as dropped with a reason in the commit message.

## General

- Always push to GitHub after committing.
- Keep PLAN.md as the single source of truth for what has been built and what is next.
