# Working in this repo

## Writing style

Docs, PRs and commit messages should sound like the readme: plain, direct and
a bit conversational. Match the existing voice instead of writing generic
reference documentation.

- Use first person for choices the author made: "I keep the three best
  checkpoints", "I've set up...".
- Address the reader as "you". Contractions are fine ("don't", "you'll",
  "that's"), and so is the occasional aside.
- Keep it short. A sentence or a bullet beats a paragraph.
- Don't use em dashes. Use commas, full stops or parentheses.
- Use British spelling: normalise, tokenise, behaviour, artefact.
- Headings are short, sentence case and unnumbered.
- Code blocks are indented four spaces, not fenced, and should be runnable.
- Avoid tables; use bullet lists instead.
- Link repo files relatively: `[doc/generation.md](doc/generation.md)`.

## Commits

Start with a lowercase type, then a sentence-case description with no full
stop:

    feat: rewrite translate as a checkpoint-based CLI
    fix: only count top-ranked terminated beams as finished
    docs: match the generation page to the repo's voice

Types in use: feat, fix, chore, docs, refactor, test.

Add a body only when the change needs explaining. Use `-` bullets, say why
before how, and call out behaviour changes.

## Pull requests

Use the same style as a commit subject for the title. Open the body with a
sentence or two on what changed, then bullets for the details, and finish with
a `Tests:` line. Link related issues or PRs.

## Never

- No `Co-Authored-By` trailers.
- No "Generated with Claude Code" or similar attribution, in commits or PRs.
