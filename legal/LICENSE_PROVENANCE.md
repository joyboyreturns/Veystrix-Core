# License provenance audit

Status: **audit of what the repository shows. Not legal advice. Items marked OPEN need
the owner's confirmation or a lawyer.** Audited 2026-11-10 on `development3.8` (`a36840a`).

## 1. What the repository establishes

| Fact | Evidence |
|---|---|
| The tree was MIT-licensed from its first release. `LICENSE` is the MIT text with the upstream author's copyright line. | `git log --diff-filter=A -- LICENSE`: added in `5cc8c6a` "Release FlashNext 0.1.0", 2026-08-27, by the upstream author. |
| `pyproject.toml` declared `license = "MIT"` and the upstream author; README and URLs pointed at the upstream project. | `pyproject.toml` at `main@10297d8`. |
| The repository was public (`joyboyreturns/Veystrix`, MIT per GitHub's license detection) until 2026-11-10, when it was made private and renamed `Veystrix-Core`. It had 0 stars, 0 forks and no releases at that time. | `gh repo view` before the change. |
| The history holds 287 commits on `main` (296 across all refs). | `git shortlog -sne --all` |

Commit authorship (all refs):

| Author identity | Commits | Dates | Notes |
|---|---:|---|---|
| Upstream author | 204 | 2026-08-27 .. 2026-09-23 | Original FlashNext / VEYSTRIX code. Holder named in the MIT notice. |
| `Vincent` (local identity) | 70 | 2026-11-06 .. 2026-11-10 | Current development (3.8 line). |
| `Gio Mantovani` | 10 | 2026-08-31 .. 2026-09-03 | 9 are merge commits of the upstream author pull requests; 1 adds `docs/MLX/MLX 0.32.2 Metal backend, read against the Flash-N...md` (internal research document). |
| `joyboyreturns` | 3 | 2026-11-08 | Documentation commits on branch `docs/chat-agentic-guides-20261108` (not part of `main`). |
| `Veystrix Review Preparation` | 6 | 2026-11 | Synthetic review-only commits on `review/*` branches. Not product code. |
| `dependabot[bot]` | 3 | 2026-08-31 | Dependency bumps. |

## 2. What follows

1. **The MIT portions stay MIT.** Anyone who received the MIT-licensed code (it was public)
   keeps the rights MIT gave them. A later proprietary license cannot withdraw that, and
   Veystrix does not claim it does. The MIT notice must stay with the code.
2. **A proprietary license for the combined work is permitted** under MIT as long as the MIT
   copyright and permission notice are kept (MIT allows sublicensing and use in proprietary
   products). What Veystrix can legitimately license as proprietary is therefore:
   * code and documentation written by the owner after the MIT release (the `Vincent` and
     `joyboyreturns` commits), as far as it is original. It is a **derivative** of the MIT code, so
     the owner's license covers the owner's own contribution and the combined work, not the
     underlying MIT portions;
   * the Swift application, packaging and release tooling written new for Veystrix;
   * the Veystrix name, logo and icons (trademark/branding, no MIT grant covers them).
   It cannot be claimed as exclusively owned: the the upstream author code (OPEN 1), MLX / MLX-LM derived passages, Sparkle, and every dependency.
3. **The source of the proprietary binary is still partly MIT.** The freeware license therefore
   says so (section 6 and 12 of `legal/FREEWARE-LICENSE.txt`) and ships `THIRD_PARTY_NOTICES.md`.

## 3. OPEN items for the owner / a lawyer

| # | Question | Why it matters | Suggested action |
|---|---|---|---|
| 1 | Who is the upstream author? | **Evidence points to a person other than the current owner.** 25 historical benchmark scripts under `results/flashnext/2026092*/run.sh` do `cd /Users/gioma/Developer/VEYSTRIX`, i.e. they were run from a different user account (`gioma`), and the commit identity `Gio Mantovani` merges the upstream author pull requests. The current owner's machine is `/Users/vincent`. This is an inference from the repository, not a fact the repository states. If the upstream author is a different person, the owner holds the code under the MIT grant only: it may be built on, modified and distributed inside a proprietary product with the notice kept, but it is **not** the owner's to relicense or to describe as owner-authored. | Owner confirms who the upstream author / `Gio Mantovani` is and whether that person agrees to being credited as in `THIRD_PARTY_NOTICES.md` (credit is required by the MIT license regardless). Lawyer reviews the combined-work position. Keep the notice either way. |
| 2 | `Gio Mantovani`: under what terms did they contribute? | 9 of 10 commits merge the upstream author PRs, so this is probably the same person as the upstream author (see 1). The 1 real change is an internal research document. Contributions to a public repo that carries a license are made under that license (GitHub Terms of Service D.6), but a written confirmation is better. | Covered by item 1. The document is private and excluded from public releases either way. |
| 3 | AI-assisted code. Branch names (`codex/...`), refs (`refs/codex/...`) and Claude Code sessions show substantial AI-assisted authorship. | Copyrightability and ownership of AI-generated material is unsettled and tool terms vary. It limits what can honestly be claimed as "owner-authored". | Lawyer review; keep human review and authorship records; do not assert exclusive authorship in marketing copy. |
| 4 | Copied/adapted upstream code. `models/flashnext/compiled.py::_ple_call` copies the body of the upstream `Qwen4ExpPLELayer.__call__` (MLX-LM/MLX-VLM, MIT). (The Bonsai-2 Metal kernels were removed together with that backend.) | Requires the upstream copyright notice and may not be claimed as original. | Identify upstream files and add their MIT notices. Notices for MLX/MLX-LM are already in `THIRD_PARTY_NOTICES.md`. |
| 5 | Checkpoint-supplied code. The K2-Horizon and Bonsai-2 backends (which loaded checkpoint code via `trust_remote_code`) were removed. | Resolved by removal; the remaining backends load no checkpoint-supplied code. | None. |
| 6 | `opencv-python` wheels bundle FFmpeg (LGPL). | Bundling an LGPL library imposes relink/notice obligations. It enters through `mlx-vlm`. | Exclude `opencv-python` (and other vision-only dependencies) from the engine bundle if the shipped checkpoints are text-only, or ship the LGPL notices and relink path. |
| 7 | Internal documents and results. `docs/` and `results/` contain benchmark evidence, absolute home-directory paths (~1,100) and handoff notes. | Not licensed for public redistribution; some paths reveal the owner's username. | Keep private. The public repository contains none of them. |
| 8 | Name. "Veystrix" trademark clearance has not been done. | Public branding. | Owner decision. |
| 9 | Contributions accepted after the repo went proprietary. `CONTRIBUTING.md` describes MIT inbound terms. | Misleading once the repo is private/proprietary. | Updated in this branch to say contributions are not accepted. |

## 4. Model checkpoints

No weights are in the repository, the app or the release artifacts. Each checkpoint has its own
publisher license (`NOTICE`); the app shows a reminder and never downloads weights by itself.
