# AGENTIC prefill readiness: a startup admission check beyond the 9 GiB gate (2026-11-08)

Code: `models/flashnext/prefill_readiness.py` (pure functions plus a status command). Tests: `models/flashnext/tests/unit/test_prefill_readiness.py`.
Applies to AGENTIC only. CHAT, its 0 GB pool baseline and every runtime guard are unchanged. Nothing was run on the full checkpoint.

```sh
python -m models.flashnext.prefill_readiness          # the verdict and every memory category on this machine; starts nothing, purges nothing
python -m models.flashnext.prefill_readiness --json
./chat.sh --mode agentic --pool-gb 0 --auto-purge      # now refuses to load (exit 69) when the verdict is REFUSED; record in ~/.flashnext/admission-log.jsonl
```

## Why the 9 GiB gate was not enough
The gate is `free + inactive + speculative`. Three real runs started from nearly the same gate figure and ended three ways:

| start state (guard's first sample) | gate | other anonymous | outcome | old rule | new rule |
|---|---|---|---|---|---|
| AGENTIC smoke, 898-token request | 9.59 GiB | 6.31 GiB | pressure level 2 stop in cold prefill | admit | **refuse** (headroom 1.40 GiB) |
| CHAT pool 0, five answers | 9.70 GiB | 5.58 GiB | passed | admit | refuse (headroom 1.89 GiB), a documented false positive |
| first CHAT, 2 GB pool | 9.15 GiB | 6.36 GiB | compressor stop | admit | **refuse** (headroom 1.47 GiB) |

The gate counts inactive pages as available. Inactive *anonymous* pages are other applications' idle memory that macOS must compress before reuse, and the first prefill triggered exactly that: anonymous memory fell 4.23 GiB while the compressor grew 2.91 GiB (ratio 0.69) in the first 2.5 s of the failed run's prefill. The gate cannot see how large that pool is.

## What macOS reports, and what it does not
Reported (`vm_stat`, `sysctl`): free, active, inactive, speculative (file-backed), wired, purgeable, file-backed and anonymous totals, compressor size, swap used, swap-out counter, pressure level; `top` gives per-process resident and compressed memory. **Not reported:** the split of active and inactive pages into file-backed and anonymous. Only bounds follow from `file_backed + anonymous == active + inactive + speculative`, and the report prints them (for example inactive anonymous between 0.00 and 2.74 GiB on today's machine). No check relies on an exact split, and anonymous memory is not subtracted wholesale: the budget below charges 0.75 of it, from the measured compression ratio.

## The rule
All must hold; the first is the unchanged mandatory minimum.
1. **Start gate:** `free + inactive + speculative >= 9 GiB` (9,663,676,416 bytes).
2. **Pressure and swap:** pressure level 1 and no swap-out between two readings one second apart.
3. **Cold-prefill budget:** `headroom = physical - (wired + compressor + 6.0 GiB cold prefill + 1.0 GiB harness + 0.75 x anonymous) >= 4.0 GiB`.

Provenance of the constants (observations, not a fit; there is no passing AGENTIC run to calibrate against):
* 6.0 GiB: MLX peak 4.62 GiB (898 tokens, failed run) to 5.4 GiB (2,046 tokens, earlier runs), plus about 0.3 GiB of process memory, rounded up.
* 1.0 GiB: the harness client. The headless client measured 0.19 GiB peak; the desktop app is unmeasured, so this is a generous allowance.
* 0.75: observed compression ratio 0.69, rounded up.
* 4.0 GiB: the 3 GiB guard floor plus 0.66 GiB above the 3.34 GiB available at which pressure level 2 first appeared.
File-backed memory is not in the budget, so a cache purge cannot change the verdict. Equivalently the machine passes only if other applications hold at most about 2.8 GiB of anonymous memory (with wired and compressor near 1.9 and 1.0 GiB).

## Purge interplay
When rule 2 or 3 fails, a purge cannot admit the launch, so none runs: `ensure_startup` returns at once instead of waiting 90 s, and repeated attempts all skip (tested for 50 attempts, zero purges). When only the gate is short and file cache can close the gap, the existing guarded purge runs as before. The idle monitor is unchanged.

## Refusal output
Names the category and the action. Anonymous-bound: "Other processes hold X GiB of anonymous memory that macOS will compress ... Quit applications until it is at least Y GiB lower ... A cache purge does not help", plus the largest processes from `top`. Pressure-bound: wait or quit applications, do not purge. Gate-bound with recoverable file cache: use `--auto-purge`. Exit code 69, nothing loaded, no guard weakened.

## Risks
* **False positives:** states that would have run are refused (the CHAT pool-0 start above). The rule is deliberately conservative.
* **False negatives:** the budget cannot see file-cache churn (about 39 GB of page-ins in the first 45 s of the failed run), prefill duration, thermals, or applications that grow after admission. The desktop harness is unmeasured. Admission therefore does not guarantee no pressure stop; the runtime guards remain the only protection during the run.
* **Calibration:** three traces, two with the same outcome class. Treat the constants as a conservative starting point to be revised by the next run's admission log, not as a model.
