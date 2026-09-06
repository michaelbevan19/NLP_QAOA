"""
QUBO assembly (report Section 4.5) and bitstring cost evaluation.

============================================================================
SIGN CONVENTION -- read this before touching any Q[i][i] or Q[i][j] value
anywhere in this project. Every other file (circuit.py, baselines.py,
evaluate.py) must match it.

  QUBO MINIMIZES.  x_i = 1 means "KEEP token i"; x_i = 0 means "drop it".

      total_cost(x) = sum_i  x_i * Q[i][i]
                    + sum_{i<j}  x_i * x_j * Q[i][j]

  Diagonal Q[i][i] -- token importance (Section 4.3):
      Q[i][i] = -(importance_i * IMPORTANCE_SCALE) + LENGTH_PENALTY
      - importance_i in [0, 1] comes from the removal test (scoring.py):
        1 = output changes a lot when the token is removed (important),
        0 = output barely changes when removed (filler).
      - NEGATIVE Q[i][i]  -> keeping token i LOWERS total cost -> the
        solver is rewarded for keeping it (an important token whose
        importance outweighs the flat length penalty).
      - POSITIVE Q[i][i]  -> keeping token i RAISES total cost -> the
        solver is rewarded for dropping it (filler, or a token whose
        importance was too small to outweigh LENGTH_PENALTY).

  Off-diagonal Q[i][j] -- redundancy (Section 4.4), i < j only:
      Q[i][j] = alpha * NLP_penalty(i, j)   -- ALWAYS >= 0.
      - If BOTH tokens of a redundant pair are kept (x_i = x_j = 1), this
        POSITIVE term RAISES total cost -> the solver is pushed to drop
        at least one of the pair.
      - Non-redundant pairs get Q[i][j] = 0 (no interaction term).

  Summary: negative diagonal = encourage keep. Positive diagonal =
  encourage drop. Off-diagonal penalties are always >= 0 and only ever
  discourage co-selection -- they never encourage it.
============================================================================

bitstring_cost() also adds a FLOOR PENALTY when a candidate bitstring
keeps fewer than 2 tokens (guard against QAOA collapsing to an empty or
single-token prompt). This is deliberately NOT baked into the Q matrix
itself: "keep at least 2" is a cardinality constraint, and a general
cardinality constraint cannot be encoded exactly as a pairwise (quadratic)
term without extra auxiliary qubits. Instead it's applied as a large fixed
penalty whenever a bitstring's cost is computed, so the brute-force
checkpoint, the classical baselines, and the QAOA loop all see the same
guard.
"""

IMPORTANCE_SCALE = 10.0   # matches the report's Figure 3 worked-example scale
# LENGTH_PENALTY was 3.0 (copied straight from the report's toy Figure-3
# example) until the Sec 5 step 7 checkpoint on a REAL prompt (code_review,
# Windows run 2026-08-13) exposed a diagonal-dominance problem: real
# removal-test importance scores for that 12-candidate prompt topped out at
# 0.109 ("function"), so even the single most important token scored
# -(0.109*10)+3.0 = +1.91 -- still positive ("encourage drop"). EVERY
# token's diagonal came out positive, so brute_force() wasn't actually
# choosing which tokens mattered; it just collapsed to bitstring_cost()'s
# 2-token floor almost by construction, picking whichever 2 tokens had the
# least-bad diagonal. Lowered to 1.0 so tokens with importance >~0.10 (like
# that run's "function") go negative and get rewarded for staying, while
# true filler (importance <0.01, e.g. that run's "Python"=0.008,
# "know"=0.008) still goes positive. Re-verify against real removal-test
# scores whenever the importance range looks very different from this.
LENGTH_PENALTY = 1.0      # flat per-token cost added to every diagonal entry
# ALPHA was 4.0 (an uncalibrated default) until a magnitude audit on real
# code_review data (2026-09-04) showed the redundancy term was carrying
# ~99% of the objective's total mass and drowning out the importance
# signal entirely:
#
#   total redundancy cost, all 12 tokens kept:  28 flagged pairs x 4.0 = +112.0
#   total diagonal contribution, all kept:                              +0.68
#
# The diagonal itself was already well balanced -- importance rewards
# (-11.3) very nearly cancel the flat length penalties (+12.0), netting
# +0.68. Redundancy was the term out of scale, by roughly 10x. With it
# that large the solver is not trading linguistic value against length at
# all; it is simply fleeing flagged pairs, which is what pinned every
# method against bitstring_cost()'s 2-token floor on real prompts.
#
# Recalibrated so redundancy MODULATES rather than dominates: its total
# should sit in the same order as the diagonal spread (~10), not ~112.
#   28 pairs x ALPHA ~= 10  ->  ALPHA ~= 0.36
# Rounded to 0.4. Note this is a per-pair FLAT penalty (see the
# Q[i][j] line below), so it scales with how many pairs redundancy.py
# flags -- prompts flagging more pairs feel proportionally more of it.
ALPHA = 0.4               # default redundancy-penalty weight (swept, Section 5 step 12)
FLOOR_PENALTY = 1000.0    # large fixed cost if fewer than 2 tokens are kept

# --- soft cardinality penalty (added 2026-09-04) ------------------------
# WHY THIS EXISTS. The floor penalty above is a HARD guard at 2 tokens,
# applied post-hoc in bitstring_cost(). It stops total collapse but it does
# not stop the solver from parking right on top of it -- and that is
# exactly what was observed on real prompts: every method pinned at 2-3
# kept tokens. That is not the optimizer balancing a trade-off, it is the
# optimizer hitting the only wall in the objective.
#
# The cause is a magnitude mismatch, measurable in the real code_review
# numbers: importance scores run 0.008-0.109 (mean ~0.048), so the diagonal
# reward for keeping a token, -(imp * IMPORTANCE_SCALE) + LENGTH_PENALTY,
# is at BEST about -0.09 and is positive (i.e. "please drop me") for 11 of
# 12 tokens. Meanwhile ONE flagged redundant pair costs +ALPHA = +4.0, and
# real prompts flag 20-33 pairs. Keeping everything therefore costs on the
# order of +138 against a best-case diagonal reward under -1. Dropping to
# the floor is not a bug in the solver; it is the correct answer to the
# question the objective is actually asking.
#
# A soft cardinality term gives the objective a reason to sit somewhere
# other than the wall. lam * (sum(x) - t)^2 expands, using x_i^2 = x_i for
# binary x, into terms this QUBO can hold exactly:
#
#   lam*(sum(x) - t)^2
#     = lam*(1 - 2t) * sum_i x_i        <- a per-token DIAGONAL shift
#     + 2*lam        * sum_{i<j} x_i x_j <- a uniform OFF-DIAGONAL term
#     + lam*t^2                          <- a constant (ignored; adding a
#                                           constant never moves an argmin)
#
# NOTE this does NOT contradict the module docstring above, which says a
# cardinality constraint "cannot be encoded exactly as a pairwise term
# without extra qubits". That statement is about a HARD constraint (a step
# function at a threshold, which genuinely needs slack/ancilla qubits).
# A SOFT quadratic preference is exactly representable, as shown -- so it
# lives inside Q, unlike FLOOR_PENALTY which must stay post-hoc.
#
# VALUE CHOSEN FROM MEASURED DATA (real code_review importances on the
# raw-cosine scale, real 28-pair flagged set), not guessed:
#
#   lambda    t=40%    t=50%    t=60%     <- tokens kept by the exact optimum
#     0.00     2/12     2/12     2/12     <- the collapse
#     0.05     4/12     5/12     5/12
#     0.10     4/12     5/12     6/12     <- target actually tracked
#     0.25     5/12     5/12     6/12
#
# 0.10 is the smallest weight at which the target fraction is genuinely
# respected (40/50/60% -> 4/5/6 tokens) rather than ignored. Deliberately
# NOT larger: at the previous ALPHA=4.0 the same sweep needed lambda>=2 to
# move the target at all, and at that size the cardinality term contributes
# ~264 against redundancy's ~112 -- i.e. it stops being a preference and
# becomes the whole objective, reducing the QUBO to a token-counter and
# discarding the importance/redundancy signal the pipeline exists to
# produce. At lambda=0.10 the three forces are peers by construction:
#   redundancy   28 pairs x ALPHA 0.4      = 11.2
#   cardinality  66 pairs x 2*lambda       = 13.2
#   diagonal     importance -11.3 + length +12.0, per-token range -1.17..+0.84
#
# HONEST CAVEAT: there is no ground truth for the "correct" number of
# tokens to keep, so this does not make the result correct -- it makes it
# non-degenerate and jointly driven by all three signals instead of pinned
# against bitstring_cost()'s floor by one runaway term. Whether output
# similarity actually holds up at ~5 tokens is a separate empirical
# question, answered by a real run, not by this constant.
CARDINALITY_WEIGHT = 0.1   # lam; 0.0 disables the term entirely
CARDINALITY_TARGET_FRACTION = 0.5  # t = this fraction of n candidates


def assemble_qubo(
    candidates: list[dict],
    importance_scores: list[float],
    redundant_index_pairs: list[tuple[int, int]],
    alpha: float = ALPHA,
    importance_scale: float = IMPORTANCE_SCALE,
    length_penalty: float = LENGTH_PENALTY,
    cardinality_weight: float = CARDINALITY_WEIGHT,
    cardinality_target_fraction: float = CARDINALITY_TARGET_FRACTION,
) -> list[list[float]]:
    """
    candidates:             clean.py candidate dicts, length n (only used
                             for n; kept as a parameter so callers don't
                             have to pass n separately and matrix/token
                             identity stay obviously paired at the call
                             site).
    importance_scores:      list[float] length n, in [0, 1].
    redundant_index_pairs:  list[(i, j)] with i < j, from
                             redundancy.redundant_pairs().
    cardinality_weight:     lam for the soft cardinality penalty (see the
                             comment block above). 0.0 disables it.
    cardinality_target_fraction:
                            target kept-token count as a fraction of n.
    Returns an n x n matrix (list of lists). Only Q[i][i] and Q[i][j] for
    i < j are populated; Q[j][i] for j > i is left at 0 and never read
    (upper-triangular storage, matching Figure 3 of the report).
    """
    n = len(candidates)
    assert len(importance_scores) == n
    Q = [[0.0] * n for _ in range(n)]

    for i in range(n):
        Q[i][i] = -(importance_scores[i] * importance_scale) + length_penalty

    for (i, j) in redundant_index_pairs:
        assert i < j, "redundant_index_pairs must be given as (i, j) with i < j"
        Q[i][j] += alpha * 1.0  # NLP_penalty(i, j): fixed "flagged" unit; alpha carries the weight

    if cardinality_weight:
        target = cardinality_target_fraction * n
        diag_shift = cardinality_weight * (1.0 - 2.0 * target)
        pair_shift = 2.0 * cardinality_weight
        for i in range(n):
            Q[i][i] += diag_shift
            for j in range(i + 1, n):
                Q[i][j] += pair_shift

    return Q


def bitstring_cost(bitstring: list[int], Q: list[list[float]], floor_penalty: float = FLOOR_PENALTY) -> float:
    """
    bitstring: sequence of 0/1, length n (1 = keep token i).
    Returns total_cost(x) as defined in the sign-convention docstring
    above, plus floor_penalty if fewer than 2 tokens are kept.
    """
    n = len(bitstring)
    cost = 0.0
    for i in range(n):
        if bitstring[i]:
            cost += Q[i][i]
    for i in range(n):
        if not bitstring[i]:
            continue
        for j in range(i + 1, n):
            if bitstring[j]:
                cost += Q[i][j]
    if sum(bitstring) < 2:
        cost += floor_penalty
    return cost


def kept_indices(bitstring: list[int]) -> list[int]:
    return [i for i, b in enumerate(bitstring) if b]


if __name__ == "__main__":
    import itertools

    print("=== sanity check 1: report Figure 3 worked example ===")
    # Reproduces the report's 4-token example (You, classify, categorize,
    # positive) directly as a Q matrix, to check bitstring_cost()'s sign
    # convention against the document's own numbers.
    Q_fig3 = [[0.0] * 4 for _ in range(4)]
    Q_fig3[0][0], Q_fig3[1][1], Q_fig3[2][2], Q_fig3[3][3] = -11, -9, -8, -10
    Q_fig3[0][1], Q_fig3[0][2], Q_fig3[0][3] = -3, -2, -1
    Q_fig3[1][2], Q_fig3[1][3] = 12, -6
    Q_fig3[2][3] = -5

    for bits in itertools.product([0, 1], repeat=4):
        print(list(bits), "-> cost", bitstring_cost(list(bits), Q_fig3))

    best = min(itertools.product([0, 1], repeat=4), key=lambda b: bitstring_cost(list(b), Q_fig3))
    print(f"\nlowest-cost bitstring: {best}")
    print(
        "Note: with these exact Figure-3 numbers the global minimum keeps ALL "
        "four tokens, including the redundant classify+categorize pair -- their "
        "individual importance (-9, -8) outweighs the +12 co-selection penalty. "
        "This isn't a bug: it demonstrates why alpha (the redundancy weight) has "
        "to be tuned relative to the importance scale (report Section 5, step 12) "
        "-- a fixed alpha that's too small relative to importance will never "
        "actually cause a redundant pair to be dropped."
    )

    print("\n=== sanity check 2: assemble_qubo() wiring ===")
    toy_candidates = [{"text": "please"}, {"text": "classify"}, {"text": "categorize"}, {"text": "positive"}]
    toy_importance = [0.1, 0.9, 0.85, 0.95]   # "please" is filler; the rest matter
    toy_redundant_pairs = [(1, 2)]            # classify/categorize flagged redundant
    Q = assemble_qubo(toy_candidates, toy_importance, toy_redundant_pairs)
    for row in Q:
        print(row)
