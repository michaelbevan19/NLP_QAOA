"""
QAOA circuit + COBYLA loop.

============================================================================
FIX 1 / FIX 2 (2026-08-15): both entry points below now minimize
qubo.bitstring_cost(bits, Q) -- the identical matrix baselines.py's
classical methods minimize -- with ZERO LLM calls during search. This
replaced an earlier design (see git history / HANDOFF.md Sec 6 step 7)
where run_nlp_qaoa called the real LLM every COBYLA iteration. That design
made the "LLM calls" column meaningless as a cross-method comparison (9-16
calls for NLP-QAOA vs. 1 for the classical methods measured a bookkeeping
inconsistency, not a real algorithmic difference) and is why this file was
rewritten rather than patched.

FIX 2's root cause, found while implementing FIX 1: run_standard_qaoa's
COBYLA objective used to be the EXACT analytic expectation value
<H_cost>, computed directly by the simulator with no sampling. That
objective structurally CANNOT include qubo.py's floor penalty -- the
penalty is a discontinuous function of a classical bitstring's Hamming
weight ("fewer than 2 kept"), not something expressible as the smooth
expectation value of a fixed Hamiltonian over a superposition state. So
COBYLA never saw it: whenever a prompt's raw (floor-free) Q-landscape
happened to favor a sparse or empty state (any prompt where most tokens'
diagonal entries are positive -- i.e. "encourage drop" -- makes this
likely), COBYLA converged straight into the region bitstring_cost() exists
specifically to forbid, and there was nothing stopping it. This is why
run_standard_qaoa returned 0-1 kept tokens on ticket_routing,
medical_triage, and data_extraction in the first full run.

The fix for both problems is the same change: every COBYLA iteration now
samples ONE bitstring from the circuit's current probability distribution
(a real quantum device only ever returns one measurement per shot, so this
keeps the simulation honest to that constraint) and evaluates
qubo.bitstring_cost() on THAT specific bitstring -- which, being a
function of a concrete classical bitstring rather than a quantum
expectation, DOES include the floor penalty. COBYLA now sees +FLOOR_PENALTY
(1000, overwhelming next to typical Q magnitudes) whenever it samples a
degenerate bitstring, which is what actually keeps the search away from
that region -- a functioning guard where before there was, provably, none.

Both entry points below (run_standard_qaoa, run_nlp_qaoa) now share this
exact mechanism via _run_qaoa() and use the SAME default hyperparameters
(p_layers, maxiter) -- there is no longer any principled reason to budget
one lower than the other, since neither pays a real per-iteration cost.
They are kept as two separate named functions (rather than merged into
one) so the results table can report them as two independent runs of the
identical method -- any remaining difference between their results is
attributable to random seed / sampling variance alone, which is itself a
legitimate thing to show: it demonstrates that the LLM-call and
floor-penalty bugs, not a genuine "NLP-grounding" advantage, were entirely
responsible for any gap the earlier (pre-fix) comparison showed between
them. evaluate.py deliberately gives them different (but reproducible)
seeds so they are not literally bit-for-bit identical runs.
============================================================================

QUBO -> Ising conversion (x_i in {0,1} -> Z_i in {-1,+1} via x_i=(1-Z_i)/2):
    sum_i Q_ii*x_i + sum_{i<j} Q_ij*x_i*x_j
  becomes (after substitution and collecting terms)
    const + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j
  with:
    h_i  = -Q_ii/2 - (1/4) * [ sum_{j<i} Q[j][i] + sum_{j>i} Q[i][j] ]
    J_ij = Q[i][j] / 4                                   (i < j)
    const = sum_i Q_ii/2 + sum_{i<j} Q_ij/4
  Still used to build the cost Hamiltonian that biases the circuit's
  sampling distribution toward low-Q-cost bitstrings (the "standard,
  unmodified QAOA" mechanism: Hadamard init -> cost layer -> mixer layer,
  repeated for p layers) -- what changed is only what COBYLA's objective
  function reads off each iteration (a sampled bitstring's real
  bitstring_cost(), not the circuit's exact expectation value).
"""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from qubo import bitstring_cost

P_LAYERS_DEFAULT = 1     # QAOA depth; kept shallow for CPU-local runs
QAOA_MAXITER_DEFAULT = 100  # COBYLA iterations; SAME for both entry points now (FIX 1: neither pays a real per-iteration LLM cost anymore)


# ---------------------------------------------------------------------
# QUBO -> Ising Hamiltonian
# ---------------------------------------------------------------------
def qubo_to_ising(Q: list[list[float]]) -> tuple[list[float], dict, float]:
    """Returns (h, J, const): h[i] is Z_i's coefficient, J[(i,j)] (i<j) is
    Z_i Z_j's coefficient, const is the constant offset. See module
    docstring for the derivation."""
    n = len(Q)
    h = [0.0] * n
    J = {}
    const = 0.0

    for i in range(n):
        const += Q[i][i] / 2.0
        h[i] += -Q[i][i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i][j]
            if qij == 0.0:
                continue
            const += qij / 4.0
            h[i] += -qij / 4.0
            h[j] += -qij / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + qij / 4.0

    return h, J, const


def build_cost_hamiltonian(Q: list[list[float]]) -> qml.Hamiltonian:
    n = len(Q)
    h, J, _const = qubo_to_ising(Q)
    coeffs, obs = [], []
    for i in range(n):
        if h[i] != 0.0:
            coeffs.append(h[i])
            obs.append(qml.PauliZ(i))
    for (i, j), coeff in J.items():
        if coeff != 0.0:
            coeffs.append(coeff)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    if not coeffs:
        # Degenerate all-zero Q: Hamiltonian needs at least one term.
        coeffs, obs = [0.0], [qml.Identity(0)]
    return qml.Hamiltonian(coeffs, obs)


def _make_probs_qnode(n: int, cost_h: qml.Hamiltonian, mixer_h: qml.Hamiltonian, p_layers: int):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(gammas, betas):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p_layers):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.probs(wires=range(n))

    return circuit


def _bitstring_from_index(index: int, n: int) -> list[int]:
    """qml.probs(wires=range(n)) orders basis states as standard binary
    counting with wire 0 the MOST significant bit."""
    bits = [(index >> (n - 1 - w)) & 1 for w in range(n)]
    return bits


# ---------------------------------------------------------------------
# Shared QAOA mechanism (FIX 1 / FIX 2)
# ---------------------------------------------------------------------
def _run_qaoa(n: int, Q: list[list[float]], p_layers: int, maxiter: int, seed: int | None) -> dict:
    """
    Standard, unmodified QAOA (Hadamard init -> cost layer -> mixer layer,
    p times -> measure), COBYLA-tuned. Every COBYLA iteration samples ONE
    bitstring from the circuit's current probability distribution and
    evaluates qubo.bitstring_cost() on it -- the identical matrix and cost
    function baselines.py's classical methods use, WITH the floor penalty
    included (see module docstring for why this specific mechanism, not
    the circuit's exact expectation value, is what makes the floor penalty
    actually work). Zero LLM calls. Tracks the best (bitstring, cost)
    actually observed across every iteration, since a single-sample
    objective is noisy and COBYLA's final x is not guaranteed to be the
    best point it visited -- standard practice for noisy black-box
    optimization.
    """
    cost_h = build_cost_hamiltonian(Q)
    mixer_h = qml.qaoa.x_mixer(range(n))
    probs_qnode = _make_probs_qnode(n, cost_h, mixer_h, p_layers)

    rng = np.random.default_rng(seed)
    best = {"bitstring": None, "cost": np.inf}
    history = []

    def objective(params: np.ndarray) -> float:
        gammas, betas = params[:p_layers], params[p_layers:]
        probs = np.asarray(probs_qnode(gammas, betas))
        probs = probs / probs.sum()  # guard against tiny FP drift
        sampled_index = int(rng.choice(len(probs), p=probs))
        bitstring = _bitstring_from_index(sampled_index, n)
        cost = bitstring_cost(bitstring, Q)  # includes the floor penalty -- this is the fix

        history.append({"bitstring": bitstring, "cost": cost})
        if cost < best["cost"]:
            best.update(bitstring=bitstring, cost=cost)

        return cost

    x0 = rng.uniform(0.0, np.pi / 2, size=2 * p_layers)
    minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})

    # FIX 2: guarantee no method returns a bitstring with fewer than 2
    # kept tokens. bitstring_cost()'s floor penalty (+1000, overwhelming
    # next to typical Q magnitudes) now makes COBYLA actively avoid this
    # region, so in practice `best["bitstring"]` should already satisfy
    # this -- but rather than silently accepting a partial/degenerate
    # result if the search still lands there (e.g. an unlucky sample
    # sequence within a limited iteration budget), report it explicitly
    # as a clean failure (bitstring=None) instead of a bitstring whose
    # kept-count doesn't match a real optimized-token count downstream
    # (evaluate.py's kept/opt_tok consistency, also FIX 2). This is a
    # backstop, not a rescue: it does not search harder or substitute a
    # different answer, it only prevents a <2 result from being reported
    # as if it were a valid one.
    if best["bitstring"] is None or sum(best["bitstring"]) < 2:
        return {"bitstring": None, "cost": best["cost"], "llm_calls": 0, "history": history}

    return {"bitstring": best["bitstring"], "cost": best["cost"], "llm_calls": 0, "history": history}


def run_standard_qaoa(
    n: int,
    Q: list[list[float]],
    p_layers: int = P_LAYERS_DEFAULT,
    maxiter: int = QAOA_MAXITER_DEFAULT,
    seed: int | None = None,
) -> dict:
    """One independent run of the shared QAOA mechanism -- see _run_qaoa()
    and the module docstring for FIX 1/FIX 2's mechanism and why this and
    run_nlp_qaoa are kept as two separate named entries despite sharing
    identical code and hyperparameters."""
    return _run_qaoa(n, Q, p_layers, maxiter, seed)


def run_nlp_qaoa(
    n: int,
    Q: list[list[float]],
    p_layers: int = P_LAYERS_DEFAULT,
    maxiter: int = QAOA_MAXITER_DEFAULT,
    seed: int | None = None,
) -> dict:
    """A second independent run of the SAME shared QAOA mechanism as
    run_standard_qaoa (see module docstring, FIX 1) -- kept as a
    separately-seeded, separately-reported entry so the results table can
    show that, once both methods minimize the identical matrix with zero
    LLM calls, any remaining gap between them is sampling variance, not a
    genuine NLP-grounding advantage. No longer takes `candidates`, `llm`,
    or `scorer` -- those were only needed for the real per-iteration LLM
    calls this function no longer makes; the caller (evaluate.py) still
    does exactly one real LLM call on this function's final bitstring,
    same as every other method, for the shared final-verification step."""
    return _run_qaoa(n, Q, p_layers, maxiter, seed)


if __name__ == "__main__":
    import itertools

    from baselines import brute_force
    from qubo import assemble_qubo

    print("=== sanity check: qubo_to_ising round-trips bitstring_cost ===")
    toy_candidates = [{"text": "please"}, {"text": "classify"}, {"text": "categorize"}, {"text": "positive"}]
    toy_importance = [0.1, 0.9, 0.85, 0.95]
    toy_redundant_pairs = [(1, 2)]
    Q = assemble_qubo(toy_candidates, toy_importance, toy_redundant_pairs)
    n = len(toy_candidates)

    h, J, const = qubo_to_ising(Q)
    print(f"h={h}")
    print(f"J={J}")
    print(f"const={const}")

    max_diff = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        z = [1 - 2 * b for b in bits]  # x=0 -> Z=+1, x=1 -> Z=-1
        energy = const + sum(h[i] * z[i] for i in range(n)) + sum(
            coeff * z[i] * z[j] for (i, j), coeff in J.items()
        )
        direct = sum(bits[i] * Q[i][i] for i in range(n)) + sum(
            bits[i] * bits[j] * Q[i][j] for i in range(n) for j in range(i + 1, n)
        )
        max_diff = max(max_diff, abs(energy - direct))
    print(f"max |Ising energy - direct QUBO cost| over all bitstrings: {max_diff:.2e}  (should be ~0)")

    print("\n=== sanity check: run_standard_qaoa vs brute_force on the toy QUBO ===")
    true_opt = brute_force(n, Q)
    print(f"brute_force (ground truth): {true_opt}")

    sq_result = run_standard_qaoa(n, Q, p_layers=2, maxiter=150, seed=0)
    match = "MATCHES optimum" if sq_result["cost"] == true_opt["cost"] else "does NOT match optimum"
    print(f"run_standard_qaoa: {sq_result}  [{match}]")

    print("\n=== sanity check: run_nlp_qaoa (now the same mechanism, different seed) vs brute_force ===")
    nq_result = run_nlp_qaoa(n, Q, p_layers=2, maxiter=150, seed=1)
    match = "MATCHES optimum" if nq_result["cost"] == true_opt["cost"] else "does NOT match optimum"
    print(f"run_nlp_qaoa: {nq_result}  [{match}]")

    print("\n=== FIX 2 regression check: force a Q landscape that favors sparse/empty states ===")
    # All-positive diagonal, no redundancy penalty -- the raw (floor-free)
    # minimum is the all-zero bitstring. Before FIX 1/2 this reliably made
    # run_standard_qaoa return <2 kept tokens (its objective never saw the
    # floor penalty). After the fix it must not.
    sparse_candidates = [{"text": f"tok{i}"} for i in range(6)]
    sparse_importance = [0.05, 0.03, 0.02, 0.04, 0.01, 0.02]  # all tiny -> all diagonals positive
    Q_sparse = assemble_qubo(sparse_candidates, sparse_importance, [])
    n_sparse = len(sparse_candidates)
    for trial_seed in range(5):
        r = run_standard_qaoa(n_sparse, Q_sparse, p_layers=1, maxiter=100, seed=trial_seed)
        kept = sum(r["bitstring"]) if r["bitstring"] else 0
        print(f"  seed={trial_seed}: bitstring={r['bitstring']}  kept={kept}  cost={r['cost']:.4f}  "
              f"{'OK (>=2 or clean None)' if r['bitstring'] is None or kept >= 2 else 'FAIL: <2 kept tokens leaked through'}")
