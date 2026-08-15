"""
QAOA circuit + COBYLA loop (report Section 4.6, HANDOFF.md Sec 6 step 7).

============================================================================
Two entry points, both built on the SAME standard, unmodified QAOA circuit
(Hadamard init -> Cost Hamiltonian from Q -> Mixer Hamiltonian -> repeat for
p layers -> measure). All linguistic novelty lives in Q (qubo.py); nothing
about the circuit itself is bespoke. They differ ONLY in what COBYLA's
objective function evaluates each iteration:

    run_standard_qaoa(n, Q, ...)
        COBYLA's objective is the exact expectation value <H_cost>,
        computed analytically by the simulator. Zero LLM calls -- same
        footing as baselines.py's random_search / greedy_removal /
        simulated_annealing (all four optimize the identical static Q
        matrix, isolating search-METHOD quality). This is the "standard
        QAOA" entry in the report's 5-method comparison table.

    run_nlp_qaoa(candidates, Q, llm, scorer, ...)
        COBYLA's objective is a REAL cost: each iteration samples ONE
        bitstring from the circuit's current probability distribution
        (report's "measured bitstring"), assembles it into a prompt,
        sends it through the real LLM, and combines output similarity to
        O_original with a length term into a scalar (report Section 4.6:
        "assemble candidate prompt from measured bitstring -> LLM call ->
        compare to O_original -> combine similarity + length penalty into
        a scalar cost -> COBYLA updates gamma/beta"). This is the "NLP-QAOA"
        entry -- the project's namesake method, and the only one of the 5
        compared methods that pays a real, nonzero LLM-calls cost per
        report Section 4.6's explicit description (HANDOFF.md Sec 2.2).

Why two functions sharing one circuit rather than one: the report's 5-way
comparison table (Section 6) needs an apples-to-apples "same search
strategy, with vs without real-LLM grounding" pair, mirroring how random /
greedy / annealing / standard-QAOA all already share "optimize the
identical Q matrix, zero LLM calls" (baselines.py's docstring). Splitting
the objective out is the only difference; everything else (circuit
structure, Hamiltonians, COBYLA) is identical code.
============================================================================

QUBO -> Ising conversion (x_i in {0,1} -> Z_i in {-1,+1} via x_i=(1-Z_i)/2):
    sum_i Q_ii*x_i + sum_{i<j} Q_ij*x_i*x_j
  becomes (after substitution and collecting terms)
    const + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j
  with:
    h_i  = -Q_ii/2 - (1/4) * [ sum_{j<i} Q[j][i] + sum_{j>i} Q[i][j] ]
    J_ij = Q[i][j] / 4                                   (i < j)
    const = sum_i Q_ii/2 + sum_{i<j} Q_ij/4
  `const` is included in run_standard_qaoa's reported expectation value so
  it's directly comparable to qubo.bitstring_cost(), but it doesn't affect
  where the expectation is minimized (adding a constant never changes an
  argmin), so it plays no role in run_nlp_qaoa's real-cost objective.

Guard against <2 kept tokens (same floor-penalty philosophy as
qubo.bitstring_cost(), applied post-hoc rather than baked into H_cost, for
the same reason documented there -- a cardinality constraint isn't exactly
representable as a pairwise term): if a measured bitstring keeps fewer than
2 tokens, run_nlp_qaoa charges FLOOR_PENALTY directly and skips the real
LLM call entirely (an empty/near-empty prompt has no meaningful output to
compare, so spending a real call on it would waste the very budget the
report's "bounded number of LLM calls" framing cares about).
"""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from clean import assemble_prompt, reading_order
from qubo import FLOOR_PENALTY, IMPORTANCE_SCALE, LENGTH_PENALTY, bitstring_cost, kept_indices

P_LAYERS_DEFAULT = 1        # QAOA depth; kept shallow for CPU-local runs
STANDARD_MAXITER = 100      # COBYLA iterations for run_standard_qaoa (0 LLM calls, cheap)
NLP_MAXITER_DEFAULT = 15    # COBYLA iterations for run_nlp_qaoa (1 real LLM call each)


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


# ---------------------------------------------------------------------
# Shared circuit
# ---------------------------------------------------------------------
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


def _make_expval_qnode(n: int, cost_h: qml.Hamiltonian, mixer_h: qml.Hamiltonian, p_layers: int):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(gammas, betas):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p_layers):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.expval(cost_h)

    return circuit


def _bitstring_from_index(index: int, n: int) -> list[int]:
    """qml.probs(wires=range(n)) orders basis states as standard binary
    counting with wire 0 the MOST significant bit."""
    bits = [(index >> (n - 1 - w)) & 1 for w in range(n)]
    return bits


# ---------------------------------------------------------------------
# run_standard_qaoa -- Q-expectation objective, zero LLM calls
# ---------------------------------------------------------------------
def run_standard_qaoa(
    n: int,
    Q: list[list[float]],
    p_layers: int = P_LAYERS_DEFAULT,
    maxiter: int = STANDARD_MAXITER,
    seed: int | None = None,
) -> dict:
    """
    Standard QAOA: COBYLA minimizes the exact expectation value of the cost
    Hamiltonian (no sampling noise, no LLM calls -- same footing as
    baselines.py's classical solvers). After convergence, the optimized
    circuit's most probable basis state is read off as the chosen
    bitstring.
    """
    cost_h = build_cost_hamiltonian(Q)
    mixer_h = qml.qaoa.x_mixer(range(n))
    _h, _J, const = qubo_to_ising(Q)

    expval_qnode = _make_expval_qnode(n, cost_h, mixer_h, p_layers)
    probs_qnode = _make_probs_qnode(n, cost_h, mixer_h, p_layers)

    def objective(params: np.ndarray) -> float:
        gammas, betas = params[:p_layers], params[p_layers:]
        return float(expval_qnode(gammas, betas)) + const

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi / 2, size=2 * p_layers)
    result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})

    best_gammas, best_betas = result.x[:p_layers], result.x[p_layers:]
    probs = np.asarray(probs_qnode(best_gammas, best_betas))
    best_index = int(np.argmax(probs))
    bitstring = _bitstring_from_index(best_index, n)

    return {
        "bitstring": bitstring,
        "cost": bitstring_cost(bitstring, Q),
        "expectation_at_optimum": float(result.fun),
        "params": {"gammas": best_gammas.tolist(), "betas": best_betas.tolist()},
        "llm_calls": 0,
    }


# ---------------------------------------------------------------------
# run_nlp_qaoa -- real LLM-grounded objective (report Section 4.6)
# ---------------------------------------------------------------------
def run_nlp_qaoa(
    candidates: list[dict],
    Q: list[list[float]],
    llm,
    scorer,
    p_layers: int = P_LAYERS_DEFAULT,
    maxiter: int = NLP_MAXITER_DEFAULT,
    sim_weight: float = IMPORTANCE_SCALE,
    len_weight: float = LENGTH_PENALTY,
    floor_penalty: float = FLOOR_PENALTY,
    seed: int | None = None,
) -> dict:
    """
    NLP-QAOA: the project's namesake method. Every COBYLA iteration:
      1. Runs the circuit at the current (gammas, betas), gets the exact
         probability distribution over bitstrings.
      2. Samples ONE bitstring from that distribution (the report's
         "measured bitstring" -- a real quantum device would only ever
         hand back one measurement per shot, not the full distribution;
         sampling here keeps the simulation honest to that constraint).
      3. If fewer than 2 tokens are kept, charges `floor_penalty` directly
         and skips the LLM call (an empty/near-empty prompt has nothing
         meaningful to compare, so spending real budget on it would waste
         the report's "bounded number of LLM calls" framing).
      4. Otherwise assembles the kept tokens into a prompt, calls
         llm.generate(), scores output similarity to O_original via
         `scorer` (a scoring.BaselineScorer with capture_baseline()
         already called), and returns
             real_cost = (1 - similarity) * sim_weight + num_kept * len_weight
         -- lower is better: high similarity to O_original AND few tokens
         kept both pull the cost down, mirroring qubo.py's diagonal
         formula so the two costs stay on a comparable scale.
      5. Tracks the best (bitstring, cost) actually observed across every
         iteration -- since the objective is a single noisy real-world
         sample each call, COBYLA's final x is not guaranteed to be the
         best point it visited, so the best OBSERVED point is what gets
         reported, a standard practice for noisy black-box optimization.

    `candidates` / `Q` / `llm` / `scorer` mirror the checkpoint wiring in
    HANDOFF.md Sec 6 step 6 (clean.py -> scoring.py -> redundancy.py ->
    qubo.py -> this function, instead of -> baselines.brute_force()).
    """
    n = len(candidates)
    cost_h = build_cost_hamiltonian(Q)
    mixer_h = qml.qaoa.x_mixer(range(n))
    probs_qnode = _make_probs_qnode(n, cost_h, mixer_h, p_layers)

    rng = np.random.default_rng(seed)

    best = {"bitstring": None, "cost": np.inf, "prompt": None, "output": None, "similarity": None}
    llm_calls = 0
    history = []

    def objective(params: np.ndarray) -> float:
        nonlocal llm_calls
        gammas, betas = params[:p_layers], params[p_layers:]
        probs = np.asarray(probs_qnode(gammas, betas))
        probs = probs / probs.sum()  # guard against tiny FP drift
        sampled_index = int(rng.choice(len(probs), p=probs))
        bitstring = _bitstring_from_index(sampled_index, n)
        num_kept = sum(bitstring)

        if num_kept < 2:
            cost = floor_penalty
            output, similarity, prompt = None, None, None
        else:
            kept_candidates = [candidates[i] for i in kept_indices(bitstring)]
            prompt = assemble_prompt(kept_candidates)
            output = llm.generate(prompt, scorer.sample_input)
            llm_calls += 1
            similarity = scorer.output_similarity(output)
            cost = (1.0 - similarity) * sim_weight + num_kept * len_weight

        history.append({"bitstring": bitstring, "cost": cost})
        if cost < best["cost"]:
            best.update(bitstring=bitstring, cost=cost, prompt=prompt, output=output, similarity=similarity)

        return cost

    x0 = rng.uniform(0.0, np.pi / 2, size=2 * p_layers)
    minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})

    return {
        "bitstring": best["bitstring"],
        "cost": best["cost"],
        "prompt": best["prompt"],
        "output": best["output"],
        "similarity": best["similarity"],
        "llm_calls": llm_calls,
        "history": history,
    }


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

    # For every bitstring, the Ising energy (+const) must equal
    # bitstring_cost() computed directly from Q (ignoring the floor
    # penalty, since that's a post-hoc addition on top of the pure
    # QUBO/Ising energy, not part of the Hamiltonian itself).
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

    qaoa_result = run_standard_qaoa(n, Q, p_layers=2, maxiter=150, seed=0)
    match = "MATCHES optimum" if qaoa_result["cost"] == true_opt["cost"] else "does NOT match optimum"
    print(f"run_standard_qaoa: {qaoa_result}  [{match}]")
