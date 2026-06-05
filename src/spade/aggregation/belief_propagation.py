"""Multi-grid loopy belief propagation over a hierarchical Potts factor graph.

After persistent-peak detection narrows the transform space to K candidate
hypotheses, this stage assigns every source patch to a label in
{h_1, ..., h_K, null} jointly across all scales {3,4,5,6}.

Factors
-------
Unary phi_i(l):
    The log Bayes factor of source patch i under hypothesis h_l (the
    associated transform applied to find the corresponding target patch and
    a Bayes factor computed). Null label has phi_i(null) = 0 by convention.

Spatial smoothness psi_{ij}(l_i, l_j):
    Potts penalty: -beta if labels disagree, 0 otherwise. Encodes "neighbor
    source patches probably belong to the same forgery region or both to
    nothing".

Cross-scale inclusion psi_{ij}^xs(l_i, l_j):
    For a coarse patch i (e.g. 6x6) and a finer patch j (e.g. 3x3) inside
    its footprint: -gamma if labels disagree (and neither is null),
    0 otherwise. Encodes "a 6x6 detection must be consistent with the
    smaller patches it contains".

Inference
---------
Loopy sum-product BP with damping. Stable for moderate (K+1)-state grids
with hundreds of nodes. Outputs:
    - per-node marginal posterior over labels
    - Bethe log-Z = global forensic free energy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

@dataclass
class BPNode:
    """One source patch in the factor graph."""
    node_id: int
    scale: int                  # patch size (3, 4, 5, 6)
    xy: Tuple[float, float]     # source-image coordinates
    log_unary: np.ndarray       # (K+1,) log-potentials under each label


@dataclass
class BPEdge:
    """A pairwise factor between two nodes."""
    a: int
    b: int
    kind: str                   # "spatial" or "inclusion"
    beta: float                 # Potts coupling strength (in nats)


@dataclass
class BPGraph:
    nodes: List[BPNode] = field(default_factory=list)
    edges: List[BPEdge] = field(default_factory=list)
    n_labels: int = 0           # K + 1 (hypotheses + null)

    def add_node(self, node: BPNode) -> None:
        self.nodes.append(node)
        if self.n_labels == 0:
            self.n_labels = len(node.log_unary)

    def add_edge(self, edge: BPEdge) -> None:
        self.edges.append(edge)


# ---------------------------------------------------------------------------
# Loopy BP
# ---------------------------------------------------------------------------

@dataclass
class BPResult:
    marginals: np.ndarray       # (N, n_labels) - posterior per node
    log_z: float                # Bethe approximation to log Z
    converged: bool
    iterations: int


def _potts_log_factor(n_labels: int, beta: float, null_label: int = -1) -> np.ndarray:
    """Log of the Potts factor matrix.

    factor[a, b] = 0   if a == b
                 = -beta otherwise
    The null label is treated like any other label - identical pairs have
    factor 1, mismatched pairs are penalized.
    """
    f = np.full((n_labels, n_labels), -beta, dtype=np.float64)
    np.fill_diagonal(f, 0.0)
    return f


def loopy_bp(
    graph: BPGraph,
    max_iter: int = 50,
    damping: float = 0.5,
    tol: float = 1e-4,
) -> BPResult:
    """Damped log-domain sum-product BP. Returns marginals + Bethe log Z."""
    n_labels = graph.n_labels
    n_nodes = len(graph.nodes)
    if n_nodes == 0 or n_labels == 0:
        return BPResult(np.zeros((0, n_labels)), 0.0, True, 0)

    # Adjacency: for every edge, store a per-direction message a->b and b->a.
    # Messages are log-domain probability vectors of length n_labels.
    n_edges = len(graph.edges)
    msg_ab = np.zeros((n_edges, n_labels), dtype=np.float64)  # a -> b
    msg_ba = np.zeros((n_edges, n_labels), dtype=np.float64)  # b -> a

    node_edges: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(n_nodes)}
    for k, e in enumerate(graph.edges):
        node_edges[e.a].append((k, "ab"))
        node_edges[e.b].append((k, "ba"))

    # Precompute log factor per edge
    log_factors: List[np.ndarray] = []
    for e in graph.edges:
        log_factors.append(_potts_log_factor(n_labels, e.beta))

    log_unary = np.stack([n.log_unary for n in graph.nodes])  # (N, n_labels)

    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        max_delta = 0.0
        new_msg_ab = msg_ab.copy()
        new_msg_ba = msg_ba.copy()

        for k, e in enumerate(graph.edges):
            log_f = log_factors[k]

            # Outgoing from a -> b: combine a's unary and all incoming msgs to a (except from b)
            incoming_a = log_unary[e.a].copy()
            for k2, dir_ in node_edges[e.a]:
                if k2 == k:
                    continue
                incoming_a += msg_ab[k2] if dir_ == "ba" else msg_ba[k2]
            # m_{a->b}(x_b) = logsumexp_{x_a} [ log_f(x_a, x_b) + incoming_a(x_a) ]
            log_combined = log_f + incoming_a[:, None]   # shape (n_labels, n_labels)
            outgoing = _logsumexp(log_combined, axis=0)
            outgoing -= outgoing.max()
            new_msg_ab[k] = (1 - damping) * outgoing + damping * msg_ab[k]

            # Outgoing from b -> a (transpose factor)
            incoming_b = log_unary[e.b].copy()
            for k2, dir_ in node_edges[e.b]:
                if k2 == k:
                    continue
                incoming_b += msg_ab[k2] if dir_ == "ba" else msg_ba[k2]
            log_combined = log_f.T + incoming_b[:, None]
            outgoing = _logsumexp(log_combined, axis=0)
            outgoing -= outgoing.max()
            new_msg_ba[k] = (1 - damping) * outgoing + damping * msg_ba[k]

            max_delta = max(
                max_delta,
                float(np.max(np.abs(new_msg_ab[k] - msg_ab[k]))),
                float(np.max(np.abs(new_msg_ba[k] - msg_ba[k]))),
            )

        msg_ab, msg_ba = new_msg_ab, new_msg_ba
        if max_delta < tol:
            converged = True
            break

    # Compute marginals
    log_marginals = log_unary.copy()
    for k, e in enumerate(graph.edges):
        log_marginals[e.b] += msg_ab[k]
        log_marginals[e.a] += msg_ba[k]
    log_marginals -= log_marginals.max(axis=1, keepdims=True)
    marginals = np.exp(log_marginals)
    marginals /= marginals.sum(axis=1, keepdims=True)

    # Bethe approximation to log Z
    # log Z_Bethe = sum_i log Z_i - sum_{ij} log Z_{ij}
    # where Z_i is the node partition (incoming msgs * unary)
    # and  Z_{ij} is the edge partition (msgs from both sides * factor)
    log_z = 0.0
    for i, _ in enumerate(graph.nodes):
        node_log = log_unary[i].copy()
        for k, dir_ in node_edges[i]:
            node_log += msg_ab[k] if dir_ == "ba" else msg_ba[k]
        log_z += float(_logsumexp(node_log))

    for k, e in enumerate(graph.edges):
        # Pair (a, b): incoming-to-a (excl b) + incoming-to-b (excl a) + factor
        in_a = log_unary[e.a].copy()
        for k2, dir_ in node_edges[e.a]:
            if k2 == k: continue
            in_a += msg_ab[k2] if dir_ == "ba" else msg_ba[k2]
        in_b = log_unary[e.b].copy()
        for k2, dir_ in node_edges[e.b]:
            if k2 == k: continue
            in_b += msg_ab[k2] if dir_ == "ba" else msg_ba[k2]
        edge_log = log_factors[k] + in_a[:, None] + in_b[None, :]
        log_z -= float(_logsumexp(edge_log.ravel()))

    return BPResult(
        marginals=marginals,
        log_z=float(log_z),
        converged=converged,
        iterations=iterations,
    )


def _logsumexp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


# ---------------------------------------------------------------------------
# Convenience: build the standard multi-scale graph
# ---------------------------------------------------------------------------

def build_multiscale_graph(
    unaries_per_scale: Dict[int, np.ndarray],
    coords_per_scale: Dict[int, np.ndarray],
    spatial_radius: float = 4.0,
    spatial_beta: float = 1.0,
    inclusion_beta: float = 2.0,
) -> BPGraph:
    """Construct a hierarchical factor graph from per-scale unaries.

    Args:
        unaries_per_scale: {scale: (N_scale, n_labels) log-unary potentials}
        coords_per_scale:  {scale: (N_scale, 2) source xy per node}
        spatial_radius:    nodes within this many pixels at the same scale get a smoothness edge
        spatial_beta:      Potts coupling for spatial smoothness
        inclusion_beta:    Potts coupling for cross-scale inclusion (typically > spatial)
    """
    graph = BPGraph()
    if not unaries_per_scale:
        return graph
    n_labels = next(iter(unaries_per_scale.values())).shape[1]
    graph.n_labels = n_labels

    # Add nodes
    node_index_per_scale: Dict[int, List[int]] = {}
    for scale, unary in unaries_per_scale.items():
        coords = coords_per_scale[scale]
        ids: List[int] = []
        for k in range(len(unary)):
            node = BPNode(
                node_id=len(graph.nodes),
                scale=scale,
                xy=(float(coords[k, 0]), float(coords[k, 1])),
                log_unary=np.asarray(unary[k], dtype=np.float64),
            )
            ids.append(node.node_id)
            graph.add_node(node)
        node_index_per_scale[scale] = ids

    # Spatial smoothness edges (within each scale)
    for scale, ids in node_index_per_scale.items():
        coords = coords_per_scale[scale]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                if dx * dx + dy * dy <= spatial_radius * spatial_radius:
                    graph.add_edge(BPEdge(
                        a=ids[i], b=ids[j], kind="spatial", beta=spatial_beta,
                    ))

    # Cross-scale inclusion edges: a coarse patch at (x, y) "contains" all
    # finer patches whose centers fall within its footprint.
    scales = sorted(node_index_per_scale.keys())
    for s_coarse in scales:
        for s_fine in scales:
            if s_fine >= s_coarse:
                continue
            half = s_coarse / 2.0
            for ic, c_id in enumerate(node_index_per_scale[s_coarse]):
                cx, cy = coords_per_scale[s_coarse][ic]
                for jf, f_id in enumerate(node_index_per_scale[s_fine]):
                    fx, fy = coords_per_scale[s_fine][jf]
                    if abs(fx - cx) <= half and abs(fy - cy) <= half:
                        graph.add_edge(BPEdge(
                            a=c_id, b=f_id, kind="inclusion", beta=inclusion_beta,
                        ))

    return graph
