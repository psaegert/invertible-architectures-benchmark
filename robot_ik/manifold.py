# SPDX-License-Identifier: MIT
"""Exact posterior sampling on one-dimensional solution sets (fibers).

Setting: x ~ prior on a box of joint angles, y = f(x) deterministic, and the
fiber F(theta) = 0 of a target y* is a curve: m = d - 1 independent
constraints in d joints. By the coarea formula the posterior on the fiber is

    p(theta | y*) d(length)  ∝  prior(theta) / sqrt(det(J J^T)) d(length),

J the m x d Jacobian of F in an orthonormal chart of the observation space.

Cofactor field. c_k(theta) = (-1)^k det(J with column k removed) is tangent
to the fiber, its orientation is consistent along the curve (det [J; c^T] =
|c|^2 > 0), and |c| = sqrt(det(J J^T)) (Cauchy-Binet). Along the flow
d(theta)/d(tau) = c the posterior under a uniform prior is therefore uniform
in tau: the mass of a piece of fiber is its tau-length. A non-uniform prior is
handled by rejection with prior / max(prior).

Method, batched over many fibers in float64:
  discovery  Levenberg-Marquardt from uniform seeds on the whole torus, on a
             residual that is valid everywhere (`discovery_residual`).
  tracing    every fiber component of a regular value is a closed loop on the
             torus. From one discovered point per component, integrate the unit
             tangent c/|c| in arclength with RK4, correct back onto the fiber
             with minimum-norm Newton steps, and accumulate tau with Simpson's
             rule on 1/|c| (midpoint by Hermite interpolation). Steps that do
             not converge or turn the tangent too sharply are halved. The loop
             is closed when the trace returns to its start.
  sampling   tau uniform over all loops of the target, Hermite interpolation in
             tau inside the segment (derivatives c at its ends), a Newton
             correction onto the fiber, then rejection by the prior weight
             (for a box prior: inside the box or not).
Seeds are never taken from the configuration that generated y*, so that
configuration can be used to check that no component was missed.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

TWO_PI = 2 * math.pi


def wrap(d):
    """Angle differences to (-pi, pi]."""
    return torch.remainder(d + math.pi, TWO_PI) - math.pi


def into_box(theta, lower):
    """Representatives of angles theta in [lower, lower + 2 pi)."""
    return lower + torch.remainder(theta - lower, TWO_PI)


def cofactor(J):
    """(B, m, m + 1) -> (B, m + 1), tangent to the fiber, |c| = sqrt(det J J^T)."""
    d = J.shape[2]
    cols = [(-1) ** k * torch.linalg.det(J[:, :, [j for j in range(d) if j != k]]) for k in range(d)]
    return torch.stack(cols, dim=1)


class FiberProblem:
    """Interface for the fibers of a batch of targets (indexed by `idx`).

    residual(theta, idx)            -> r (B, m), J (B, m, d): the constraint in an
                                       orthonormal chart; used for tangents,
                                       correction and the coarea weight.
    discovery_residual(theta, idx)  -> r (B, k), J (B, k, d): globally valid
                                       residual for finding fibers from far away.
    weight(theta, idx)              -> (B,) prior weight in [0, 1] (box prior:
                                       0 or 1); theta may be unwrapped.
    dim, n_targets, device
    periodic: True for joint angles (fibers are loops on the torus); False for
    Euclidean coordinates (closed fibers only).
    """
    periodic = True

    def wrap(self, d):
        return wrap(d) if self.periodic else d


def project(problem, theta, idx, iters=4):
    """Minimum-norm Newton steps onto the fiber; returns theta and |r|."""
    for _ in range(iters):
        r, J = problem.residual(theta, idx)
        A = J @ J.transpose(1, 2)
        A = A + 1e-300 * torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
        theta = theta - (J.transpose(1, 2) @ torch.linalg.solve(A, r.unsqueeze(-1))).squeeze(-1)
    r, _ = problem.residual(theta, idx)
    return theta, r.norm(dim=1)


def levenberg_marquardt(problem, theta, idx, iters=80, lam=1e-2):
    """Damped Gauss-Newton on the discovery residual, then Newton on the chart
    residual. Returns theta and the final chart residual norm."""
    lam = torch.full((len(theta),), lam, dtype=theta.dtype, device=theta.device)
    r, J = problem.discovery_residual(theta, idx)
    cost = (r * r).sum(1)
    for _ in range(iters):
        A = J @ J.transpose(1, 2) + lam[:, None, None] * torch.eye(J.shape[1], dtype=J.dtype, device=J.device)
        cand = theta - (J.transpose(1, 2) @ torch.linalg.solve(A, r.unsqueeze(-1))).squeeze(-1)
        r2, J2 = problem.discovery_residual(cand, idx)
        cost2 = (r2 * r2).sum(1)
        ok = cost2 < cost
        theta = torch.where(ok[:, None], cand, theta)
        r = torch.where(ok[:, None], r2, r)
        J = torch.where(ok[:, None, None], J2, J)
        cost = torch.where(ok, cost2, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-12, 1e8)
    near = cost < 1e-6
    theta_n, res = project(problem, theta[near], idx[near])
    out = theta.clone()
    out[near] = theta_n
    full = torch.full((len(theta),), float('inf'), dtype=theta.dtype, device=theta.device)
    full[near] = res
    return out, full


def unit_tangent(problem, theta, idx):
    _, J = problem.residual(theta, idx)
    c = cofactor(J)
    n = c.norm(dim=1, keepdim=True)
    return c / n, n.squeeze(1)


@dataclass
class Loop:
    """One traced fiber component of target `idx`: vertices (n + 1, d), unwrapped
    and continuous, with the last vertex equal to the first modulo 2 pi; tau
    (n + 1,) cumulative."""
    idx: int
    vertices: torch.Tensor
    tau: torch.Tensor
    closed: bool
    closure_error: float
    n_rejected: int = 0
    info: dict = field(default_factory=dict)

    @property
    def mass(self):
        return float(self.tau[-1])


def _simpson_dtau(problem, a, b, ta, tb, na, nb, h, idx):
    """tau increment over a step of arclength h from a to b (on the fiber),
    with unit tangents ta, tb and |c| values na, nb at the ends."""
    mid = 0.5 * (a + b) + (h[:, None] / 8) * (ta - tb)
    _, nm = unit_tangent(problem, mid, idx)
    return h / 6 * (1 / na + 4 / nm + 1 / nb)


def trace(problem, start, idx, h_max=0.02, h_min=1e-7, max_turn=0.15, tol=1e-10, max_steps=200_000):
    """Trace the loops through the points `start` (B, d) on the fibers of
    targets idx (B,). Returns a list of Loop, one per start."""
    B, d = start.shape
    dev, dt = start.device, start.dtype
    theta = start.clone()
    t_cur, n_cur = unit_tangent(problem, theta, idx)
    t_start = t_cur.clone()
    h = torch.full((B,), h_max, dtype=dt, device=dev)
    travelled = torch.zeros(B, dtype=dt, device=dev)
    tau = torch.zeros(B, dtype=dt, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    closed = torch.zeros(B, dtype=torch.bool, device=dev)
    closure = torch.full((B,), float('nan'), dtype=dt, device=dev)
    rejected = torch.zeros(B, dtype=torch.long, device=dev)
    verts, taus, owners = [start.clone()], [tau.clone()], [torch.arange(B, device=dev)]

    for _ in range(max_steps):
        a = active.nonzero().squeeze(1)
        if len(a) == 0:
            break
        th, ia, hh = theta[a], idx[a], h[a]
        t0, n0 = t_cur[a], n_cur[a]
        # Close the loop when the start is within one step ahead.
        to_start = problem.wrap(start[a] - th)
        dist = to_start.norm(dim=1)
        ahead = (to_start * t0).sum(1) > 0.9 * dist
        closing = (travelled[a] > 4 * h_max) & (dist <= hh) & ahead
        hh = torch.where(closing, dist, hh)
        # RK4 predictor in arclength, Newton corrector.
        k1 = t0
        k2, _ = unit_tangent(problem, th + 0.5 * hh[:, None] * k1, ia)
        k3, _ = unit_tangent(problem, th + 0.5 * hh[:, None] * k2, ia)
        k4, _ = unit_tangent(problem, th + hh[:, None] * k3, ia)
        pred = th + (hh[:, None] / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        new, res = project(problem, pred, ia, iters=3)
        t1, n1 = unit_tangent(problem, new, ia)
        turn = torch.arccos(((t0 * t1).sum(1)).clamp(-1, 1))
        ok = (res < tol) & (turn < max_turn) & torch.isfinite(new).all(1)
        # Rejected steps: halve and retry (a closing step that fails also retries).
        bad = a[~ok]
        h[bad] = h[bad] / 2
        rejected[bad] += 1
        dead = bad[h[bad] < h_min]
        active[dead] = False
        # Accepted steps.
        g = ok.nonzero().squeeze(1)
        if len(g):
            ag, hg = a[g], hh[g]
            cl = closing[g]
            end = torch.where(cl[:, None], th[g] + to_start[g], new[g])   # land exactly on the start
            # closure error: offset of the predicted end from the start normal to the
            # curve (the tangential gap is chord vs arc, O(h^3), and not an error)
            gap = problem.wrap(new[g][cl] - start[ag[cl]])
            tan = t_start[ag[cl]]
            closure[ag[cl]] = (gap - (gap * tan).sum(1, keepdim=True) * tan).norm(dim=1)
            t_end, n_end = unit_tangent(problem, end, ia[g])
            dtau = _simpson_dtau(problem, th[g], end, t0[g], t_end, n0[g], n_end, hg, ia[g])
            tau[ag] = tau[ag] + dtau
            theta[ag], t_cur[ag], n_cur[ag] = end, t_end, n_end
            travelled[ag] = travelled[ag] + hg
            grow = (turn[g] < max_turn / 4) & ~cl
            h[ag] = torch.where(grow, (hg * 1.5).clamp(max=h_max), torch.minimum(h[ag], torch.full_like(hg, h_max)))
            closed[ag[cl]] = True
            active[ag[cl]] = False
            verts.append(end)
            taus.append(tau[ag].clone())
            owners.append(ag)

    V, T, O = torch.cat(verts), torch.cat(taus), torch.cat(owners)
    loops = []
    for b in range(B):
        sel = (O == b).nonzero().squeeze(1)
        loops.append(Loop(int(idx[b]), V[sel], T[sel], bool(closed[b]), float(closure[b]),
                          int(rejected[b])))
    return loops


def on_loops(problem, points, loops, tol):
    """(k,) bool: which points (k, d) lie within tol (wrapped) of a vertex of any loop."""
    if not loops or len(points) == 0:
        return torch.zeros(len(points), dtype=torch.bool, device=points.device)
    V = torch.cat([l.vertices for l in loops])
    out = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for s in range(0, len(points), 512):
        diff = problem.wrap(points[s:s + 512, None, :] - V[None])
        out[s:s + 512] = diff.norm(dim=2).min(dim=1).values <= tol
    return out


def discover_and_trace(problem, n_seeds, gen, h_max=0.02, chunk=200_000, verbose=False, seed_scale=math.pi):
    """All loops of every target. Returns (loops per target, diagnostics).
    Seeds are uniform on [-seed_scale, seed_scale]^d (the whole torus by default)."""
    dev, dt = problem.device, torch.float64
    M = problem.n_targets
    cand_theta, cand_idx = [], []
    for s in range(0, M * n_seeds, chunk):
        n = min(chunk, M * n_seeds - s)
        idx = torch.arange(s, s + n, device=dev) // n_seeds
        seeds = (torch.rand(n, problem.dim, generator=gen, dtype=dt, device='cpu').to(dev) * 2 - 1) * seed_scale
        th, res = levenberg_marquardt(problem, seeds, idx)
        keep = res < 1e-10
        cand_theta.append(th[keep])
        cand_idx.append(idx[keep])
    cand_theta, cand_idx = torch.cat(cand_theta), torch.cat(cand_idx)   # sorted by target
    n_found = torch.bincount(cand_idx, minlength=M)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(n_found, 0)]).tolist()
    loops = [[] for _ in range(M)]
    covered = torch.zeros(len(cand_idx), dtype=torch.bool, device=dev)
    pos = torch.arange(len(cand_idx), device=dev)
    rounds = 0
    while not covered.all():
        rounds += 1
        # first uncovered candidate of every target that still has one
        key = torch.where(covered, torch.full_like(pos, len(pos)), pos)
        first = torch.full((M,), len(pos), dtype=torch.long, device=dev).scatter_reduce(0, cand_idx, key, 'amin')
        pick = first[first < len(pos)]
        new = trace(problem, cand_theta[pick], cand_idx[pick], h_max=h_max)
        covered[pick] = True
        for lp in new:
            loops[lp.idx].append(lp)
            lo, hi = offsets[lp.idx], offsets[lp.idx + 1]
            open_ = (~covered[lo:hi]).nonzero().squeeze(1) + lo
            if len(open_):
                covered[open_] = on_loops(problem, cand_theta[open_], [lp], tol=h_max)
        if verbose:
            print(f'  round {rounds}: traced {len(pick)} loops, {int((~covered).sum())} candidates left', flush=True)
    diag = {
        'n_candidates': n_found.cpu().numpy(),
        'n_loops': np.array([len(l) for l in loops]),
        'all_closed': np.array([all(lp.closed for lp in l) for l in loops]),
        'max_closure_error': np.array([max([lp.closure_error for lp in l], default=np.nan) for l in loops]),
        'loop_mass': np.array([sum(lp.mass for lp in l) for l in loops]),
    }
    return loops, diag


def hermite(V0, V1, c0, c1, dtau, u):
    """Cubic Hermite in tau on a segment with end derivatives c (d theta / d tau)."""
    u = u[:, None]
    h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
    h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * V0 + h10 * dtau[:, None] * c0 + h01 * V1 + h11 * dtau[:, None] * c1


def sample_loops(problem, loops, i, n, gen, max_rounds=200):
    """n draws from the posterior of target i, given all its loops.
    Returns (n, d) theta (unwrapped) and the accepted fraction (in-box share
    of the tau-mass for a box prior)."""
    dev = problem.device
    V = [l.vertices for l in loops]
    T = [l.tau for l in loops]
    lengths = torch.tensor([float(t[-1]) for t in T], dtype=torch.float64)
    total = float(lengths.sum())
    out, n_prop = [], 0
    for _ in range(max_rounds):
        k = max(2 * n, 256)
        u = torch.rand(k, generator=gen, dtype=torch.float64) * total
        which = torch.searchsorted(torch.cumsum(lengths, 0), u, right=True).clamp(max=len(loops) - 1)
        offs = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, 0)[:-1]])
        u = u - offs[which]
        pts = []
        for j in range(len(loops)):
            sel = (which == j).nonzero().squeeze(1)
            if len(sel) == 0:
                continue
            uu = u[sel].to(dev)
            seg = torch.searchsorted(T[j], uu, right=True).clamp(1, len(T[j]) - 1) - 1
            V0, V1 = V[j][seg], V[j][seg + 1]
            dtau = T[j][seg + 1] - T[j][seg]
            ii = torch.full((len(seg),), i, device=dev)
            _, J0 = problem.residual(V0, ii)
            _, J1 = problem.residual(V1, ii)
            frac = (uu - T[j][seg]) / dtau
            pts.append(hermite(V0, V1, cofactor(J0), cofactor(J1), dtau, frac))
        pts = torch.cat(pts)
        ii = torch.full((len(pts),), i, device=dev)
        pts, _ = project(problem, pts, ii, iters=2)
        w = problem.weight(pts, ii)
        acc = torch.rand(len(pts), generator=gen, dtype=torch.float64).to(dev) < w
        out.append(pts[acc])
        n_prop += len(pts)
        if sum(len(o) for o in out) >= n:
            break
    got = torch.cat(out)
    # The accepted points are grouped by loop; keep a uniformly random subset,
    # never the first n (that would drop the later loops).
    keep = torch.randperm(len(got), generator=gen)[:n].to(dev)
    return got[keep], len(got) / max(n_prop, 1)
