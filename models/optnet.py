"""Differentiable feasibility projection (Eq. 30).

The paper uses an OptNet QP layer; with box + distance constraints the
closest-point projection is the analytic clamp implemented in
``cometa.envs.market`` (``_set_prices`` / ``_project_zone``), and the
``--clip-actions`` ablation replaces it by hard clipping with a detached
gradient (Table II, w/o OptNet).
"""


def project_price(p, lo, hi):
    """Elementwise box projection (the analytic QP solution)."""
    return min(max(p, lo), hi)
