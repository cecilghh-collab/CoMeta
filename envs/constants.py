"""Auto-migrated from cometa/env.py -- see module docstrings."""

# ET states
ST_IDLE, ST_ORDER, ST_CHARGE = 0, 1, 2
# ET trip roles
R_NONE, R_PICKUP, R_DEST, R_TO_CS, R_CRUISE = 0, 1, 2, 3, 4
# MCS states
M_IDLE, M_MOVE, M_SERVE, M_SELF, M_WAIT = \
    "idle", "move", "serve", "self", "wait_charge"

EPS = 1e-9


def _is_mcs_entry(x):
    return x < 0


def _mcs_entry(mid):
    return -1 - int(mid)
