"""Baselines package: FP / QW / MADDPG / SAC / MAGC / Mac-dp / MPC /
ReplayOracle (paper Sec. V-B)."""
from .rules import FPController, QWController, MacdpController
from .mpc import MPCController, ReplayOracle
from .marl import MADDPGAgent, SACAgent
from .magc import MAGCAgent


def make_controller(name, cfg, env=None):
    if name == "FP":
        return FPController()
    if name == "QW":
        return QWController()
    if name == "Mac-dp":
        return MacdpController()
    if name == "MPC":
        return MPCController(cfg)
    if name == "Oracle":
        return ReplayOracle(cfg)
    if name == "MADDPG":
        return MADDPGAgent(cfg, env, sac=False)
    if name == "SAC":
        return SACAgent(cfg, env)
    if name == "MAGC":
        return MAGCAgent(cfg, env)
    if name == "MAGC-nomove":
        return MAGCAgent(cfg, env, move_mcs=False)
    raise KeyError(name)


def run_day(env, controller, day, train=False):
    """Run one episode. controllers expose act(env); RL ones also
    push/update when train=True."""
    env.reset(day=day)
    if hasattr(controller, "preday"):
        controller.preday(env, day)
    while True:
        a = controller.act(env, greedy=not train)
        rew = env.step(a)
        if hasattr(controller, "push"):
            controller.push(env, rew)
        if hasattr(controller, "update") and train:
            controller.update()
        if rew["done"]:
            break
    return env.metrics()
