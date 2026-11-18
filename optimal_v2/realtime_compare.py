"""Run the saved optimal PID, LQR and PPO on three live cart-pole simulations.

From the repository root:
    python optimal_v2/realtime_compare.py
    python optimal_v2/realtime_compare.py --theta-deg -25 --x 0.3

Space pauses/resumes, R restarts from the same initial state, Q/Esc closes.
Each controller advances its own CartPoleSim on the same 0.02 s clock.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cartpole_common as C
import cartpole_lqr as L
import cartpole_pid as P
import cartpole_ppo as PP
import evaluate_metrics as E


COLORS = {"PID": "#2a78d6", "LQR": "#eb6834", "PPO": "#1baf7a"}
MODEL_DIR = HERE / "models"


def load_optimal_controllers(model_dir=MODEL_DIR):
    """Load the same saved models used by optimal_compare.py; fail if any is missing."""
    model_dir = Path(model_dir)
    paths = [model_dir / "pid_gains_opt.json", model_dir / "lqr_weights_opt.json",
             model_dir / "ppo_opt.zip"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("최적 모델 파일이 없습니다: " + ", ".join(missing))

    with paths[0].open(encoding="utf-8") as f:
        pid = json.load(f)
    with paths[1].open(encoding="utf-8") as f:
        lqr = json.load(f)

    # The saved K is the result of optimal_compare's LQR design and F = -K s.
    # Reuse it directly so this viewer does not re-optimize or change the gains.
    controllers = {
        "PID": P.PIDController(pid["PID_THETA"], pid["PID_X"]),
        "LQR": L.LQRController(lqr["K"]),
    }
    try:
        controllers["PPO"] = PP.PPOController(str(model_dir / "ppo_opt"))
    except ImportError as exc:
        raise RuntimeError("PPO 실행에 stable-baselines3와 PyTorch가 필요합니다. "
                           "README의 설치 명령을 먼저 실행하세요.") from exc
    return controllers


class LiveSimulation:
    """One RK4 step per controller per tick, with C.simulate's noise/failure rules."""

    def __init__(self, controllers, s0):
        self.controllers = controllers
        self.s0 = np.asarray(s0, dtype=np.float64).copy()
        self.n_steps = int(round(C.SIM_TIME / C.DT))
        self.k = 0
        self.states = {name: [self.s0.copy()] for name in controllers}
        self.forces = {name: [] for name in controllers}
        self.failed = {name: False for name in controllers}
        self.sims = {name: C.CartPoleSim() for name in controllers}
        # Independent streams with the same seed, as in optimal_compare.py.
        self.rng = {name: np.random.default_rng(C.SEED) for name in controllers}
        self.noise = np.asarray(C.NOISE_STD, dtype=np.float64)
        for controller in controllers.values():
            controller.reset()

    @property
    def done(self):
        return self.k >= self.n_steps or all(self.failed.values())

    def step(self):
        if self.done:
            return
        t = self.k * C.DT
        for name, controller in self.controllers.items():
            if self.failed[name]:
                continue
            s = self.states[name][-1]
            meas = s + self.rng[name].normal(0.0, 1.0, 4) * self.noise
            F = float(np.clip(controller.act(meas), -C.F_MAX, C.F_MAX))
            Fd = C.disturbance_force(t)
            next_s = self.sims[name].step(s, F + Fd)
            self.states[name].append(next_s.copy())
            self.forces[name].append(F)
            self.failed[name] = bool(C.is_failed(next_s))
        self.k += 1

    def log(self, name):
        s = np.asarray(self.states[name])
        return dict(t=np.arange(len(s)) * C.DT, s=s,
                    u=np.asarray(self.forces[name]), failed=self.failed[name])


def show_live(controllers, s0, speed=1.0, fps=30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    run = LiveSimulation(controllers, s0)
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(4, 3, height_ratios=[2.2, 0.75, 0.75, 0.75],
                           hspace=0.45, top=0.91, bottom=0.08)
    names = list(controllers)
    cart_w, cart_h = 0.30, 0.15
    pivot_y = cart_h / 2
    pole_length = C.PHYS["l"]
    panels = {}

    for i, name in enumerate(names):
        ax = fig.add_subplot(grid[0, i])
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-0.45, 0.65)
        ax.set_aspect("equal")
        ax.axhline(-cart_h / 2, color="#777777", linewidth=2)
        ax.axvline(0, color="#cccccc", linestyle=":")
        ax.set_title(name + " (optimized)", color=COLORS[name], fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel("Cart position [m]")
        cart = Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h,
                         facecolor="#596273", edgecolor="black")
        ax.add_patch(cart)
        pole, = ax.plot([], [], linewidth=5, color=COLORS[name], solid_capstyle="round")
        bob, = ax.plot([], [], "o", markersize=10, color=COLORS[name])
        force, = ax.plot([], [], linewidth=3, color="#c84141")
        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
        status = ax.text(0.98, 0.98, "", transform=ax.transAxes, ha="right", va="top",
                         fontweight="bold", fontsize=10)
        panels[name] = (ax, cart, pole, bob, force, info, status)

    trace_axes = [fig.add_subplot(grid[row, :]) for row in (1, 2, 3)]
    limits = [(-40, 40), (-C.X_MAX - 0.2, C.X_MAX + 0.2),
              (-C.F_MAX - 2, C.F_MAX + 2)]
    labels = ["Angle [deg]", "Cart x [m]", "Force [N]"]
    traces = []
    for ax, (lo, hi), label in zip(trace_axes, limits, labels):
        ax.set_xlim(0, C.SIM_TIME)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        traces.append({name: ax.plot([], [], color=COLORS[name], label=name)[0]
                       for name in names})
    trace_axes[0].axhspan(-E.CRIT["theta_deg"], E.CRIT["theta_deg"], color="#dddddd", alpha=0.4)
    trace_axes[1].axhspan(-E.CRIT["x"], E.CRIT["x"], color="#dddddd", alpha=0.4)
    trace_axes[0].legend(ncol=3, loc="upper right")
    trace_axes[2].set_xlabel("Simulation time [s]")
    fig.text(0.5, 0.01, "Space: pause/resume    R: restart    Q / Esc: close",
             ha="center", fontsize=10)

    paused = [False]
    last_wall = [None]
    backlog = [0.0]
    animation = [None]
    reported = [False]

    def render():
        t = run.k * C.DT
        phase = "PAUSED" if paused[0] else "FINISHED" if run.done else "LIVE"
        fig.suptitle(f"Optimized PID / LQR / PPO  |  {phase}  {t:.2f}/{C.SIM_TIME:g} s"
                     f"  |  s0=[{s0[0]:g}, {s0[1]:g}, {np.degrees(s0[2]):g}°, {s0[3]:g}]"
                     f"  |  {speed:g}× speed", fontsize=15)
        for name in names:
            ax, cart, pole, bob, force, info, status = panels[name]
            s = run.states[name][-1]
            x, _, theta, _ = s
            F = run.forces[name][-1] if run.forces[name] else 0.0
            tip = (x + pole_length * np.sin(theta), pivot_y + pole_length * np.cos(theta))
            cart.set_xy((x - cart_w / 2, -cart_h / 2))
            pole.set_data([x, tip[0]], [pivot_y, tip[1]])
            bob.set_data([tip[0]], [tip[1]])
            force.set_data([x, x + 0.28 * F / C.F_MAX], [0, 0])
            info.set_text(f"angle  {np.degrees(theta):+6.2f}°\nx      {x:+6.3f} m\nF      {F:+6.2f} N")
            if run.failed[name]:
                result = "FAIL: rail"
            elif run.done:
                result = "SUCCESS" if E.judge(run.log(name))[0] else "FAIL: unsettled"
            else:
                result = "RUNNING"
            status.set_text(result)
            status.set_color("#16754a" if result == "SUCCESS" else
                             "#c84141" if result.startswith("FAIL") else "#555555")
            # Keep the cart on screen if an initial position or a failure is far from zero.
            center = np.sign(x) * max(abs(x) - 0.95, 0.0)
            ax.set_xlim(center - 1.35, center + 1.35)

            log = run.log(name)
            traces[0][name].set_data(log["t"], np.degrees(log["s"][:, 2]))
            traces[1][name].set_data(log["t"], log["s"][:, 0])
            traces[2][name].set_data(log["t"][:-1], log["u"])

        # A fall past 40° remains visible in the angle trace.
        peak = max(abs(np.degrees(np.asarray(run.states[name])[:, 2])).max() for name in names)
        span = max(40.0, float(peak) * 1.1)
        trace_axes[0].set_ylim(-span, span)

    def report():
        if reported[0]:
            return
        reported[0] = True
        print(f"\nFinished at t={run.k * C.DT:.2f} s (same initial state, dt and force limit)")
        for name in names:
            metrics = E.compute_pdf_metrics(run.log(name), np.degrees(s0[2]))
            settling = metrics["settling_time_s"]
            print(f"{name:3s}: {'SUCCESS' if metrics['success'] else 'FAIL':7s}  "
                  f"settling={settling:.2f} s  effort={metrics['control_effort']:.1f} N²·s  "
                  f"reason={','.join(metrics['why']) or '-'}")

    def update(_frame):
        now = time.monotonic()
        if last_wall[0] is not None and not paused[0] and not run.done:
            backlog[0] += (now - last_wall[0]) * speed
            steps = min(int((backlog[0] + 1e-12) / C.DT), run.n_steps - run.k)
            for _ in range(steps):
                run.step()
            backlog[0] -= steps * C.DT
        last_wall[0] = now
        render()
        if run.done:
            report()
            animation[0].event_source.stop()

    def on_key(event):
        nonlocal run
        if event.key in ("q", "escape"):
            plt.close(fig)
        elif event.key == " ":
            paused[0] = not paused[0]
            last_wall[0] = time.monotonic()
            render()
            fig.canvas.draw_idle()
        elif event.key == "r":
            run = LiveSimulation(controllers, s0)
            paused[0] = False
            last_wall[0] = None
            backlog[0] = 0.0
            reported[0] = False
            render()
            animation[0].event_source.start()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    animation[0] = FuncAnimation(fig, update, interval=1000 / fps,
                                 cache_frame_data=False, blit=False)
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live comparison of three saved optimal cart-pole controllers")
    parser.add_argument("--theta-deg", type=float, default=30.0, help="initial pole angle in degrees (default: 30)")
    parser.add_argument("--x", type=float, default=0.0, help="initial cart position [m]")
    parser.add_argument("--x-dot", type=float, default=0.0, help="initial cart velocity [m/s]")
    parser.add_argument("--theta-dot", type=float, default=0.0, help="initial pole angular velocity [rad/s]")
    parser.add_argument("--speed", type=float, default=1.0, help="playback rate relative to real time")
    parser.add_argument("--fps", type=int, default=30, help="maximum plot refreshes per second")
    args = parser.parse_args(argv)
    if not np.isfinite([args.theta_deg, args.x, args.x_dot, args.theta_dot, args.speed]).all():
        parser.error("초기조건과 재생속도는 유한한 숫자여야 합니다.")
    if args.speed <= 0 or args.fps <= 0:
        parser.error("--speed와 --fps는 0보다 커야 합니다.")
    s0 = C.make_s0(dict(theta_deg=args.theta_deg, x=args.x, x_dot=args.x_dot,
                        theta_dot=args.theta_dot))
    controllers = load_optimal_controllers()
    show_live(controllers, s0, speed=args.speed, fps=args.fps)


if __name__ == "__main__":
    main()
