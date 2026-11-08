"""
Cart-Pole 역진자 시뮬레이터 — PID / LQR / PPO 시각화 & 비교
================================================================
cartpole_pid.py, cartpole_lqr.py, cartpole_ppo.py 에서 제어기를 불러와
cartpole_common.py 의 같은 시뮬레이터·같은 초기조건으로 실행하고 보여줍니다.

실행:
  python simulator.py                      # 애니메이션 (아래 INIT 초기조건)
  python simulator.py animate theta_deg=20 # 초기조건을 명령줄에서 바로 바꾸기 (여러 개 가능: x=0.5 theta_dot=0.2)
  python simulator.py compare              # 모든 실험 세트 × 모든 제어기 → results/compare/

  * PPO는 먼저 'python cartpole_ppo.py train' 으로 모델을 만들어야 합니다 (없으면 자동으로 건너뜀).
"""

# ================================================================
# 0. 시뮬레이터 설정 — 여기만 바꾸면 됩니다
# ================================================================
MODE = "animate"                # "animate": 한 초기조건에서 제어기들을 애니메이션으로 비교
                                # "compare": 모든 실험 세트를 모든 제어기로 실행 (CSV + 비교 그래프)

CONTROLLERS = ["PID", "LQR", "PPO"]   # 보고 싶은 제어기 (1개만 써도 됨)

# 애니메이션 초기조건: cartpole_common.BASE_S0 에서 바꿀 항목만 적습니다 (각도는 degree)
#   {}                     → Baseline [0, 0, 10°, 0]
#   dict(theta_deg=20)     → 실험 A의 20° 조건
#   dict(x=0.5)            → 실험 B의 +0.5 m 조건
INIT = dict()

PLAYBACK_SPEED = 1.0            # 1.0 = 실시간, 0.5 = 느리게
DISPLAY_FPS = 25                # 화면 갱신 속도 (느린 컴퓨터는 낮추세요)
VIEW_X_RANGE = 1.2              # 애니메이션 가로 표시 범위 ±[m] (카트가 화면 밖으로 나가면 키우세요)
SAVE_ANIMATION = None           # 예: "sim.gif" (pillow) 또는 "sim.mp4" (ffmpeg 필요). None이면 저장 안 함
SHOW = True                     # 창으로 보여주기

OVERLAY_EXPERIMENTS = ["Baseline"]   # compare 모드에서 세 제어기를 한 그래프에 겹쳐 그릴 실험

# ================================================================
# 이하 구현부
# ================================================================
import sys
import numpy as np
import cartpole_common as C

COLORS = {"PID": "tab:blue", "LQR": "tab:orange", "PPO": "tab:green"}


def load_controllers(names):
    """각 제어기 파일의 build_controller()로 제어기 생성 (각 파일의 설정이 그대로 반영됨)"""
    ctrls = []
    for n in names:
        try:
            if n == "PID":
                import cartpole_pid as mod
            elif n == "LQR":
                import cartpole_lqr as mod
            elif n == "PPO":
                import cartpole_ppo as mod
            else:
                raise ValueError(f"알 수 없는 제어기 이름: {n}")
            ctrls.append(mod.build_controller())
        except Exception as e:
            print(f"[{n}] 건너뜀 → {type(e).__name__}: {e}")
    if not ctrls:
        sys.exit("실행할 수 있는 제어기가 없습니다.")
    return ctrls


def parse_cli_init(args):
    """명령줄의 key=value 를 초기조건으로 변환 (예: theta_deg=20 x=0.5)"""
    init = dict(INIT)
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            if k not in C.BASE_S0:
                sys.exit(f"초기조건 키는 {list(C.BASE_S0)} 중 하나여야 합니다: {k}")
            init[k] = float(v)
    return init


def animate(controllers, init):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    s0 = C.make_s0(init)
    label = C.condition_label(init)
    # 1) 먼저 공통 simulate()로 끝까지 계산 (평가 결과와 완전히 동일한 궤적)
    logs = [C.simulate(c, s0, np.random.default_rng(C.SEED)) for c in controllers]

    rows = []
    for c, log in zip(controllers, logs):
        rows.append(dict(experiment="animate", condition=label, controller=c.name,
                         **C.compute_metrics(log)))
    print(f"\n초기상태 s0 = [x, x_dot, theta, theta_dot] = {np.round(s0, 4)}  ({label})")
    C.print_table(rows)

    # 2) 화면 구성: 위 = 제어기별 카트폴, 아래 = theta / x / F 시간응답
    n = len(controllers)
    fig = plt.figure(figsize=(max(4.5 * n, 9), 8.5))
    gs = fig.add_gridspec(4, n, height_ratios=[1.3, 1, 1, 1], hspace=0.4, top=0.93, bottom=0.06)
    L = C.PHYS["l"]
    cart_w, cart_h = 0.25, 0.12
    pivot_y = cart_h / 2

    anim_art = []          # 프레임마다 갱신할 객체들
    for i, (c, log) in enumerate(zip(controllers, logs)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(-VIEW_X_RANGE, VIEW_X_RANGE)
        ax.set_ylim(-0.25, L + 0.2)
        ax.set_aspect("equal")
        ax.axhline(-cart_h / 2, color="gray", lw=1.5)                     # 레일
        for w in (-C.X_MAX, C.X_MAX):                                    # 이동 한계
            if abs(w) <= VIEW_X_RANGE:
                ax.axvline(w, color="red", ls="--", lw=1)
        ax.axvline(0, color="lightgray", ls=":", lw=1)                   # 목표 위치
        ax.set_title(c.name, color=COLORS.get(c.name, "k"), fontweight="bold")
        ax.set_yticks([])
        cart = Rectangle((0, 0), cart_w, cart_h, fc="slategray", ec="k")
        ax.add_patch(cart)
        pole, = ax.plot([], [], lw=4, color=COLORS.get(c.name, "k"), solid_capstyle="round")
        bob, = ax.plot([], [], "o", ms=9, color=COLORS.get(c.name, "k"))
        force, = ax.plot([], [], lw=3, color="crimson")                  # 힘 화살표(길이 ∝ F)
        txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace")
        fail = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", color="red",
                       fontsize=16, fontweight="bold")
        anim_art.append((log, cart, pole, bob, force, txt, fail))

    ts_axes = [fig.add_subplot(gs[r, :]) for r in (1, 2, 3)]
    ylabels = ["theta [deg]", "x [m]", "F [N]"]
    series = [lambda lg: np.degrees(lg["s"][:, 2]),
              lambda lg: lg["s"][:, 0],
              lambda lg: np.append(lg["u"], lg["u"][-1] if len(lg["u"]) else 0.0)]
    ts_lines = []
    for ax, yl, f in zip(ts_axes, ylabels, series):
        ys = np.concatenate([f(lg) for lg in logs])
        pad = 0.1 * (ys.max() - ys.min() + 1e-6)
        ax.set_xlim(0, C.SIM_TIME); ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_ylabel(yl); ax.grid(alpha=0.3)
        lines = [ax.plot([], [], color=COLORS.get(c.name, None), label=c.name)[0] for c in controllers]
        ts_lines.append((lines, f))
    ts_axes[0].axhspan(-C.SETTLE_TOL_DEG, C.SETTLE_TOL_DEG, color="gray", alpha=0.15)
    ts_axes[0].legend(loc="upper right", fontsize=8)
    ts_axes[2].set_xlabel("time [s]")
    fig.suptitle(f"Cart-Pole balance  |  s0: {label}  |  F_max = ±{C.F_MAX:g} N, dt = {C.DT:g} s")

    N = max(len(lg["s"]) for lg in logs)
    skip = max(1, int(round(1.0 / (C.DT * DISPLAY_FPS))))
    frames = list(range(0, N, skip))
    if frames[-1] != N - 1:
        frames.append(N - 1)

    def update(k):
        artists = []
        for (log, cart, pole, bob, force, txt, fail) in anim_art:
            j = min(k, len(log["s"]) - 1)                 # 실패해 끝난 경우 마지막 상태 유지
            x, _, th, _ = log["s"][j]
            F = log["u"][min(j, len(log["u"]) - 1)] if len(log["u"]) else 0.0
            cart.set_xy((x - cart_w / 2, -cart_h / 2))
            tip = (x + L * np.sin(th), pivot_y + L * np.cos(th))
            pole.set_data([x, tip[0]], [pivot_y, tip[1]])
            bob.set_data([tip[0]], [tip[1]])
            force.set_data([x, x + 0.4 * F / C.F_MAX], [0, 0])
            txt.set_text(f"t={j * C.DT:5.2f}s\nθ={np.degrees(th):6.2f}°\nx={x:6.3f}m\nF={F:6.2f}N")
            fail.set_text("FAILED" if (log["failed"] and j == len(log["s"]) - 1) else "")
            artists += [cart, pole, bob, force, txt, fail]
        for lines, f in ts_lines:
            for line, log in zip(lines, logs):
                j = min(k, len(log["s"]) - 1)
                line.set_data(log["t"][:j + 1], f(log)[:j + 1])
                artists.append(line)
        return artists

    interval_ms = skip * C.DT * 1000 / PLAYBACK_SPEED
    ani = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=True, repeat=False)

    if SAVE_ANIMATION:
        fps = 1000.0 / interval_ms
        writer = "pillow" if SAVE_ANIMATION.endswith(".gif") else "ffmpeg"
        print(f"애니메이션 저장 중... {SAVE_ANIMATION}")
        ani.save(SAVE_ANIMATION, writer=writer, fps=fps)
        print("저장 완료")
    if SHOW:
        plt.show()
    return ani


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args and "=" not in args[0] else MODE
    ctrls = load_controllers(CONTROLLERS)
    if mode == "animate":
        animate(ctrls, parse_cli_init(args))
    elif mode == "compare":
        C.evaluate_and_save(ctrls, "compare", overlay_experiments=OVERLAY_EXPERIMENTS)
    else:
        sys.exit("MODE는 'animate' 또는 'compare'")
