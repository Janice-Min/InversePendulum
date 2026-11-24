# optimal_v3 — 카트 위치 조건 없는 최적 PID · LQR · PPO

v2(`optimal_v2/`)와 같은 절차(같은 목적함수 J로 세 제어기를 각각 최적화 → 따로 떼어 둔 상태에서 비교)에서 아래만 바꿨어요.
설정값은 각 파일 맨 앞 `# 0. 설정` 블록에 모여 있어요.

## v2 대비 변경

| | v2 | v3 |
|---|---|---|
| 성공 기준 (마지막 2 s) | \|θ\|≤3°, \|θ̇\|≤0.2 rad/s, **\|x\|≤0.2 m** | \|θ\|≤3°, \|θ̇\|≤0.2 rad/s, **\|ẋ\|≤0.05 m/s** (카트는 어디서든 멈추면 됨) |
| 실패 | \|x\| > 2.4 m | 같음 |
| Settling Time | θ, θ̇, x | θ, θ̇ (`SETTLING_USE_XDOT=True`면 ẋ 포함) |
| LQR 탐색 변수 | Q 대각 4개 (R=1.6e-6 고정) | Bryson 허용 크기 z_max 4개 (R = 1/F_MAX², Q = 1/z_max²) |
| PPO 보상 | 기준 밖 벌점(θ, θ̇, x), w_x = 0.1 | 기준 밖 벌점(θ, θ̇) + 카트 이동 벌점 1[\|ẋ\|>0.05], w_x = 0.01 |
| 평가 세트 | 테스트 1개 (설계 상태를 늘릴 때 결과를 봄) | **val**(= v2 테스트, 개발 중 확인) / **test**(새 seed, 독립 600개, 마지막에 1회) |
| 탐색 기록 | 최종 J | DE·NM 설정, seed별 J, 실제 평가한 후보 수, NM 전후 J (모델 JSON의 `search`) |
| PPO 학습 편차 | 최적 1개만 보고 | W_U 3개 × seed 3개, run별 선택 모델의 평균 ± 표준편차 (`ppo_seeds.csv`) |

- Settling에 ẋ를 넣지 않은 이유: 넣으면 Ts가 진자가 아니라 카트 제동 시간으로 정해져요. 예를 들어 v2 최적 LQR은 θ만 보면 1.34 s, ẋ까지 보면 2.58 s예요(설계 상태 평균).
- 힘 예산 `EFFORT_BUDGET = 115`는 v2와 같게 뒀어요(기준 변경 효과만 비교). `"auto"`로 바꾸면 Bryson 초기 LQR의 effort를 써요.

## 파일

| 파일 | 역할 |
|---|---|
| `optimal_common.py` | 기준(성공·Settling·J), 설계/val/test 상태, 배치 시뮬레이터, Bryson Q·R |
| `optimal_pid_lqr.py` | PID 게인 6개, LQR z_max 4개 최적화 (DE → Nelder-Mead, seed 반복) |
| `optimal_ppo.py` | 새 보상으로 PPO 학습, run별·전체 체크포인트 선택 |
| `optimal_compare.py` | val / test 세트에서 비교 → `results/val/`, `results/test/` |

## 실행 순서

```bash
python optimal_pid_lqr.py ref     # 몇 초: Bryson 초기 LQR 결과 확인
python optimal_pid_lqr.py check   # 몇 초: 배치 시뮬레이터 = 원본 simulate() 확인
python optimal_pid_lqr.py all     # PID·LQR 최적화 (DE seed 3개, 약 30분 — seed 1개당 약 10분)
python optimal_ppo.py grid        # PPO 9 run 학습 + 선택 (2코어 약 1시간)
python optimal_compare.py         # 검증 세트 비교 — 개발 중엔 이것만
python optimal_compare.py test    # 모든 설정을 확정한 뒤 1회
python optimal_pid_lqr.py sweep   # (선택) 힘 예산별 최적화 = 민감도 분석
```

## 주의

- **test 세트**는 결과를 본 뒤 설정을 바꾸면 튜닝에 쓴 셈이 돼요. 실행할 때마다 `results/test/test_log.txt`에 날짜와 모델 해시가 남아요.
- **평가 횟수**: SciPy DE의 `nfev`는 vectorized 모드에서 "호출 횟수"라서, JSON의 `n_candidates_evaluated`(× 설계 상태 148 = 시뮬레이션 수)를 보고하세요.
- PPO 학습 중에 compare를 같이 돌리면 CPU를 나눠 써서 많이 느려져요.
