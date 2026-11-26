# optimal_v4 — PID · LQR · PPO 역진자 비교 (v4 보고서용 코드)

공통 목적함수 없이 세 제어기를 각자의 설계 절차로 만들고, **각도 RMSE · 입력 사용량 E · 안정화 시간 Ts** 로 비교합니다.
물리 모델과 적분기는 레포 루트의 `cartpole_common.py` 를 그대로 씁니다. 모든 명령은 이 폴더(`optimal_v4/`)에서 실행합니다.

```
pip install numpy scipy matplotlib stable-baselines3 gymnasium
```

## 파일

| 파일 | 역할 | 보고서 |
| --- | --- | --- |
| `common_v4.py` | 공통 기준: 성공 판정, 세 지표, 초기상태 집합(설계 148 / 검증 600 / 테스트 600, 테스트 시드 20261126), 배치 시뮬레이터, PID·선형·PPO 제어기, Bryson LQR, 극점(PID 안정성은 이산시간으로 판정), 그래프 | 3장, 4.2절 |
| `pid_tuning_v4.py` | PID 수동 튜닝 도구: 설계 집합 평가, 각도별 그래프, 시간응답, 튜닝 기록, 게인 sweep, ±10% 파레토 확인 | 4.1절, 표 4 |
| `ppo_v4.py` | PPO 학습 → 체크포인트 선택 → 검증/테스트 평가 (보상 1/(1+C), C = Bryson 허용 크기로 정규화한 제곱합, 실패 벌점 −10, 시드 5개) | 4.3절, 부록 A.3 |
| `compare_v4.py` | 세 제어기 비교: 종합 표, 초기각별 그림, 30° 시간응답, 대응 비교(t 신뢰구간, Wilcoxon) | 6장, 표 11·12, 그림 1~3 |
| `sensitivity_v4.py` | 민감도: LQR 허용 크기, PID 게인, PPO 보상 허용 크기(재학습) | 5장, 표 8·9·10 |
| `input_limit_v4.py` | 입력 제한 25 → 5 N 축소 실험, `check` 로 실패 원인 검증(PID 적분 처리, LQR 게인 크기, PPO 대 국소 선형 근사) | 7장 |
| `analysis_v4.py` | 본문 보조 수치: 설계 집합 지표, 극점·감쇠비, 식 (6) 확인, PPO 국소 선형 근사 | 2.4, 4.1~4.3, 5.3, 8장 |

## 실행 순서

```
# 1) PID 수동 튜닝 (pid_tuning_v4.py 맨 위 GAINS 를 바꿔 가며 반복)
python pid_tuning_v4.py                 # 평가 + 그래프 + 튜닝 기록 한 줄
python pid_tuning_v4.py sweep kd_x      # 한 게인만 바꿔 보기
python pid_tuning_v4.py pareto          # ±10% 파레토 개선 확인 (종료 조건)
python pid_tuning_v4.py save            # 최종 게인 저장 → models/pid_gains_manual.json

# 2) PPO 학습·선택 (시드 3개 × 2M 스텝, 수십 분~수 시간)
python ppo_v4.py train                  # 시드 0~4
python ppo_v4.py select                 # → models/ppo_v4.zip
python ppo_v4.py eval val

# 3) 비교·분석 (검증 집합으로 확인)
python compare_v4.py val
python sensitivity_v4.py lqr
python sensitivity_v4.py pid
python sensitivity_v4.py ppo            # PPO 재학습 포함, 오래 걸림
python input_limit_v4.py
python input_limit_v4.py check
python analysis_v4.py all

# 4) 모든 설계를 확정한 뒤 마지막에 한 번
python compare_v4.py test
python ppo_v4.py eval test
```

LQR 은 Bryson 허용 크기(`common_v4.BRYSON`, 표 5)에서 매번 계산되며 `models/lqr_bryson.json` 에 기록됩니다.

## 출력 위치

- `models/` — PID 게인, LQR 게인, PPO 모델과 체크포인트, PID 튜닝 기록
- `results/<val|test|sensitivity|input_limit|analysis|pid_tuning>/` — CSV · JSON (엑셀에서 바로 열림), `table11_summary.md`
- `figures/<pid_tuning|val|test|sensitivity|input_limit>/` — PNG 그림

## 주의

- 검증(val)은 개발 중 몇 번이고 써도 되지만, 테스트(test)는 모델을 모두 확정한 뒤 한 번만 봅니다 (3.3절).
  초판(커밋 9b77730)에서 이미 평가한 테스트 시드 20261124 대신, 수정 후에는 새 시드 20261126 을 씁니다. 이전 결과는 `results/archive/` 에 있습니다.
- PID 튜닝 기록은 `models/pid_tuning_log.csv`, 최종 게인은 `models/pid_gains_manual.json` 입니다 (7단계, kpx 2.97).
- PPO 는 학습 도중 일시적으로 성능이 무너질 수 있으므로, 마지막 모델이 아니라 `select` 가 고른 모델을 씁니다.
- 그래프 한글 글꼴은 Windows 맑은 고딕, macOS AppleGothic, Linux Nanum/Noto 를 자동으로 찾습니다.
