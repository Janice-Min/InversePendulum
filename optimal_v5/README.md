# v5: 10초 누적 직립시간 평가

기존의 **마지막 2초 유지에 따른 성공/실패 판정**을 제거했다. 주 지표는 다음 두 조건을 동시에 만족한 **누적 직립시간**이며 클수록 좋다.

- `|θ| ≤ 3°`
- `|θ̇| ≤ 0.2 rad/s`

카트 속도는 이 지표의 조건에 포함하지 않는다. `|ẋ| ≤ 0.05 m/s`까지 만족한 시간은 `stationary_upright_time_s`라는 보조 지표로 저장한다. 제어기, 물리 모델, 초기상태 집합과 PPO 보상식은 그대로다.

## 계산 방법

`upright_time_s = DT × 조건을 만족한 적분 후 샘플 수`

`DT = 0.02 s`인 500개 구간의 오른쪽 끝점으로 시간을 근사한다. 초기 상태를 별도의 시간으로 더하지 않으므로 값은 **0~10초**이다. 중간에 조건을 벗어났다가 재진입하면 재진입한 구간도 합산한다. 마지막 구간의 최소 유지시간과 8초 회복 시한은 없다.

레일 `|x| ≤ 2.4 m`를 이탈한 스텝과 이후 구간에는 시간을 적립하지 않는다. 이탈 전 적립한 시간은 보존한다. 평균은 레일 이탈 시행을 포함한 **전체 시행**으로 계산하며, 시간 비율의 분모는 항상 10초다. RMSE와 입력 제곱 적분 E는 기존처럼 이탈 전 관측 구간의 값이므로 레일 이탈 수와 함께 해석한다.

## 결과 파일

| 위치 또는 열 | 의미 |
| --- | --- |
| `results/upright_duration/` | 새 CSV·JSON·종합 표 |
| `figures/upright_duration/` | 새 그래프 |
| `models/pid_tuning_upright_log.csv` | 새 기준의 PID 튜닝 기록 |
| `models/ppo_v4_upright_runs/` | 새 학습 기록과 시드별 선택 모델 |
| `models/ppo_v4_upright.zip` | 새 기준으로 선택한 PPO 정책 |
| `upright_time_s` | 시행별 누적 직립시간 [s]; 요약 표에서는 전체 시행 평균 |
| `upright_fraction` | 누적시간/10초 (0~1) |
| `stationary_upright_time_s` | 직립과 카트 속도 조건을 모두 만족한 누적시간 [s] |
| `survived`, `survival_rate` | 레일 내 생존 여부·비율. 안정화 성공률이 아님 |
| `settling_time_s`, `stop_time_s` | 종료까지 해당 조건을 유지하기 시작한 시각 (보조 지표) |

보조 정착시간은 종료 시 조건을 만족하지 못하거나 레일을 이탈하면 `NaN`이다. 누적 직립시간은 이런 경우에도 0 또는 적립된 시간으로 남는다. 기존 `success`, `success_rate`, `ts_fail10` 열은 새 출력에 없다. 과거 결과 파일은 그대로 남고, 새 결과의 `evaluation_schema`는 `upright_duration_v1`이다.

## 실행 위치와 의존성

저장소 루트에서 `cd optimal_v4` 후 실행한다. 기존 환경을 사용할 수 있다.

```powershell
python -m pip install numpy scipy matplotlib gymnasium stable-baselines3
```

### 기존 제어기를 고정하고 평가 지표만 바꿀 때

```powershell
python compare_v4.py val --ppo-model models/ppo_v4.zip
```

PPO 모델이 없는 환경에서 계산과 그래프를 확인하려면:

```powershell
python compare_v4.py val --no-ppo
```

### 기존 PPO 체크포인트를 새 기준으로 다시 선택할 때

보상식은 바뀌지 않았으므로, 저장된 모든 체크포인트가 있다면 재학습 없이 새 지표로 재선택할 수 있다. 과거 성공 수만으로 누적시간을 복원하지 않고 각 체크포인트의 정책을 실제로 다시 평가한다.

```powershell
python ppo_v4.py select --legacy
python ppo_v4.py eval val
python compare_v4.py val
```

`models/ppo_v4_runs/seed*/ckpt/*.zip`과 `history.json`이 필요하다. 체크포인트가 없으면 현재 모델 하나를 `--ppo-model`로 고정 평가하거나 새로 학습한다. `select --legacy --tag fmax12.5`처럼 보상 민감도 조건에도 적용할 수 있다.

### PID를 조정하고 PPO를 새로 학습할 때

```powershell
python pid_tuning_v4.py --no-log
python pid_tuning_v4.py pareto
python pid_tuning_v4.py save
python ppo_v4.py train
python ppo_v4.py select
python compare_v4.py val
```

PID는 `GAINS`를 수동 조정한다. `pareto`는 누적 직립시간 증가, RMSE·E 감소, 레일 이탈 수 비증가를 확인한다. 새 평가 기준 아래의 확인이므로 기존 게인이 최적이라고 전제하지 않는다.

PPO는 설계 집합 148개의 **평균 누적 직립시간 → 레일 내 생존 수 → 평균 에피소드 보상**의 내림차순으로 체크포인트와 시드를 선택한다. 뒤의 두 항목은 앞 항목이 동률일 때만 사용한다. 학습 보상과 평가 지표는 서로 다른 역할을 하며, 보상식을 직립시간 보상으로 바꾸지는 않았다. 새 학습은 새 run 폴더를 사용한다.

### 설계를 확정한 뒤 평가와 보조 실험

```powershell
python compare_v4.py test
python sensitivity_v4.py lqr
python sensitivity_v4.py pid
python input_limit_v4.py
python input_limit_v4.py check
python input_limit_pid_gain_v4.py
python analysis_v4.py all
```

PPO 보상 민감도는 `python sensitivity_v4.py ppo`로 새 기준의 학습·선택·평가를 실행한다. `--collect`는 새 형식의 기존 평가 결과만 모은다. 기존의 마지막 2초 성공률 요약은 새 요약으로 취급하지 않는다. 테스트 초기상태 생성 규칙과 시드는 유지했으므로 이번 결과는 같은 상태 집합을 새 평가 기준으로 재평가한 결과이다.

## 코드 검사

```powershell
python -m unittest discover -s tests -v
```

직립시간의 0~10초 범위, 재진입 합산, 각도·각속도 조건, 카트 속도 분리, 레일 이탈 후 적립 중단, 전체 시행 평균과 PPO의 기존 체크포인트 재평가를 검사한다.
