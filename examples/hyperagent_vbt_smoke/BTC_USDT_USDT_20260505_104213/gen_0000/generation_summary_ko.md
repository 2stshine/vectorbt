# Generation 0 요약

- 부모 generation: None
- 부모 선택 방식: `score_child_prop`
- 전략 패밀리: `ma_crossover_ls`
- 전략 이름: `ma_crossover_ls`
- 가설: 초기 세대에서는 추세 추종형 롱/숏 MA 크로스로 기준선을 잡습니다.
- 제안 이유: 기본 기준 전략을 먼저 평가해 이후 세대가 비교할 출발점을 만듭니다.
- 탐색 공간: `{"fast_window": [6, 12, 18, 24, 36, 48, 72, 96, 144], "slow_window": [6, 12, 18, 24, 36, 48, 72, 96, 144]}`
- 선택 기준: `sharpe_ratio`, 후보 8개, 최소 거래 1회, 방향 `both`
- Stage 통과 여부: 실패
- Full verdict: 미실행
- 점수: None
- 실패 체크: ['median_test_score', 'test_outperformed_hold_ratio', 'mean_test_return']
- 선택 후보: ['fast_window=6, slow_window=72', 'fast_window=36, slow_window=144']
- 메타 반성: 최근 여러 세대가 연속 실패해서 탐색 비중을 높이고 패밀리 전환을 더 빠르게 하도록 조정했습니다. 홀드 초과 성능이 없어서 성과 좋은 계보만 파는 대신 넓게 분기하는 선택 전략을 유지합니다. validation 대비 test 하락폭이 커서 stage 기준을 조금 더 보수적으로 만들었습니다.
- 메타 정책 스냅샷: parent_selection `score_child_prop`, exploration 0.43, family_switch_after_failures 1
- 전략 메모: seed 전략입니다.
- 백테스트 요약: [summary_ko.md](gen_0000/stage_runs/BTC_USDT_USDT_20260505_104213/summary_ko.md)

## 시각화

### 홀드 비교
![홀드 비교](gen_0000/stage_runs/BTC_USDT_USDT_20260505_104213/plots/hold_comparison.png)

### 수익률 타임라인
![수익률 타임라인](gen_0000/stage_runs/BTC_USDT_USDT_20260505_104213/plots/return_timeline.png)

### 오버피팅 타임라인
![오버피팅 타임라인](gen_0000/stage_runs/BTC_USDT_USDT_20260505_104213/plots/overfitting_timeline.png)
