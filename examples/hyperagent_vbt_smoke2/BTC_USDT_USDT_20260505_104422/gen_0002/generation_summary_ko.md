# Generation 2 요약

- 부모 generation: 1
- 부모 선택 방식: `score_child_prop`
- 전략 패밀리: `channel_breakout_ls`
- 전략 이름: `channel_breakout_ls`
- 가설: 기존 패밀리의 구조적 약점을 피하기 위해 다른 신호 구조를 시험합니다.
- 제안 이유: `rsi_reversion_ls` 패밀리가 반복적으로 약해서 `channel_breakout_ls` 패밀리로 전환합니다.
- 탐색 공간: `{"breakout_window": [24, 48, 96, 144], "exit_window": [6, 12, 24, 48]}`
- 선택 기준: `total_return`, 후보 8개, 최소 거래 1회, 방향 `both`
- Stage 통과 여부: 통과
- Full verdict: fail
- 점수: 23.904265313439296
- 실패 체크: ['median_test_score', 'test_outperformed_hold_ratio', 'mean_test_return']
- 선택 후보: ['breakout_window=144, exit_window=24', 'breakout_window=144, exit_window=48']
- 메타 반성: 최근 여러 세대가 연속 실패해서 탐색 비중을 높이고 패밀리 전환을 더 빠르게 하도록 조정했습니다. 홀드 초과 성능이 없어서 성과 좋은 계보만 파는 대신 넓게 분기하는 선택 전략을 유지합니다.
- 메타 정책 스냅샷: parent_selection `score_child_prop`, exploration 0.59, family_switch_after_failures 1
- 전략 메모: 패밀리 로테이션을 적용했습니다.
- 백테스트 요약: [summary_ko.md](gen_0002/full_runs/BTC_USDT_USDT_20260505_104525/summary_ko.md)

## 시각화

### 홀드 비교
![홀드 비교](gen_0002/full_runs/BTC_USDT_USDT_20260505_104525/plots/hold_comparison.png)

### 수익률 타임라인
![수익률 타임라인](gen_0002/full_runs/BTC_USDT_USDT_20260505_104525/plots/return_timeline.png)

### 오버피팅 타임라인
![오버피팅 타임라인](gen_0002/full_runs/BTC_USDT_USDT_20260505_104525/plots/overfitting_timeline.png)
