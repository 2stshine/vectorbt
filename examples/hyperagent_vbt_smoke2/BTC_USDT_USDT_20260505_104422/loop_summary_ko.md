# HyperAgent VBT 루프 요약

- 총 세대 수: 3
- 종료 이유: 최대 세대 수에 도달해서 종료했습니다.
- 현재 메타 정책: parent_selection `score_child_prop`, exploration 0.59, family_switch_after_failures 1
- 기억된 강한 패밀리: 없음
- 기억된 약한 패밀리: {'ma_crossover_ls': 1, 'rsi_reversion_ls': 1, 'channel_breakout_ls': 1}
- 최고 세대: 1, family `rsi_reversion_ls`, score 27.06549021841161
- Gen 0: family `ma_crossover_ls`, stage 통과, full `fail`, score 23.68104268527643
  자세히 보기: [generation_summary_ko.md](gen_0000/generation_summary_ko.md)
- Gen 1: family `rsi_reversion_ls`, stage 통과, full `fail`, score 27.06549021841161
  자세히 보기: [generation_summary_ko.md](gen_0001/generation_summary_ko.md)
- Gen 2: family `channel_breakout_ls`, stage 통과, full `fail`, score 23.904265313439296
  자세히 보기: [generation_summary_ko.md](gen_0002/generation_summary_ko.md)

## 메모리 관찰

- ma_crossover_ls: 최근 선택 후보는 fast_window=6, slow_window=72, fast_window=36, slow_window=144 쪽으로 모였습니다.
- rsi_reversion_ls: 최근 선택 후보는 rsi_window=21, long_entry_threshold=25, long_exit_threshold=55, short_entry_threshold=75, rsi_window=21, long_entry_threshold=30, long_exit_threshold=55, short_entry_threshold=80 쪽으로 모였습니다.
- channel_breakout_ls: 최근 선택 후보는 breakout_window=144, exit_window=24, breakout_window=144, exit_window=48 쪽으로 모였습니다.
