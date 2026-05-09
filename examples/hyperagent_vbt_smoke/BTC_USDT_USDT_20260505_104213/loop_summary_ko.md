# HyperAgent VBT 루프 요약

- 총 세대 수: 2
- 종료 이유: 최대 세대 수에 도달해서 종료했습니다.
- 현재 메타 정책: parent_selection `score_child_prop`, exploration 0.51, family_switch_after_failures 1
- 기억된 강한 패밀리: 없음
- 기억된 약한 패밀리: {'ma_crossover_ls': 2}
- Gen 0: family `ma_crossover_ls`, stage 실패, full `미실행`, score None
  자세히 보기: [generation_summary_ko.md](gen_0000/generation_summary_ko.md)
- Gen 1: family `ma_crossover_ls`, stage 실패, full `미실행`, score None
  자세히 보기: [generation_summary_ko.md](gen_0001/generation_summary_ko.md)

## 메모리 관찰

- ma_crossover_ls: 최근 선택 후보는 fast_window=6, slow_window=72, fast_window=36, slow_window=144 쪽으로 모였습니다.
