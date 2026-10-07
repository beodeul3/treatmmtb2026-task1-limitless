# v7 official-score + tail-loss patch

변경 사항:

1. Validation objective를 official-like score로 변경
   - `official_score = 0.7 * accuracy + 0.3 * official_mean_dice`
   - official mean Dice는 provided evaluator와 동일하게 both-empty case를 NaN으로 제외합니다.

2. 체크포인트를 4종으로 저장
   - `best_official_score.pt`: recall constraint 없는 official score 최고
   - `best_official_recall90.pt`: recall >= 0.90 조건에서 official score 최고
   - `best_official_recall95.pt`: recall >= 0.95 조건에서 official score 최고
   - `best_legacy_objective.pt`: 이전 legacy objective 최고 비교용
   - 호환용 `best_cavity_mtl.pt`는 `best_official_score.pt`와 같은 내용으로 갱신됩니다.

3. Classification tail loss 추가
   - low-logit positive와 high-logit negative에 extra BCE
   - hard positive/negative pairwise margin loss
   - 추천 초기값: `--cls_tail_weight 0.15 --cls_margin_weight 0.05 --cls_margin 1.0`

4. 후반 sampler schedule 변경
   - 기존 70% 지점 pos/b=2 전환이 recall을 떨어뜨려, 기본값을 `--late_pos2_start 0.85`로 늦춤
   - late hard ratio 기본값도 `0.28`로 완화

5. Validation sweep 결과 저장
   - `val_threshold_sweep_top200_latest.csv`
   - `last_metrics.json`에 official/recall90/recall95/legacy 후보 모두 기록

## v7.1 per-epoch official Mean Dice logging

이 패치에서는 validation이 끝날 때마다 stdout에 아래 요약 라인이 출력됩니다.

```text
VAL official summary:
  best_official: score=... mean_dice=... acc=... recall=... spec=... fn=... fp=... t_cls=... t_mask=... min_area=... dice_count=...
  recall>=0.90: score=... mean_dice=... acc=... recall=... spec=...
  recall>=0.95: score=... mean_dice=... acc=... recall=... spec=...
```

또한 매 epoch 결과가 다음 파일에 누적 저장됩니다.

- `metrics_history.csv`: Excel로 열어 official mean Dice와 score 변화를 바로 확인하는 용도
- `metrics_history.jsonl`: nested validation record 전체를 epoch별로 저장
- `last_metrics.json`: 마지막 epoch validation 결과

특히 `metrics_history.csv`에서 먼저 볼 컬럼은 다음입니다.

- `best_official_official_mean_dice`
- `best_official_official_score`
- `best_official_acc`
- `best_official_recall`
- `best_official_specificity`
- `best_official_t_cls`
- `best_official_t_mask`
- `best_official_min_area`

