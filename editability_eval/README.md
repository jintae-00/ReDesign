# Editability Eval (Additive)

기존 데이터셋/기존 코드(`evaluation_figma.py` 등)는 **수정하지 않고**, `editability_eval/`만 추가해서 구성했습니다.

## 핵심 구조

1. **사전 매칭 저장**
- GT:Qwen, GT:Agent 매칭을 먼저 전부 저장
- 이후 Editability Task는 저장된 매칭쌍만 사용

2. **태스크별 GT subset 추출**
- Text: GT text만
- SVG: GT rectangle path만
- Atomic: text/non-text 균형 샘플링

3. **subtask 파일 분리 + wrapper 실행**
- subtask별 파일을 분리
- `run_*_edit.py`에서 카테고리 전체를 일괄 실행하고 qwen/agent 비교 JSON 생성

---

## 파일 구조

### 매칭
- `editability_eval/match_gt_qwen.py`
- `editability_eval/match_gt_agent.py`
- `editability_eval/run_precompute_matching.py` (qwen+agent 한번에)
- `editability_eval/matching_core.py` (GT-centric greedy)

### 공통 유틸
- `editability_eval/common_utils.py`
- `editability_eval/loaders.py`
- `editability_eval/task_common.py`
- `editability_eval/subtasks/common.py`

### Atomic subtasks
- `editability_eval/subtasks/atomic/delete.py`
- `editability_eval/subtasks/atomic/transition.py`
- `editability_eval/subtasks/atomic/rotation.py`
- `editability_eval/subtasks/atomic/opacity.py`
- `editability_eval/subtasks/atomic/z_order.py`
- wrapper: `editability_eval/run_atomic_edit.py`

### Text subtasks
- `editability_eval/subtasks/text/content_recognition.py`
- `editability_eval/subtasks/text/content_modification.py`
- `editability_eval/subtasks/text/style_scaling.py`
- `editability_eval/subtasks/text/style_bold.py`
- `editability_eval/subtasks/text/style_italic.py`
- `editability_eval/subtasks/text/style_recolor.py`
- `editability_eval/subtasks/text/style_combo.py`
- wrapper: `editability_eval/run_text_edit.py`

### SVG subtasks
- `editability_eval/subtasks/svg/super_scaling.py`
- `editability_eval/subtasks/svg/stroke.py`
- `editability_eval/subtasks/svg/corner_radius.py`
- `editability_eval/subtasks/svg/aspect_ratio.py`
- `editability_eval/subtasks/svg/recolor.py`
- wrapper: `editability_eval/run_svg_edit.py`

---

## [1] Text Content Recognition 로직 (요청 반영)

`OCR 후 GT bbox 매칭`이 아니라,

1. GT 기준으로 매칭된 `best_single_pred`를 먼저 고르고
2. 해당 **개별 matched element**에서
- Agent: `parsed.content` 있으면 사용
- Qwen/LayerD식 CCA: content 없으면 해당 element에 OCR 수행
3. CER/WER 계산

구현: `editability_eval/subtasks/text/content_recognition.py`, `editability_eval/subtasks/text/_shared.py`

---

## [2] 원하는 GT subset만 추출 가능한가?

가능합니다.

- Text 태스크: `is_text_gt` 필터
- SVG 태스크: `is_rectangle_path_gt` 필터
- Atomic: 모든 GT에서 후보 생성 후 text/non-text 균형 샘플링

구현: `editability_eval/subtasks/common.py`

---

## [3] 사전 매칭 후 seed 기반 샘플링

- 먼저 매칭을 모델별로 저장
- subtask별 candidate 생성
- random seed로 셔플 후 원하는 개수만 선택
- 각 subtask마다 capacity JSON/summary JSON 생성

---

## [4] 실행 방법

아래는 예시입니다.

### 1) 매칭 사전 계산 (qwen+agent)

현재 매칭 cost는 `GT+Pred 합집합 영역 L1 + IoU`를 사용합니다.
- 기본은 `RGBA` 채널(`--l1-mode rgba`)로 계산하여, 검정 텍스트/투명 배경 같은 경우의 오탐을 줄입니다.
- 필요하면 기존 방식과 동일한 `RGB` 채널(`--l1-mode rgb`)로 바꿀 수 있습니다.
- GT/PRED 교집합이 없으면 `l1=1.0` (최대 페널티)
- 최종 cost는 `lambda_l1 * l1 + lambda_iou * (1 - iou)` 입니다.

```bash
python -m editability_eval.run_precompute_matching \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --output ./editability_matches \
  --lambda-l1 1.0 --l1-mode rgba --max-merge-n 5
```

진행상황/병목 확인용 옵션:

```bash
python -m editability_eval.run_precompute_matching \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --output ./editability_matches \
  --lambda-l1 1.0 --max-merge-n 5 \
  --trace-episodes \
  --detailed-logs \
  --gt-progress-every 20 \
  --gt-progress-sec 5 \
  --slow-episode-sec 15
```

속도 튜닝 팁:
- `--max-merge-n`을 5 -> 3으로 줄이면 greedy trial 수가 크게 감소
- `--min-gt-overlap`을 0.0 -> 0.01~0.05로 올리면 후보 pred 수가 줄어듦
  (bbox 교집합 면적 / GT bbox 면적 기준)

멀티워커 옵션 (CPU-only):
- `--num-workers`: episode 단위 병렬 처리 워커 수
- `--no-resume`: 기존에 생성된 episode json을 건너뛰지 않고 다시 실행

매칭 중 동시 시각화/상세 로그:
- `--visualize-during-matching`: 매칭 수행 중 episode 시각화 PNG 저장
- `--viz-save-pairs`: episode 이미지 외에 GT별 pair PNG도 저장
- `--viz-output`: 시각화 루트 경로 (기본: `<output>/viz`)
- `--detailed-logs`: GT 진행 로그 + episode별 cost/eval/union-render breakdown 출력

```bash
python -m editability_eval.run_precompute_matching \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --output ./editability_matches \
  --lambda-l1 1.0 --max-merge-n 3 \
  --min-gt-overlap 0.01 \
  --num-workers 16 \
  --detailed-logs \
  --visualize-during-matching \
  --viz-save-pairs \
  --viz-output ./editability_match_viz_live
```

### 2) 매칭쌍 시각화 (에피소드별, L1/cost 포함)

```bash
python -m editability_eval.visualize_matching \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --output ./editability_match_viz \
  --model qwen \
  --max-episodes 20
```

옵션:
- `--episode-ids <id1> <id2> ...`: 특정 에피소드만 렌더링
- `--max-rows`: 에피소드 이미지당 GT row 수 제한
- `--panel-width`: GT/Pred/intersection 패널 너비

### 2-1) 매칭과 분리해서 "누락된 시각화만" 생성

매칭은 `--visualize-during-matching` 없이 빠르게 돌리고, 이후 누락분만 채울 때 사용:

```bash
python -m editability_eval.run_visualize_pending_matching \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --viz-root ./editability_match_viz_live \
  --save-pairs
```

특징:
- 공통으로 파싱된 episode 중 `min(개수, 50)`개를 대상으로 qwen+agent를 함께 렌더
- 결과 구조: `viz-root/<episode_id>/{qwen|agent}/episode.png`
- `--save-pairs` 시: `viz-root/<episode_id>/{qwen|agent}/pairs/gt_*.png`
- 이미 qwen/agent 둘 다 있는 episode는 자동 skip
- `--force`를 주면 기존 시각화가 있어도 다시 그림

### 3) Subset manifest 생성 (Atomic/Text/SVG)

```bash
python -m editability_eval.build_category_subsets \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --output ./editability_results/subset_manifest.json \
  --atomic-iou-min 0.5 \
  --atomic-l1-max 0.2
```

### 4) Atomic 전체 subtask 실행 + qwen/agent 비교

```bash
python -m editability_eval.run_atomic_edit \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --output ./editability_results/atomic \
  --subset-manifest ./editability_results/subset_manifest.json \
  --seed 42 --max-tasks-per-subtask 500 \
  --log-every 25 \
  --num-workers 32 \
  --save-pair-viz \
  --pair-viz-max-per-subtask 200
```

`--save-pair-viz`를 켜면 subtask별 element pair 시각화가 저장됩니다.
- 경로 예시: `editability_results/atomic/qwen/atomic_delete/element_pairs/<pair_key>/panel.png`
- 함께 저장: `gt_before.png`, `gt_after.png`, `pred_before.png`, `pred_after.png`, `roi.png`, `meta.json`

### 5) Text 전체 subtask 실행 + qwen/agent 비교

```bash
python -m editability_eval.run_text_edit \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --output ./editability_results/text \
  --subset-manifest ./editability_results/subset_manifest.json \
  --nanobanana-retries 2 \
  --seed 42 --max-tasks-per-subtask 500 \
  --num-workers 32
```

### 6) SVG 전체 subtask 실행 + qwen/agent 비교

```bash
python -m editability_eval.run_svg_edit \
  --figma-data ./figma_data \
  --exp-pairs \
    ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
    ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
  --match-root ./editability_matches \
  --output ./editability_results/svg \
  --subset-manifest ./editability_results/subset_manifest.json \
  --nanobanana-retries 2 \
  --seed 42 --max-tasks-per-subtask 500 \
  --num-workers 32
```

### 7) subtask edit 적용 검증 리포트 생성

```bash
python -m editability_eval.validate_subtask_edits \
  --output ./editability_results/subtask_edit_validation.json
```

---

## 출력

- 매칭 결과: `editability_matches/{qwen|agent}/episodes/*.json`
- 매칭 시각화: `editability_match_viz/{qwen|agent}/*.png`
- 매칭 중 pair 시각화(옵션): `<viz_output>/{qwen|agent}/pairs/<episode_id>/gt_*.png`
- Atomic 비교: `editability_results/atomic/atomic_comparison_qwen_vs_agent.json`
- Text 비교: `editability_results/text/text_comparison_qwen_vs_agent.json`
- SVG 비교: `editability_results/svg/svg_comparison_qwen_vs_agent.json`

`episodes/*.json`의 `match_stats`에는 병목 분석용 상세 지표가 포함됩니다.
- `total_eval_sec`, `total_union_render_sec`, `union_cache_hit/miss`
- `total_candidate_filter_sec`, `pred_cache_prepare_sec`
- `eval_count_empty/single/multi`

각 subtask별로:
- `*_capacity.json` (생성 가능 task 수)
- `*_results.json` (개별 샘플 결과)
- `*_summary.json` (평균 지표)
