# AIVC Trade – PhaseB 改善仕様書
> 更新: 2026-02-17
> 本書は Codex による実装依頼のための技術仕様。今回のバックテスト分析・バグ修正の結果と、今後の改善項目をまとめる。

---

## 1. 現状スナップショット (2026-02-17 時点)

| 指標 | 値 |
|------|-----|
| バックテスト期間 | 2025-09-01 〜 2026-02-13 (5.5 ヶ月) |
| トレード数 | 24 |
| 総リターン | +10.43% |
| Sharpe 比 | 1.584 |
| 最大 DD | -7.22% |
| 利益係数 | 2.078 |
| 勝率 | 37.5% (9/24) |
| PhaseB Spearman | 0.243 (弱い正の相関) |
| レジームガード発動 | 9 回 |

---

## 2. 今回修正済みのバグ

### Bug 1 ― `size_mult=1.0` ハードコード（最重要）

**ファイル**: `aivc_trade/backtest/simulator.py`
**修正日**: 2026-02-17
**症状**: `Position` 生成時に `size_mult=1.0` がハードコードされており、PhaseB スコアがポジションサイズに一切反映されていなかった。
**根拠**: `simulator.py` の旧 L512 `size_mult=1.0`。`phaseb_score` はフィールドに保存されるが qty 計算に使われていなかった。
**修正内容**:
- `_compute_phaseb_size_mult()` ヘルパーメソッドを追加
- `risk_multiplier * size_mult` を `compute_qty` に渡すよう変更
- `Position.size_mult`, `phaseb_z_score`, `phaseb_raw_signal`, `phaseb_clipped_low/high` を実値で設定

---

### Bug 2 ― `phase_b_sizing` キーが `last_run_stats` に存在しない

**ファイル**: `aivc_trade/backtest/simulator.py` / `aivc_trade/main_backtest.py`
**修正日**: 2026-02-17
**症状**: `main_backtest.py` は `sim.last_run_stats.get("phase_b_sizing", [])` を参照するが、`simulator.py` の `last_run_stats` にそのキーが存在しなかった。結果として `phaseb_sizing.csv` はシミュレーター本体では一度も生成されていなかった。既存の `/logs/phaseb_sizing.csv` は削除済みの別スクリプトが生成したもの。
**修正内容**:
- `run()` に `phase_b_sizing_logs: List[Dict]` を追加
- sizing 発動時にログエントリを追記
- `last_run_stats["phase_b_sizing"] = phase_b_sizing_logs` を追加

---

### Bug 3 ― z-score 正規化のシグマ不一致

**ファイル**: 削除済みの `threshold_sweep.py` (後方互換問題)
**修正日**: 2026-02-17
**症状**: 削除済みの最適化スクリプトが学習セット統計 (`mu_long=0.025, sigma_long=0.00563`) で z-score を計算していたが、実際のバックテストスコア分布 (`mu≈0.039, sigma≈0.009`) と乖離していた。`clip_rate_total=0.0` かつ `size_mult_var=7e-7` という異常に小さい分散はこの逆算で説明できる (有効 sigma ≈ 3.4)。
**修正内容**:
- **ローリング z-score** を採用: バックテスト中に蓄積した直近 `rolling_window=20` 本のスコアで `mean/sigma` を随時更新
- スコアは sizing 計算**後**にヒストリへ追加 (先読みなし)
- 最初の `warmup_n=5` トレードはウォームアップ期間として `size_mult=1.0`

---

## 3. 今後の改善項目

### 優先度マトリクス

| # | 優先 | 対象 | ステータス | 内容 | 期待効果 |
|---|------|------|----------|------|---------|
| A | 🔴 最高 | ML モデル | 未着手 | 目的関数を quantile → binary に変更し再学習 | Spearman 0.243 → 0.4+ |
| B | 🔴 最高 | ML モデル | **済** | 学習データを拡張（220k サンプル / sigma 0.014 に改善） | サンプル数・スコア分布改善済 |
| C | 🟡 高 | gate.py / config | 未着手・B 完了待ち | モデル改善後に `rank_only` → `top_pct` フィルタリングを有効化 | 低品質エントリー除去 |
| D | 🟡 高 | simulator / config | 未着手 | 銘柄数またはエントリー条件を緩和してトレード数を増加 | 統計有意性の確保 |
| E | 🟢 中 | config | 未着手 | Win rate 向上のための early_fail 調整 | 37.5% → 42%+ |
| F | 🟢 中 | feature_builder | 未着手 | 特徴量にファンディングレート・VWAP 乖離を追加 | モデル識別力向上 |

---

### 改善 A: 目的関数を binary 分類に変更

**ファイル**: `aivc_trade/ml/trainer.py`, `aivc_trade/config/config.yaml`

**理由**:
- 現在の分位点回帰 (`alpha=0.8`) は「将来リターンの 80 パーセンタイル予測」であり、勝率を直接最適化していない
- Spearman = 0.243 は識別力が弱い
- binary 分類 (`predict_proba`) にすることで「このトレードが勝ちかどうか」を直接学習できる

**変更内容**:

`config.yaml` の `phaseb.model_long` / `model_short`:
```yaml
# 変更前
objective: quantile
alpha: 0.8

# 変更後
objective: binary   # LightGBM binary classification
alpha: null         # 不要になる
```

`trainer.py` の `train_directional_models()`:
- ラベル生成: `y = (future_return > cost_bps_roundtrip / 10000) ? 1 : 0`  (手数料込みで勝ったか)
- LightGBM パラメータ: `objective="binary"`, `metric="binary_logloss"`
- スコア取得: `model.predict_proba(X)[:, 1]` (勝率スコア)
- 閾値: validation set で `F1-score` が最大になる threshold を採用

**期待効果**: スコアと勝率の相関が上がり Spearman > 0.40 を目標とする

---

### 改善 B: 学習データを 2023 年〜に拡張 ✅ 完了

**ファイル**: `models/phaseb_meta.json` (再学習済み)

**実施済み内容** (`phaseb_meta.json` より確認):

| 項目 | 旧値 | 新値 |
|------|------|------|
| n_train | 44,331 | **220,762** |
| train 期間 | 不明 | 2021-01-01 〜 2023-10-20 |
| val 期間 | 不明 | 2023-10-20 〜 2024-09-25 |
| test 期間 | 不明 | 2024-09-25 〜 2025-08-31 |
| score_stats mu (long) | 0.025 | **0.034** |
| score_stats sigma (long) | 0.00563 | **0.014** |
| score_stats sigma (short) | 0.00397 | **0.012** |

> **重要**: sigma が 0.006 → 0.014 に拡大したことで、ローリング z-score による sizing が実際に機能するようになった（z-score の振れ幅が以前の 2〜3 倍）。

**残課題**: 目的関数は依然として `quantile (alpha=0.8)` のため、改善 A による binary 化が次の優先事項。

---

### 改善 C: `rank_only` → `top_pct` フィルタリングを有効化

**ファイル**: `aivc_trade/config/config.yaml`

**前提**: 改善 A・B でモデル Spearman > 0.35 を達成してから適用する

**理由**:
- 現在 `mode: rank_only` のため PhaseB ゲートは実質素通り
- Spearman が改善すればスコアが低いトレードを除外することで勝率が上がる

**変更内容**:
```yaml
model_long:
  threshold:
    mode: top_pct    # ← rank_only から変更
    value: 0.40      # 上位 40% のシグナルのみ許可 (下位 60% を除外)

model_short:
  threshold:
    mode: top_pct
    value: 0.35      # SHORT はより厳しく (上位 35%)
```

**チューニング方法**: `value` を 0.2〜0.6 で grid search し、validation set の Sharpe が最大になる値を採用する

---

### 改善 D: トレード数の増加

**理由**:
- 24 トレード / 5.5 ヶ月 では統計的有意性なし (95% 信頼区間が広すぎる)
- 最低 60〜80 トレードが必要
- バックテスト期間を延長するか、シグナル数を増やす必要がある

**選択肢**:

| アプローチ | 難易度 | 効果 |
|-----------|--------|------|
| バックテスト期間を 2024 年から開始 | 低 | +20〜30 トレード |
| 対象銘柄を 10 → 15 に増やす | 低 | +10〜15 トレード |
| PhaseA エントリー条件を緩和して PhaseB で絞る | 中 | +15〜25 トレード (品質維持) |
| 4 時間足を追加して多時間足シグナル | 高 | +30〜40 トレード |

**推奨**: まずバックテスト期間延長 (`start_date: "2024-01-01"`) を試みる

```yaml
# config.yaml
backtest:
  start_date: "2024-01-01"   # ← 2025-09-01 から変更
  end_date: "2026-02-13"
```

---

### 改善 E: Win rate 向上 (37.5% → 42%+)

**ファイル**: `aivc_trade/config/config.yaml`

**理由**:
- 現在 37.5% は 2.08 利益係数でかろうじて収益的
- スリッページ増加や regime 変化で容易にマイナスに転落するリスクがある

**調整候補**:
```yaml
# early_fail をより積極的に
early_fail:
  enabled: true
  min_hold_bars: 2     # ← 3 → 2 (早めに損切り判断)
  max_bars: 6          # 変更なし
  mae_atr_k: 1.2       # ← 1.5 → 1.2 (MAE 1.2×ATR で早期切り)
  mfe_atr_k: 0.5       # ← 0.3 → 0.5 (わずかな含み益でも保護)
```

**注意**: この調整は利益係数を下げる可能性があるため、バックテストで勝率・利益係数・Sharpe のトレードオフを必ず確認すること

---

### 改善 F: 特徴量の追加

**ファイル**: `aivc_trade/ml/feature_builder.py`

**追加候補特徴量**:

| 特徴量 | 計算方法 | 理由 |
|--------|---------|------|
| `funding_rate` | Binance funding rate API | 過熱感・ショートスクイーズ検知 |
| `vwap_dev_pct` | `(close - VWAP) / VWAP * 100` | 価格の過/割安判断 |
| `btc_corr_24h` | BTC と対象銘柄の 24h ローリング相関 | BTC 連動リスク |
| `ret_vs_btc_24h` | `ret_24h - btc_ret_24h` | アルファ成分の抽出 |
| `hv_ratio` | `ATR / ATR_50bar_mean` | ボラティリティの相対水準 |

**実装場所**: `aivc_trade/ml/feature_builder.py` の `build_features()` 関数末尾に追加し、`phaseb_meta.json` の `feature_cols` を更新する

---

## 4. テスト・検証方法

### バックテスト実行
```bash
cd /app
python3 -m aivc_trade.main_backtest
```

### モデル再学習
```bash
cd /app
python3 -m aivc_trade.ml.train_phaseb
```

### sizing バグ修正の確認方法
バックテスト後、以下を確認:
```python
# logs/phaseb_sizing.csv が生成されること
# size_mult 列が 1.0 以外の値を持つ行があること
# z_score 列が [-3, 3] 程度の範囲に分布していること
import pandas as pd
df = pd.read_csv("logs/phaseb_sizing.csv")
print(df[["size_mult", "z_score", "clipped_low", "clipped_high"]].describe())
```

### 合格基準

| 指標 | 現状 (Bug 修正前) | 目標 (改善 A+B 後) |
|------|------------------|-------------------|
| Sharpe | 1.584 | >= 1.70 |
| 最大 DD | -7.22% | <= -8% (許容範囲内) |
| 利益係数 | 2.078 | >= 2.20 |
| 勝率 | 37.5% | >= 42% |
| PhaseB Spearman | 0.243 | >= 0.40 |
| `size_mult_var` | ~0 (バグ) | > 0.01 (実際に変動) |

---

## 5. 実装推奨順序

```
[済] Bug 1: size_mult ハードコード修正
[済] Bug 2: phase_b_sizing ログ欠落修正
[済] Bug 3: ローリング z-score 正規化採用

次のステップ:
1. バックテスト実行 → sizing が正しく動作することを確認 (size_mult_var > 0)
[済] 改善 B: 学習データ拡張 (n_train 220k, sigma 0.014)
2. 改善 A: binary 分類で再学習 → Spearman を計測 ← 現在の最優先
3. Spearman > 0.35 なら改善 C: top_pct フィルタ有効化
4. 改善 D: バックテスト期間延長でサンプル数確保
5. 改善 E/F: Win rate・特徴量チューニング
```

---

## 6. 変更済みファイル一覧

| ファイル | 変更日 | 内容 |
|---------|--------|------|
| `aivc_trade/backtest/simulator.py` | 2026-02-17 | Bug 1,2,3 修正: rolling-z sizing 実装、ログ追加 |
| `aivc_trade/config/config.yaml` | 2026-02-17 | `phaseb.sizing` セクション新設 |
| `models/phaseb_long_lgbm.pkl` | 2026-02-17 | モデル再学習 (n_train 220k) |
| `models/phaseb_short_lgbm.pkl` | 2026-02-17 | モデル再学習 (n_train 220k) |
| `models/phaseb_meta.json` | 2026-02-17 | score_stats 更新 (sigma 0.014/0.012) |
