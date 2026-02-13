# PhaseB学習後にバックテスト結果が変化しない件: 修正依頼

## 背景
`train_phaseb` を実行してモデルを更新しても、`main_backtest` の結果（trades/metrics）が変わらない。

調査の結果、主に以下の2系統の問題が確認された。

1. **MLフィルタが実質適用されていない（時刻キー不一致）**
2. **仮に適用されても閾値/学習状態により全通過になりやすい**

---

## 確認済みの問題点（修正対象）

### 1) `ts` の timezone が失われ、推論時に `ts in index` が不一致になる
- 対象: `aivc_trade/ml/feature_builder.py`
- 現状: `compute_ml_features()` 内で `out["ts"] = df["ts"].values`
- 問題: `tz-aware` の `Timestamp` が `tz-naive` になり、
  `simulator.py` 側の `ts`（UTC aware）と一致しない。
- 影響: `aivc_trade/backtest/simulator.py` の `if ml_df is None or ts not in ml_df.index:` が常に真になり、MLフィルタがバイパスされる。

### 2) モデル未ロード時に全通過するため、気づかず無効化状態で動く
- 対象: `aivc_trade/ml/entry_filter.py`
- 現状: モデル未ロード時は warning のみで pass-through（全通過）。
- 問題: 学習成果物が壊れている/見つからない場合でも結果が大きく変わらず、異常検知しづらい。

### 3) 閾値最適化結果が弱く、`latest` が実質全通過設定になっている
- 対象: `aivc_trade/ml/lgbm_trainer.py`（閾値最適化ロジック）
- 現象例: `latest.json` の threshold が `0.3`、fold meta で `test_n_pass == test_n_total`。
- 問題: フィルタとしての選別力が不足。

### 4) バックテストがネットワーク依存で、再現比較（ON/OFF）がしづらい
- 対象: `aivc_trade/main_backtest.py`, `aivc_trade/data/binance_client.py`
- 現状: 起動時に Binance client 初期化/疎通が走る。
- 問題: キャッシュが揃っていてもオフライン環境で検証が失敗しやすい。

---

## 必須修正（Must）

### A. timezone不整合の解消（最優先）
1. `feature_builder.compute_ml_features()` で `ts` を `.values` で落とさない。
   - 例: `out["ts"] = pd.to_datetime(df["ts"], utc=True)` など、**常にUTC aware**で保持。
2. `simulator` 側で `ml_feat_indexed` の index 作成時に timezone を明示的に揃える。
   - 例: `ml_feat["ts"] = pd.to_datetime(ml_feat["ts"], utc=True)` 後に `set_index("ts")`。
3. `ts` 照合前に、比較する両者が同じ timezone（UTC aware）であることを保証。

### B. モデル未ロード時の挙動を選択可能にする
1. `ml_filter` に fail mode を追加（例: `on_missing_model: pass|fail`）。
2. Backtestでは最低でもログを強化し、どちらのモードで動いたか明示。
3. 推奨: CI/検証実行では `fail` を使えるようにして、静かな全通過を防ぐ。

### C. 閾値最適化の見直し
1. `min_trades_for_threshold` の設定と foldサイズの関係を見直し。
2. 閾値候補が1つしか残らない場合のフォールバック方針を定義。
   - 例: `pass_rate` 上限を制約に追加（通過率が高すぎる閾値を除外）。
3. 学習完了時に以下を必ず保存/表示:
   - validation/test の `n_pass`, `pass_rate`, `precision`, `recall`
   - 採用閾値とその選定理由

---

## 推奨修正（Should）

### D. 推論時デバッグ情報の拡充
- `simulator` にデバッグ集計を追加:
  - `signals_total`
  - `signals_scored_by_ml`
  - `signals_missing_ml_ts`
  - `signals_passed_ml`
  - `signals_blocked_ml`
- Backtest終了時に summary を INFOで出す。

### E. オフライン再現性の向上
- キャッシュデータが要件を満たす場合は Binance API 初期化をスキップ。
- `--offline`（または config）でネットワークアクセスを禁止できるようにする。

---

## 受け入れ条件（Acceptance Criteria）

1. **timezone整合**
   - ML特徴の `ts` index と simulator timeline の `ts` が同一型（UTC aware）で一致する。
   - `signals_missing_ml_ts == 0`（少なくともデータが存在する期間では0）。

2. **MLフィルタが実際に効いていること**
   - 同一データ・同一期間で `ml_filter.enabled=true/false` を比較したとき、
     `signals_passed_ml` または最終 `trades.csv` に差分が発生する（モデルに選別力がある場合）。

3. **異常時の検知性**
   - モデル未配置時、設定に応じて `fail` で停止可能。
   - `pass` モードでもログに明確な警告が出る。

4. **閾値選定の妥当性確認が可能**
   - 学習出力に pass_rate を含む閾値評価が残り、`latest.json` の閾値採用根拠を追跡可能。

---

## 実装後に実施してほしい確認

1. ユニットテスト追加
   - `compute_ml_features` の `ts` が UTC aware で保持されること。
   - `simulator` で `ts` 照合が通ること（`ts in ml_df.index` が期待通り）。

2. 回帰確認
   - `ml_filter.enabled=false` の既存挙動に意図しない変更がないこと。

3. 比較実験
   - 同一期間で `enabled=true/false` を実行し、
     `signals_scored_by_ml`, `signals_blocked_ml`, `trades.csv` 差分をレポート。

---

## 参考: 問題箇所
- `aivc_trade/ml/feature_builder.py`
- `aivc_trade/backtest/simulator.py`
- `aivc_trade/ml/entry_filter.py`
- `aivc_trade/ml/lgbm_trainer.py`
- `aivc_trade/main_backtest.py`

