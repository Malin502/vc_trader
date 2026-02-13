# AIVCトレード（Crypto / Binance / USDC）設計仕様書

## 1. ゴールと非ゴール
### ゴール
- 有利な局面のみ参加し、安定的に資産を増やす
- Phase A ではAIを使わず、統計・ルールで「壊れない勝ち筋」を作る
- 1日1〜3回のトレード頻度（条件満たさない日は0回）
- 対象：現物ロングのみ
- 最大ドローダウン目標：50%以下
- 1トレード最大損失：口座資産の5%以下
- AWS上で24時間完全自動運用可能な構成

### 非ゴール（Phase A）
- 最高収益率の追求（まずは堅牢性）
- ショート、先物、レバレッジ（現物のみで開始）
- 複雑な深層学習（Phase B/Cで導入）

## 2. 取引仕様
- 取引所: Binance Spot
- シンボル: BTCUSDC, ETHUSDC
- シグナル生成: 1h足
- 執行フィルタ: 5m足
- 監視: 5m足（推奨）
- 最大1ポジション（BTC or ETHのみ）
- クールダウン: 決済後6時間再エントリ禁止
- コスト: 往復0.30%（手数料+スリッページ）

## 3. 戦略概要
- 参加厳選（Selective Participation）
- レジーム分類: TREND_UP / RANGE / CHAOS
- エントリ: 押し目→再加速のブレイク
- 出口: ATR基準ストップ+トレーリング+時間切れ

## 4. 指標・特徴量
- 1h: EMA(20/50), EMA傾き, ADX(14), ATR(14), ATR%, ATR%z, Donchian(20), VolumeSMA(20), recent_swing_low(10)
- 5m: EMA(20/50), ATR(14), micro_vol, spread_proxy, micro_vol_pct90

## 5. レジーム判定
- CHAOS: atrp_z>2.0, close<ema_slowかつadx>25, 直近6h最大下落<-3*atrp
- TREND_UP: ema_fast>ema_slow, slope>0.0015, adx>=18, atrp_z>=-0.5
- RANGE: 上記以外

## 6. エントリ条件
- TREND_UPのみ
- Pullback: |close-ema_fast|/close<=0.3%
- Breakout: close>donchian_high_prev, volume>SMA(volume,20)
- 5mフィルタ: EMA20>EMA50, micro_vol<=90%点
- 追加: ポジション/クールダウン中は不可

## 7. エグジット条件
- 初期ストップ: entry-ATR*2.2, recent_swing_low-ATR*0.3 の高い方
- トレーリング: 含み益>=ATRから追随、幅=ATR*2.5
- 時間切れ: 48h経過かつMFE<1.2*ATR
- レジームCHAOS化: 即時決済

## 8. ポジションサイズ
- 損失上限: 口座資産の5%
- サイズ: risk/stop_dist, lot丸め, 最大投下資金=95%
- BTC/ETH同時シグナル時はスコアで選択

## 9. システム構成
- ディレクトリ: aivc_trade/以下にconfig,core,data,strategy,execution,backtest,ops,main
- 型定義: Candle, FeatureRow, Regime, Signal, Position, Order
- データフロー: binance_client→feature_engine→regime→signal→sizing→order_manager→position_manager→persistence→notifier

## 10. 発注方式
- エントリ: MARKET BUY（スリッページ大ならLIMIT）
- ストップ: Bot側で監視し到達でMARKET SELL

## 11. バックテスト仕様
- データ: Binance Public API
- 約定モデル: 1hシグナルは次バー始値、5mフィルタは直近5m終値
- コスト: 往復0.30%
- 指標: final_equity, CAGR, Sharpe, MaxDD, PF, n_trades, avg_hold_hours, win_rate
- 合格基準: MaxDD<=0.50, PF>=1.20, 月10回以上

## 12. 例外・安全設計
- サーキットブレーカ: 24h-15%, 7d-25%で停止
- 状態永続化: position_state.json
- 通知: Discord Webhook

## 13. AWS配置
- EC2/ECS, systemd, CloudWatch, SecretsManager

## 14. Phase B/C拡張方針
- Phase B: LightGBMによるAIフィルタ
- Phase C: 5m自己教師ありで執行最適化
- signal.pyの後ろにfilter_ai.py/exec_opt.pyを差し込むだけで拡張可能

---

# 今後の設計・拡張方針

## Phase B（AI参加フィルタ）
- Phase AのSignalを入力に「入る/見送る」をAIで分類
- LightGBMモデル、目的: PF改善・DD圧縮・トレード回数最適化
- signal.pyの後ろにfilter_ai.pyを追加、インターフェース固定

## Phase C（執行最適化）
- 1hで方向→5m latentで「踏み上げ/急落」回避
- 目的: スリッページ低減、損切り率低下
- exec_opt.pyをsignal.pyの後ろに差し込む

## その他拡張
- 新ペア追加（USDT建て等）
- 先物/レバレッジ対応（Phase C以降）
- 複数戦略/AIアンサンブル
- Webダッシュボード/可視化

---

# ディレクトリ構成（2026/02/12時点）

```
aivc_trade/
  config/
  core/
  data/
  strategy/
  execution/
  backtest/
  ops/
  main_live.py
  main_backtest.py
```

# テスト
- pytestによる自動テスト: tests/test_aivc.py
- テストカバレッジ: 指標/特徴量/レジーム/シグナル/サイジング/リスク/永続化/バックテスト/統合

---

# 参考: バックテスト期間
- デフォルト: 2023-01-01〜2025-12-31（約3年）
- データは初回実行時に自動取得・Parquet保存
- 期間変更はconfig.yamlで管理（`backtest.start_date` / `backtest.end_date`）
- walk-forward実行はconfig.yamlで切替（`backtest.walkforward`）
