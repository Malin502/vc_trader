# MS6-prep レポート — 1枠化・SOL 除外の再構成 / 実板記録 / シャドー観測

- 作成: 2026-11-19（engineer 席）
- 対象ブランチ: `feature/m4-stop-wiring`（main 変更・push なし）
- データ: Binance 公開 1h 足キャッシュ（`m5_data.py` 取得済み、`data/candles`）
- 期間: 2024-05-01 〜 2026-11-19（M5 と同一）
- 決定の正本: にゃお側 `spec-v2-draft.md` §8（本タスクはその実装）

## 0. 結論

| 項目 | 結果 |
|---|---|
| Step 0 ゲート（1枠 vs 2枠の再検証） | **通過**。S0/S1 の両方で 1枠が PF・Sharpe・リターンすべて優位。M5 報告値と小数第 3 位まで一致 |
| Step 1 再構成 | 完了（1枠 / BTC・ETH・XRP / paper 1% / MaxDD 運用線を config 化） |
| Step 2 SOL シャドー観測 | 完了（仕組み＋テスト。蓄積は連続運転で開始） |
| Step 3 シグナル時点の実板記録 | 完了（仕組み＋モックテスト。蓄積は連続運転で開始） |
| Step 4 損益分岐 bps | 完了。報告の標準指標に追加し、再構成後は **30.0bps（S1）** |
| テスト | **258 passed**（既存 218 + 新規 40） |

再構成後の代表値（1枠・3ペア・risk 1%・paper）:

| シナリオ | 建玉 | PF | MaxDD | Sharpe | リターン | 期待値/件 | 損益分岐 |
|---|---|---|---|---|---|---|---|
| S0 片道5bps | 64 | 1.767 | -5.00% | 0.882 | +15.24% | +238 JPY | 30.08bps |
| S1 実測ベース | 64 | 1.975 | -4.74% | 1.014 | +17.82% | +278 JPY | 30.01bps |

（参考: 同条件で SOL を運用対象に戻した 4 ペアでは PF 1.39 / 1.51、リターン +13.1% / +16.3%。
1 枠を SOL が奪う形になり、3 ペアの方が良い。）

## 1. Step 0（ゲート）: 1枠 vs 2枠の数値不整合の検証

### 1.1 条件を揃えた再実行

`ms6_step0.py` で、M5 §6.1 の比較を「max_concurrent だけを変える」形で再実行した。
M5 報告時点の比較条件（paper 2%）を明示的に固定している（現在の config 既定は 1% なので、
そのまま流すと再現にならないため）。

同一: データ / 期間（`trade_start=2024-05-01`）/ パラメータ / 枠以外のすべて / シナリオ S0・S1。

| シナリオ | 枠 | 記録数 | 建玉数 | PF | MaxDD | Sharpe | リターン | 期待値/件 | 同時保有ピーク |
|---|---|---|---|---|---|---|---|---|---|
| S0 5bps | 1 | 112 | 97 | **1.338** | **-8.81%** | **0.602** | **+13.12%** | +135 | 1 |
| S0 5bps | 2 | 139 | 123 | 1.077 | -9.87% | 0.193 | +3.77% | +31 | 2 |
| S1 実測 | 1 | 112 | 97 | **1.457** | **-8.35%** | **0.752** | **+16.88%** | +174 | 1 |
| S1 実測 | 2 | 139 | 123 | 1.174 | -9.34% | 0.353 | +8.15% | +66 | 2 |

### 1.2 M5 報告との一致

| 枠 | 記録数 | PF | Sharpe | リターン | MaxDD |
|---|---|---|---|---|---|
| 1（M5 報告 → 再実行） | 112 → 112 | 1.45 → 1.457 | 0.75 → 0.752 | +16.88% → +16.88% | -8.35% → -8.35% |
| 2（M5 報告 → 再実行） | 139 → 139 | 1.17 → 1.174 | 0.35 → 0.353 | +8.15% → +8.15% | -9.34% → -9.34% |

**同一条件での再実行で、M5 報告の数値はそのまま再現した。** 席が指摘した「比較条件の同一性」に
ついては、この再実行の範囲では不整合は検出されなかった（差異は丸めのみ）。

### 1.3 112 vs 139 の内訳（記録数と建玉数の違い）

M5 報告の「記録 112 / 139」は **約定レッグ数**（部分利確 + 最終決済）で、建玉数は 97 / 123。
差の内訳は次のとおり:

| シナリオ | 1枠 | 2枠 | 共通 | 2枠のみ | 1枠のみ |
|---|---|---|---|---|---|
| S0 | 97 | 123 | 93 | 30 | 4 |
| S1 | 97 | 123 | 93 | 30 | 4 |

- **2枠のみ 30 本**（BTC 1 / ETH 7 / XRP 11 / SOL 11）= 2 枠目が空いていたから建った建玉。
  この 30 本の成績は S0 で **-2,904 JPY（期待値 -97/件、PF 0.74）**、S1 で **-1,913 JPY
  （-64/件、PF 0.82）**。共通 93 本は S0 +6,675 / S1 +10,066 JPY。
  → 2 枠目は「スコア下位のシグナルで埋まり、共通部分の利益を食い潰す」という M5 の主張は、
  建玉単位の内訳でも裏付けられた。
- **1枠のみ 4 本**は、2 枠目由来の建玉がエクイティ・クールダウン経路をずらした結果の差分
  （1 枠が 2 枠の部分集合ではない）。

**ゲート判定: 通過。** 1枠優位は S0（保守側）でも再現しており、Step 1 以降に進んだ。

## 2. Step 1: 再構成

### 2.1 config の変更（`aivc_trade/config/config.yaml`）

| キー | 変更前 | 変更後 |
|---|---|---|
| `position.max_concurrent` | 2 | **1** |
| `exchange.symbols` | BTC/ETH/XRP/SOL | **BTC/ETH/XRP** |
| `exchange.shadow_symbols` | （なし） | **`[SOLJPY]`**（新設） |
| `sizing.risk_per_trade_paper` | 0.02 | **0.01**（live と同一） |
| `risk.max_drawdown.review_line_pct` / `safe_line_pct` | （なし） | **0.15 / 0.20**（新設） |
| `shadow.*` | （なし） | 新設（Step 2 の設定） |
| `orderbook.*` | （なし） | 新設（Step 3 の設定） |
| `persistence.signal_journal_path` | （なし） | 新設 |

`max_notional_pct=0.5` は据え置き。risk 1% / stop 2% で qty は 0.5×equity に張り付くため、
実効リスク 1.0%（M5 §6.2 の「paper 2% は実効 1.18%」という乖離が解消）。

### 2.2 SOL を「運用対象外」として明示的に扱う

- `exchange.symbols`（運用対象）と `exchange.shadow_symbols`（観測のみ）を分離した。
  データ取得・特徴量・レジーム判定は **両方**（`loader.watched_symbols()`）、
  シグナルからの発注は **`exchange.symbols` のみ**（`generate_signals` は従来どおり
  `cfg["exchange"]["symbols"]` を走査するので、コード変更なしで SOL は候補にならない）。
- `main_live.py` はデータ取得・lot 情報取得・レジーム判定を `watched_symbols()` で回し、
  建玉管理とエントリーは `symbols` のみを対象にする。
- loader に `operational_symbols()` / `shadow_symbols()` / `watched_symbols()` を追加。
  検証も追加: 重複、運用対象との重なり、建て通貨の不一致を弾く。

### 2.3 MaxDD 運用線

- `risk.max_drawdown.review_line_pct=0.15` / `safe_line_pct=0.20` を config に追加。
- `strategy/risk.py` に `drawdown_fraction()` / `classify_drawdown()` / `drawdown_status()`
  を追加し、レポートで equity ピークからの DD を判定（`OK` / `REVIEW` / `HALT`）して出す。
- ライブループの自動停止には結線していない（モンテカルロ裏付けが別タスクのため。
  現状は「判定を出す」ところまで）。

### 2.4 既存テストの追随

- `tests/test_portfolio_m3.py`: 2 枠機構の検証なので、`_cfg()` で 2 枠 + 4 ペアを明示。
- `tests/test_portfolio_m3.py::test_risk_per_trade_paper_beats_live` →
  `test_risk_per_trade_paper_equals_live`（paper=live=1% が仕様）。明示 `risk_per_trade` の
  比例性は別テストで維持。
- `tests/test_aivc.py` / `tests/test_config_loader.py`: 既定 1% / 3 ペア / 1 枠に更新。

## 3. Step 2: SOL シャドー観測

### 3.1 仕組み（`aivc_trade/ops/shadow.py`）

`ShadowObserver` が 1h バーごとに:

1. 保有中のシャドーポジションを backtest と同じロジックで前進（stop / trail / 部分利確 / 決済）。
   売買ロジックは `Simulator.open_position()` / `Simulator.step_position()`（MS6 で公開した
   ラッパー）を呼ぶので、**バックテストと同一実装**。別実装による期待値のズレがない。
2. 空きがあればシャドー銘柄のシグナルを評価（`generate_signals` をシャドー銘柄だけ・枠は銘柄数
   ぶんで呼ぶ。実運用の 1 枠に引きずられて候補が消えないようにするため）。
3. エントリー / 部分利確 / 決済を JSONL に追記（`shadow.journal_path`）。

記録項目（1 行 1 イベント）:

- `SHADOW_ENTRY`: `trade_id` / `symbol` / `ts` / `entry_price` / `qty` / `stop_price` /
  `planned_risk` / `regime_at_entry` / `score` / `entry_type` / `virtual_equity`
- `SHADOW_PARTIAL_TP`: `ts` / `exit_price` / `qty` / `exit_reason` / `gross_pnl` / `net_pnl` / `cost`
- `SHADOW_EXIT`: `entry_ts` / `exit_ts` / `entry_price` / `exit_price` / `qty` / `exit_reason` /
  `gross_pnl` / `net_pnl` / `cost` / `notional_units` / `holding_hours` / `legs`

仮想資金は `shadow.initial_equity`（10 万円）、サイズは本番と同じ `compute_qty`。
同一銘柄 1 ポジション、決済後は本番と同じクールダウン（連敗で延長）。

### 3.2 再入場条件のカウンタ

`evaluate_reentry(records)`:

```
S0 期待値 = mean( gross_pnl - (commission_bps + slippage_bps)/10000 * notional_units )
eligible  = (n_trades >= min_trades) and (S0 期待値 > 0)
```

- `min_trades` = `shadow.min_trades`（既定 20）、`slippage_bps` = 5.0、`commission_bps` = 10.0。
- 期待値は **記録時のスリッページ設定に依存せず**、`gross_pnl` と `notional_units` から
  S0 前提で再計算する（観測時の想定を後から変えても判定が揺れない）。
- 条件を満たすと `shadow.flag_path`（`data/shadow/reentry_flag.json`）に判定結果を書く。
  書くのは条件成立後の最初の 1 回（ログにも WARNING で残す）。
- **実際の運用復帰は人間の判断**。本タスクは材料を揃えるところまで。

### 3.3 実データでの通し確認（仕組みが動くことの確認）

テスト（モック）だけでなく、実際の SOLJPY 1h 足（2024-05-01〜2026-11-19）で
`ShadowObserver` を通しで回して動作を確認した（`ms6_shadow_replay.py`）。シャドーは枠の
取り合いをしないので、SOL のシグナルを全部拾う。

| 項目 | 結果 |
|---|---|
| エントリー / 決済 | 53 / 53 件（期末に未決済なし）、部分利確 58 回 |
| 決済理由 | STOP_LOSS 32 / EARLY_FAIL_EXIT 12 / TRAILING_STOP 6 / RUNNER 2 / REGIME_RANGE 1 |
| 勝率 | 13/53 = 24.5% |
| S0 期待値（片道5bps+手数料） | **-144.3 JPY/件（PF 0.61）** |
| S1 期待値（片道2.5bps+手数料） | **-124.7 JPY/件（PF 0.65）** |
| 再入場判定 | `eligible = False`（件数は足りるが期待値が非正） |

→ 仕組みは実データで end-to-end に動く。SOL 単体（枠の競合なし）では期待値がマイナスで、
再入場フラグは立たない。**除外の判断と整合する。**

## 4. Step 3: シグナル時点の実板記録

### 4.1 取得

- `BinanceClient.fetch_order_book(symbol, limit)` → `GET /api/v3/depth`（公開 API、認証不要）。
- `data/orderbook.py::summarize_depth()` が以下に要約:
  `best_bid` / `best_ask` / `mid` / `spread_abs` / `spread_bps` / レベル数 /
  ±0.5%・±1% の `bid_base` / `ask_base` / `bid_quote` / `ask_quote` / `imbalance`
  （`orderbook.bands_pct` で変更可）。
- 板が空・交差・価格不正なら `ok: False` + `reason` を返す（壊れた板を正常扱いしない）。

### 4.2 シグナルとの紐づけ

- `ops/signal_journal.py::SignalJournal` が 1 シグナル 1 行の JSONL を書く:
  `signal_id`（`YYYYMMDDTHHMMSS-SYMBOL-8桁`）/ `ts` / `mode`（paper/live）/ `action` /
  `symbol` / `side` / `entry_type` / `score` / `entry_price`（＝想定約定価格）/ `stop_price` /
  `orderbook`（上記要約）。
- `main_live.py` の発注ブロックで、`generate_signals` が返した各シグナルについて板を取り、
  ジャーナルに書いてから発注する。**`entries_blocked`（安全網なし）で発注を見送った場合も
  記録する**ので、「取れなかったシグナルの板」も後から検証できる。
- 板取得の失敗はログ + `ok: False` で記録し、発注処理は止めない（記録は best effort）。

### 4.3 使い方（後日の裏取り）

`signal_journal.jsonl` の `entry_price`（シグナル時点の想定約定価格）と、同じ時刻の実際の
fill（約定履歴）を突き合わせれば、シグナル単位の実効スリッページが出る。`orderbook.spread_bps`
と比較すれば「ハーフスプレッドで説明できる範囲か」も判定できる。

## 5. Step 4: 損益分岐 bps の標準指標化

### 5.1 定義と実装（`aivc_trade/backtest/metrics.py`)

コストは建玉金額に線形なので、建玉ごとに

```
net_i(s) = gross_i - (commission_bps + s) / 10000 * notional_units_i
```

（`notional_units_i` = Σ 約定レッグの `qty × price`、エントリー側・決済側の両方）。
`PF(s) = Σ win(net) / |Σ loss(net)|` は s について単調減少なので、**PF(s)=1 となる s を
二分法で解く**（`breakeven_slippage_bps()`）。

- `PF(0) <= 1` → `0.0`（スリッページ 0 でも赤字）
- `PF(1000bps) > 1` → `inf`（実質無感応）
- それ以外 → 損益分岐値（精度 1e-3 bps）

`m5_report.py` の `slice_report()` に組み込み、全シナリオの JSON に
`breakeven_slippage_bps` が出るようにした（M5 報告の再現値も併記して確認済み）。

### 5.2 再計算値

| 構成 | シナリオ | PF | 損益分岐（片道） | 現行想定との余裕 |
|---|---|---|---|---|
| **再構成後**（1枠/3ペア/1%） | S0 5bps | 1.767 | **30.08bps** | 5bps の 6.0 倍 |
| **再構成後**（1枠/3ペア/1%） | S1 実測 | 1.975 | **30.01bps** | 実測 1.0〜1.5bps の 20〜30 倍 |
| 参考: 4ペア（SOL 含む） | S0 5bps | 1.390 | 20.25bps | 4.0 倍 |
| 参考: 4ペア（SOL 含む） | S1 実測 | 1.511 | 20.38bps | 13〜20 倍 |
| 参考: M5 構成（2枠/4ペア/2%） | S0 5bps | 1.076 | 8.06bps | 1.6 倍 |
| 参考: M5 構成（2枠/4ペア/2%） | S3 10bps | 0.970 | 8.68bps | — |

M5 構成の損益分岐が約 8bps で、S3（10bps）で PF<1 になる事実と整合する
（＝指標として内部整合的）。再構成後は 30bps まで耐える。

## 6. 再構成後の全体結果（`ms6_report.py`）

S0 / S1、1 枠・3 ペア・risk 1%・初期資金 10 万円。

| シナリオ | 建玉 | PF | MaxDD | Sharpe | リターン | 勝率 | 期待値/件 | グロス | コスト | DD 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 5bps | 64 | 1.767 | -5.00% | 0.882 | +15.24% | 36.8% | +238 | +24,348 | 9,112 | OK |
| S1 実測 | 64 | 1.975 | -4.74% | 1.014 | +17.82% | 38.2% | +278 | +24,578 | 6,758 | OK |

銘柄別（S1）: BTC n=4 PF 0.00（-1,339）/ ETH n=30 PF 2.33（+10,868）/ XRP n=30 PF 1.97（+8,292）。
BTC は件数 4 で判断不能（M5 と同じ）。

MaxDD は -5.0%（最悪シナリオでも -4.7〜-5.0%）で、運用線（レビュー 15% / 停止 20%）に対して
余裕がある。ただしこれはリターンが小さいことの裏返しでもある（M5 §6.3 と同じ注意）。

## 7. テスト

`258 passed`（既存 218 + 新規 40）。新規は `tests/test_ms6_reconfigure.py`:

- 既定値（1 枠 / 3 ペア / paper 1% / MaxDD 運用線）と loader の検証（重複・通貨・運用対象との重なり）
- 既定 config で SOL がシグナル候補にならないこと / 1 枠を超えて保有しないこと
- MaxDD 判定（OK / REVIEW / HALT、ピークからの算出）
- シャドー: エントリー→ストップ決済の記録、保有中の二重エントリー禁止、クールダウン、
  `enabled=False` の無効化、JSONL からの読み戻し
- 再入場判定: 20 件未満 / S0 期待値マイナス / 両方満たす / 空、フラグファイル出力
- 板要約: スプレッド・±0.5%/±1% の厚み・imbalance、空板・交差板の拒否
- 板レコーダ: JSONL 追記、取得失敗時の非例外化、無効時 None
- シグナルジャーナル: 板との紐づけ、読み戻し、欠損ファイル
- 損益分岐 bps: 既知の交点（10bps）、赤字時 0、無感応時 inf、空、部分利確レッグの集計

## 8. 変更ファイル

新規:
- `aivc_trade/ops/shadow.py`（シャドー観測）
- `aivc_trade/ops/signal_journal.py`（シグナル記録）
- `aivc_trade/data/orderbook.py`（板の要約・取得）
- `tests/test_ms6_reconfigure.py`
- `ms6_step0.py`（Step 0 ゲート検証）
- `ms6_report.py`（再構成後の再計算）
- `ms6_shadow_replay.py`（シャドー観測の過去データ・リプレイ）
- `doc/ms6_prep_report.md`（本レポート）

変更:
- `aivc_trade/config/config.yaml`（1 枠 / 3 ペア / shadow_symbols / paper 1% / MaxDD 線 / shadow / orderbook）
- `aivc_trade/config/loader.py`（symbol ヘルパ 3 つ、検証 3 つ）
- `aivc_trade/backtest/simulator.py`（`open_position` / `step_position` の公開ラッパー）
- `aivc_trade/backtest/metrics.py`（`position_cost_rows` / `profit_factor_at_slippage` / `breakeven_slippage_bps`）
- `aivc_trade/data/binance_client.py`（`fetch_order_book`）
- `aivc_trade/strategy/risk.py`（DD 判定ヘルパ）
- `aivc_trade/main_live.py`（watched/data_symbols 分離、シャドー、板記録、ジャーナル）
- `m5_report.py`（損益分岐の標準出力、M5 比較条件 paper 2% の固定）
- `doc/README.md`（シンボル・枠・リスク・MaxDD 線の記述を更新）
- `tests/test_aivc.py` / `tests/test_config_loader.py` / `tests/test_portfolio_m3.py`（既定値の追随）

## 9. 制約・未確認事項（正直な申告）

1. **シャドーのポジションは再起動で失われる。** `ShadowObserver` の保有状態はメモリ上のみ。
   再起動を跨いだ建玉は、ジャーナルに `SHADOW_ENTRY` が残るが `SHADOW_EXIT` が付かない。
   連続運転（M6 本体）で再起動が入る場合、その 1 件は未完了として残る（件数・期待値の計算には
   出てこない）。状態永続化は次の改善候補。
2. **SOL 単体の成績は「負け」で、除外と整合する。** 参考として 1 枠・4 ペアで回すと SOL は
   n=41 で PF 1.08（S0）/ 1.14（S1）、期待値 +31〜+57/件とプラスに見えるが、これは
   **枠を取れた（スコア上位だった）41 件だけ**を見ている（選択効果）。枠の競合を外して
   SOL のシグナルを全部拾うと（§3.3 の通し確認）n=53 で **PF 0.61 / 0.65、期待値 -144 / -125**
   と明確にマイナス。加えて 1 枠を SOL が使うと全体が悪化する（PF 1.39→1.77、リターン
   13.1%→15.2%）。→ 除外の判断は「単体で負け」＋「1 枠の希釈」の両面で支持される。
   シャドー観測の再入場条件（20 件・S0 期待値>0）は、この 53 件の実績に対しては未達。
3. **板スナップショットの実蓄積はまだ無い。** 仕組みとモックテストまで。実 API での疎通は
   連続運転の初回シグナルで確認する（`/api/v3/depth` は公開・認証不要）。
4. **損益分岐 bps は「PF=1 になる片道スリッページ」であり、スプレッドの非対称・
   アドバースセレクション・板の入れ替わりは織り込んでいない。** 実効スリッページの裏取りは
   Step 3 の蓄積で行う。
5. M5 報告（`doc/m5_phase1_report.md`）は当時の比較条件（2 枠・4 ペア・paper 2%）のまま。
   `m5_report.py` はその条件を明示的に固定して再現する。**再構成後の数値は `ms6_report.py`**。
6. `m5_report.py` の `config_snapshot` は `max_concurrent: 2` を固定表示している（M5 当時の
   構成を記録するため）。現行 config を見る場合は `ms6_report.json` の `config_snapshot`。
7. ライブループの MaxDD 自動停止は未結線（判定の出力まで）。

## 10. 再現手順

```
.venv/bin/python ms6_step0.py         # Step 0: 1枠 vs 2枠の再検証（paper 2% 条件を固定）
.venv/bin/python ms6_report.py        # 再構成後の S0/S1 + 損益分岐 bps
.venv/bin/python ms6_shadow_replay.py # シャドー観測の過去データ・リプレイ（SOL 単体）
.venv/bin/python m5_report.py         # M5 当時条件の再現（損益分岐 bps 付き）
.venv/bin/python -m pytest -q         # 258 passed
```

出力: `data/ms6_results/step0_gate.json` / `data/ms6_results/ms6_report.json` /
`data/m5_results/m5_report.json`（`data/` は git 管理外）。
