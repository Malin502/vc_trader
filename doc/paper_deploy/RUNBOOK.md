# 実板ペーパー RUNBOOK（Nyao VM）

対象: `main_live` の **実板ペーパー**（`exchange.env: paper`）連続運転。
本番の公開データ（認証不要）でシグナルを出し、**実際の板**を消費して模擬約定する。
実マネーは動かない。Testnet 版のユニット／手順はこのバンドルで置き換えた。

役割の整理（2026-11-20 改訂）:

- 性能測定はバックテスト（walk-forward 含む）の仕事。**ペーパーは性能を測らない**
- Testnet は API 結線の実地証明（M4b）までで役割を終えた（systemd は停止済み）
- 実運用での発揮度は「実際の板 × 模擬約定」で見る = 本モード

---

## 0. 前提と禁止事項

- `exchange.env: paper` 以外では起動しない（`preflight_paper.py` が弾く）
- 実板ペーパーは **API キーを読まない**。署名付きエンドポイントへの経路が無い
  （資格情報ファイルは配置不要）
- 作業コピーでのコミットのみ。`main` への push はしない
- 本番 config（`env: prod`）を指すと発注経路が有効になる。`AIVC_EXCHANGE_ENV=paper`
  をユニット側で強制しているが、手動起動時も必ず確認する

---

## 1. 初期配置

```bash
git clone https://github.com/Malin502/vc_trader.git /opt/aivc_trade
cd /opt/aivc_trade
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp aivc_trade/config/config.yaml config.paper.yaml   # 差分は README 参照
install -m 600 deploy/env.paper.example .env.paper   # 通知先だけ（任意）

sudo install -m 644 deploy/aivc-paper.service /etc/systemd/system/
sudo install -m 644 deploy/aivc-paper-report.service /etc/systemd/system/
sudo install -m 644 deploy/aivc-paper-report.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

`config.paper.yaml` で変えるのは次の 3 点だけ（他は main と同一に保つ）:

```yaml
exchange:
  env: paper                       # ← 接続先スイッチ（本番の公開ホストを使う）
persistence:
  state_file: data/position_state.json
  candles_dir: data/candles
  signal_journal_path: data/signals/signal_journal.jsonl
paper:
  initial_equity: 100000.0         # 模擬口座の初期資金（JPY）
  account_file: data/paper/paper_account.json
```

**資格情報は不要**。`.env.testnet` / `BINANCE_API_KEY` は参照されない。
移行後は削除してよい（`paper_report.py` もキーを読まない）。

---

## 2. 起動前チェック（必須）

```bash
cd /opt/aivc_trade
.venv/bin/python preflight_paper.py --config config.paper.yaml
```

見るもの:

- `exchange.env=paper` / `endpoint=https://api.binance.com`
- 監視銘柄すべてで `price` / `book=bids…/asks…` / `1h_bars` が取れている
- 板が空でないこと（空なら模擬約定が ticker fallback に落ちる）
- 終了コード 0

終了コード 2 は「公開データに触れない」状態。原因（ネットワーク・レート制限・
メンテナンス）を記録してから再実行する。0 以外では systemd が本体を起動しない。

---

## 3. 有限回の通し確認（任意・初回推奨）

signal を送れない環境でも必ず止まる確認用ハーネス。本番と同じ `main_live.main()`
を 3 回転だけ回し、停止フラグ経由で状態保存して抜ける。**資格情報は読まない**。

```bash
.venv/bin/python paper_loop_check.py --config config.paper.yaml --iterations 3
# → [loop-check] OK: 3 iterations, env=paper, endpoint=https://api.binance.com
```

`data/signals/signal_journal.jsonl` に `{"event": "STOP", "kind": "RECONCILED",
"ok": true, "venue": "paper_mock"}` が 1 行出ていれば、記録経路も生きている。
`data/paper/paper_account.json` が作られ、模擬口座の残高が残る。

---

## 4. 起動 / 停止

```bash
sudo systemctl enable --now aivc-paper     # 起動
sudo systemctl status aivc-paper
journalctl -u aivc-paper -f                # ログ（loguru は logs/ にも書く）

sudo systemctl stop aivc-paper             # 停止（SIGTERM）
```

- `SIGTERM` は「今のイテレーションを終えて状態を保存してから抜ける」経路に載せて
  ある（`TimeoutStopSec=90`）。`SIGKILL` は使わない（状態が古いまま残る）
- 再起動は `systemctl restart`。起動時に必ず模擬 STOP の復元（reconcile）が走る
  （建玉の `stop_price` から安全網を作り直す。復元できなければ新規エントリを止める）
- 復元時の正は **模擬口座**（`data/paper/paper_account.json`）。state と口座の建玉が
  食い違うと `Position drift` を出して建玉をリセットする（口座ファイルだけ消すと
  建玉が失われるので、消すときは state も一緒に消す）
- `Restart=on-failure`。異常終了は 30 秒後に自動復帰する

---

## 5. 記録される指標（どこを見るか）

すべて `persistence.signal_journal_path` の 1 ファイル（JSONL）に集約。
実板ペーパーのレコードには `venue: "paper_mock"` と `execution: "simulated"` が付く
ので、実約定のジャーナルと機械的に区別できる。

| event | 内容 |
|---|---|
| `SIGNAL` | シグナル時点の想定価格・STOP・スコア・板・`bar_ts`（1h 足の時刻） |
| `FILL` | 模擬約定（`kind` = ENTRY / PARTIAL / EXIT）、`expected_price` との `slippage_bps`、手数料 |
| `STOP` | 模擬 STOP の `PLACED` / `RATCHET` / `RESIZE` / `CANCELLED` / `RECONCILED` / `TRIGGERED` |

`FILL` には板の消費情報（`levels_consumed` / `shortfall_qty` / `source`）が
`PaperBroker.last_fill` から渡る。`source: ticker_fallback` は「板が取れず現在値
＋既定スリッページで埋めた」印なので、その件数は板の取得失敗率として読む。

模擬 STOP が到達して決済された場合は `STOP kind=TRIGGERED` の後に
`FILL kind=EXIT` が続き、FILL 側に `stop_kind: "PAPER_MOCK"` / `stop_price` /
`trigger_price` が付く。**実 STOP との区別はこの 2 か所で行う**。

`bar_ts` があるので、バックテスト側のシグナル（symbol + 1h 足）と機械的に突き合わせ
できる。

---

## 6. 日次レポート

```bash
.venv/bin/python paper_report.py                    # 全期間サマリ
.venv/bin/python paper_report.py --date 2026-11-21  # その日（UTC）
.venv/bin/python paper_report.py --json             # 機械可読
.venv/bin/python paper_report.py \
    --backtest-signals data/backtest_signals.jsonl  # 一致率つき
```

出力: シグナル数 / 約定乖離（median・worst）/ 手数料合計 / 往復コスト実測 /
STOP 操作の成功率 / シグナル一致率 / **停止条件の該当有無**。
既存の集計はそのまま使える（`venue` / `source` は追加フィールドなので無視される）。

停止条件に該当すると終了コード **3**。`--notify` を付けると `DISCORD_WEBHOOK_URL`
へも上げる。timer で毎日回す場合:

```bash
sudo systemctl enable --now aivc-paper-report.timer
journalctl -u aivc-paper-report --since today
```

---

## 7. 停止条件と通知経路

| 条件 | 検知する場所 | 通知 |
|---|---|---|
| 模擬 STOP を復元できない建玉（起動時 reconcile） | `New entries paused (exchange stop): …` で新規エントリ停止 | Discord（`notify_error`） |
| 板が取れず `ticker_fallback` が続く | ログ `PAPER depth fetch failed` / 日次レポートの `source` 内訳 | Discord（レート制限・障害時） |
| 模擬口座の残高不整合（state と paper_account のズレ） | 起動時 `Position drift` / 照合 | Discord |
| シグナル一致率 < 90%（目安） | 日次 `paper_report.py --backtest-signals` | 終了コード 3 / `--notify` で Discord |
| スリッページが S1 想定超（`costs.slippage_bps`） | 日次レポートの `SLIPPAGE_OVER_S1` | 同上 |

実板ペーパーには取引所側 STOP が無いため「STOP 設置失敗」系の停止条件は発生しない。
代わりに **模擬 STOP の復元失敗** を同じ経路（新規エントリ停止）で扱う。
異常を見たら `systemctl stop` → `preflight_paper.py` で状態確認 → 原因を記録して
から再開する。

---

## 8. 運用中の確認（毎日 5 分）

1. `paper_report.py` の「停止条件」が「該当なし」か
2. `journalctl -u aivc-paper --since "24 hours ago" | grep -E "ERROR|paused|fallback"` が空か
3. `fills.slippage_bps.worst` が S1 想定（既定 5bps）内か
4. ジャーナルの全行に `venue: paper_mock` が付いていること（実約定が混ざっていないこと）
5. `data/paper/paper_account.json` の `quote_balance` が想定どおり減っているか
   （`realized_pnl` は取得単価ベースの実現損益。手数料は `quote_balance` 側で引かれる）

---

## 9. 既知の制約（このバンドル時点）

- 模擬 STOP は **プロセス内の価格監視**。板に常駐しないため、プロセスが落ちている
  間は保護されない（再起動時に state の `stop_price` から復元する）。実運用の
  「取引所側 STOP」と同じ保証は無い
- STOP の到達判定は 5 分ループの現在値。約定はその時点の板（bids）を消費するので、
  実 STOP のトリガー価格との差は「板の厚み」ぶんだけ動く
- 板が薄い／取れないときは `ticker_fallback`（現在値 ± `paper.fallback_slippage_bps`）。
  その件数はスリッページ統計から分離して読むこと
- 仮想口座の equity は quote 残高のみ（建玉の含み益は含めない）。サイジングは
  実運用と同じ 1% ルールなので、含み益の扱いが実口座と厳密には一致しない
- 日足ゲート（G3）は Step 2a 実装の照合が済むまで **結線しない**（`gates.g3.enabled`
  は既定 false のまま）。有効化は別途の裁定後
- シグナル一致率の比較対象（バックテスト側のシグナル JSONL）は別途出力が必要
