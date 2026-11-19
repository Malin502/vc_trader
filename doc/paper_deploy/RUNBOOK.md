# ペーパー運転 RUNBOOK（Testnet / Nyao VM）

対象: `main_live` の Testnet 連続運転（実マネーなし）。目的は **運用検証** と
**バックテスト一致検証**。成績ゲートは置かない（仕様 v2 改訂 2026-11-20）。

---

## 0. 前提と禁止事項

- `exchange.env: testnet` 以外では起動しない（`preflight_reconcile.py` が弾く）
- `.env.testnet` は 600。**リポジトリにコミットしない**（`.gitignore` 済み）
- 作業コピーでのコミットのみ。`main` への push はしない

---

## 1. 初期配置

```bash
git clone https://github.com/Malin502/vc_trader.git /opt/aivc_trade
cd /opt/aivc_trade
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

install -m 600 deploy/env.testnet.example .env.testnet   # 中身を編集
cp aivc_trade/config/config.yaml config.testnet.yaml      # 差分は README 参照

sudo install -m 644 deploy/aivc-trade.service /etc/systemd/system/
sudo install -m 644 deploy/aivc-paper-report.service /etc/systemd/system/
sudo install -m 644 deploy/aivc-paper-report.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

`.env.testnet` に入れるキー（`deploy/env.testnet.example` と同内容）:

- `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` — Testnet の資格情報
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — `main_live` が読む名前。**同じ Testnet の値**を入れる
- `DISCORD_WEBHOOK_URL` — 通知先（未設定なら通知は無効。ログには残る）

---

## 2. 起動前チェック（必須）

```bash
cd /opt/aivc_trade
set -a; . ./.env.testnet; set +a
.venv/bin/python preflight_reconcile.py \
    --config config.testnet.yaml --env-file .env.testnet
```

見るもの:

- `endpoint: https://testnet.binance.vision` になっていること
- `signed GET ok`（時刻同期と残高）
- `open orders` の一覧（想定外の注文が無いこと）
- `reconcile: clean`
- 終了コード 0

**建玉と STOP の照合（起動前 reconcile）で行われること**:

| 取引所の状態 | 動作 |
|---|---|
| 建玉あり / STOP あり | その STOP を採用（重複設置しない） |
| 建玉あり / STOP なし | その場で STOP を設置 |
| 建玉なし / STOP あり（孤児） | 取消 |
| 数量不一致 | STOP 数量を建玉に合わせる |
| 一覧が取れない | **起動しない**（2 で終了） |

終了コード 2 のときは `reconcile` の `details()` を読み、手で片付けてから再実行する。

---

## 3. 有限回の通し確認（任意・初回推奨）

signal を送れない環境でも必ず止まる確認用ハーネス。本番と同じ `main_live.main()`
を 3 回転だけ回し、停止フラグ経由で状態保存して抜ける。

```bash
.venv/bin/python paper_loop_check.py --config config.testnet.yaml --iterations 3
# → [loop-check] OK: 3 iterations, endpoint=https://testnet.binance.vision
```

`data/signals/signal_journal.jsonl` に `{"event": "STOP", "kind": "RECONCILED",
"ok": true}` が 1 行出ていれば、記録経路も生きている。

---

## 4. 起動 / 停止

```bash
sudo systemctl enable --now aivc-trade     # 起動
sudo systemctl status aivc-trade
journalctl -u aivc-trade -f                # ログ（loguru は logs/ にも書く）

sudo systemctl stop aivc-trade             # 停止（SIGTERM）
```

- `SIGTERM` は「今のイテレーションを終えて状態を保存してから抜ける」経路に載せて
  ある（`TimeoutStopSec=90`）。`SIGKILL` は使わない（状態が古いまま残る）
- 再起動は `systemctl restart`。起動時に必ず reconcile が走る
- `Restart=on-failure`。異常終了は 30 秒後に自動復帰する

---

## 5. 記録される指標（どこを見るか）

すべて `persistence.signal_journal_path` の 1 ファイル（JSONL）に集約。

| event | 内容 |
|---|---|
| `SIGNAL` | シグナル時点の想定価格・STOP・スコア・板・`bar_ts`（1h 足の時刻） |
| `FILL` | 実約定（`kind` = ENTRY / PARTIAL / EXIT）、`expected_price` との `slippage_bps`、手数料 |
| `STOP` | 取引所 STOP の `PLACED` / `RATCHET` / `RESIZE` / `CANCELLED` / `RECONCILED` の成否 |

`bar_ts` があるので、バックテスト側のシグナル（symbol + 1h 足）と機械的に突き合わせ
できる。手数料は `fills[].commission` から拾い、不明なら `fee_asset` 空で残す。

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
STOP 成功率 / シグナル一致率 / **停止条件の該当有無**。

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
| 取引所 STOP の失敗 1 件（設置・取消・照合） | 即時。`New entries paused (exchange stop): …` で新規エントリ停止 | Discord（`notify_error`） |
| ラチェット失敗が連続 3 回（`stop_order.ratchet_alert_after`） | `STOP ratchet failing repeatedly: SYM (n consecutive …)` | Discord（明示的にエスカレーション） |
| シグナル一致率 < 90%（目安） | 日次 `paper_report.py --backtest-signals` | 終了コード 3 / `--notify` で Discord |
| スリッページが S1 想定超（`costs.slippage_bps`） | 日次レポートの `SLIPPAGE_OVER_S1` | 同上 |
| 建玉ドリフト・残高不整合 | 起動時 reconcile / 照合失敗 | Discord |

「STOP 失敗 1 件で停止」は **人間の判断**。自動停止はしない（安全網が外れた状態で
新規エントリは止まるが、既存建玉の手仕舞いロジックは動き続ける）。異常を見たら
`systemctl stop` → `preflight_reconcile.py` で状態確認 → 原因を記録してから再開。

---

## 8. 運用中の確認（毎日 5 分）

1. `paper_report.py` の「停止条件」が「該当なし」か
2. `stops.success_rate_pct` が 100% か（失敗があれば `failures` を読む）
3. `fills.slippage_bps.worst` が S1 想定（既定 5bps）内か
4. `journalctl -u aivc-trade --since "24 hours ago" | grep -E "ERROR|paused"` が空か
5. `data/shadow/` は今回止めている（`shadow.enabled: false`）。Step2 の判断後に戻す

---

## 9. 既知の制約（このバンドル時点）

- Testnet の JPY 残高は小さい（例: 18,000 円程度）。`minNotional` に届かず
  「シグナルは出るが発注できない」ことがある。その場合も `SIGNAL` は記録される
  （一致率の分母は確保できる）が、`FILL` は増えない
- 取引所 STOP が **実際にトリガーして約定した** ことの自動検知は未実装。
  現状は `STOP` 操作の成功率と、照合時の `sync`（状態取得）で判断する。
  必要になったら `reconcile` の adopt 経路に「発動済み stop の検出」を足す
- シグナル一致率の比較対象（バックテスト側のシグナル JSONL）は別途出力が必要。
