# 実板ペーパー配置バンドル（Nyao VM 用）

`exchange.env: paper`（実板ペーパー）の一式。**本番の公開エンドポイント
（api.binance.com）にだけ接続**し、発注は模擬ブローカーが実際の板を消費して行う。
API キーは不要。VM への配置と起動は にゃお が行う前提で、ここでは「何をどこに置き、
どの順で確認するか」を固定する。

Testnet 版（`aivc-trade.service` / `.env.testnet` / `preflight_reconcile.py`）は
本バンドルで置き換えた。Testnet の役割は M4b の API 結線証明までで完了している。

## 前提

- GitHub `main` をクローンした作業ディレクトリ（例: `/opt/aivc_trade`）
- Python 3.10+ と venv（`requirements.txt`）
- 外部への HTTPS（`api.binance.com` の公開エンドポイント）が出られること
- **API キーは不要**（署名付きエンドポイントへの経路が無い）
- 通知を受け取りたい場合は Discord Webhook URL（`DISCORD_WEBHOOK_URL`）

## 置くもの

| ファイル | 置き場所（例） | 用途 |
|---|---|---|
| `config.paper.yaml` | `/opt/aivc_trade/config.paper.yaml` | paper 用 config（下記の差分だけ変える） |
| `.env.paper` | `/opt/aivc_trade/.env.paper` | 通知先だけ（任意・600・git 管理外） |
| `deploy/aivc-paper.service` | `/etc/systemd/system/` | 本体（常駐） |
| `deploy/aivc-paper-report.service` / `.timer` | `/etc/systemd/system/` | 日次レポート |
| `preflight_paper.py` | リポジトリ直下 | 起動前チェック（公開データ疎通。署名なし） |
| `paper_report.py` | リポジトリ直下 | 日次集計（評価指標） |
| `paper_loop_check.py` | リポジトリ直下 | 有限回の通し確認（`--env paper`） |

`config.paper.yaml` は `aivc_trade/config/config.yaml` をコピーして、次の点だけ
変える（他は main と同一に保つ。変えたら差分を記録する）:

```yaml
exchange:
  env: paper                        # ← これがモードスイッチ（本番公開データ + 模擬約定）
  symbols: [BTCJPY, ETHJPY, XRPJPY] # 運用対象（実発注はしない）
  shadow_symbols: [SOLJPY]          # シャドー観測は継続
notification:
  discord_webhook_url: ""           # 実値は DISCORD_WEBHOOK_URL で渡す
persistence:
  state_file: data/position_state.json          # VM 上の絶対パスでもよい
  candles_dir: data/candles
  signal_journal_path: data/signals/signal_journal.jsonl
paper:
  initial_equity: 100000.0          # 模擬口座の初期資金（JPY）
  fallback_slippage_bps: 5.0        # 板が取れないときの片道スリッページ
  account_file: data/paper/paper_account.json
```

## 手順（要約）

1. クローン + venv + `requirements.txt`
2. `config.paper.yaml` を配置
3. `.env.paper` を配置（通知先のみ・任意。`deploy/env.paper.example` 参照）
4. `preflight_paper.py` で起動前チェック（0 以外なら起動しない）
5. `paper_loop_check.py --env paper --iterations 3` で有限回の通し確認
6. systemd ユニットを配置して `systemctl enable --now aivc-paper`
7. 日次レポート（timer）を有効化し、初回は手動で `paper_report.py` を実行

詳細・停止条件・障害時の手順は `RUNBOOK.md`。
