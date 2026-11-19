# ペーパー運転バンドル（Testnet・Nyao VM 配置用）

Step 1'（ペーパー開始の整備）で用意した一式。**Testnet 専用**で、本番 API には
接続しない。VM への配置と起動は にゃお が行う前提で、ここでは「何をどこに置き、
どの順で確認するか」を固定する。

## 前提

- GitHub `main` をクローンした作業ディレクトリ（例: `/opt/aivc_trade`）
- Python 3.10+ と venv（`requirements.txt`）
- Testnet の API キー（`.env.testnet`、**コミットしない**）
- 通知を受け取りたい場合は Discord Webhook URL（`DISCORD_WEBHOOK_URL`）

## 置くもの

| ファイル | 置き場所（例） | 用途 |
|---|---|---|
| `config.testnet.yaml` | `/opt/aivc_trade/config.testnet.yaml` | Testnet 用 config（下記の差分だけ変える） |
| `.env.testnet` | `/opt/aivc_trade/.env.testnet` | 資格情報（600、git 管理外） |
| `deploy/aivc-trade.service` | `/etc/systemd/system/` | 本体（常駐） |
| `deploy/aivc-paper-report.service` / `.timer` | `/etc/systemd/system/` | 日次レポート |
| `preflight_reconcile.py` | リポジトリ直下 | 起動前チェック（systemd の `ExecStartPre`） |
| `paper_report.py` | リポジトリ直下 | 日次集計（評価指標） |
| `paper_loop_check.py` | リポジトリ直下 | 有限回の通し確認（起動前の疎通確認に使う） |

`config.testnet.yaml` は `aivc_trade/config/config.yaml` をコピーして、次の点だけ
変える（他は main と同一に保つ。変えたら差分を記録する）:

```yaml
exchange:
  env: testnet                      # ← これが接続先スイッチ
  symbols: [BTCJPY]                 # 枠数に合わせる（1枠なら1銘柄）
  shadow_symbols: []                # シャドーは Step2 の判断まで止める
notification:
  discord_webhook_url: ""           # 実値は DISCORD_WEBHOOK_URL で渡す
persistence:
  state_file: data/position_state.json          # VM 上の絶対パスでもよい
  candles_dir: data/candles
  signal_journal_path: data/signals/signal_journal.jsonl
```

## 手順（要約）

1. クローン + venv + `requirements.txt`
2. `.env.testnet` を配置（`deploy/env.testnet.example` を参照、`chmod 600`）
3. `config.testnet.yaml` を配置
4. `preflight_reconcile.py` で起動前チェック（0 以外なら起動しない）
5. `paper_loop_check.py --iterations 3` で有限回の通し確認
6. systemd ユニットを配置して `systemctl enable --now aivc-trade`
7. 日次レポート（timer）を有効化し、初回は手動で `paper_report.py` を実行

詳細・停止条件・障害時の手順は `RUNBOOK.md`。
