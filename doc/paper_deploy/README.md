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
| `config.paper.yaml` | `/opt/aivc_trade/config.paper.yaml` | paper 用 config（リポジトリ直下に版管理。そのまま使う） |
| `.env.paper` | `/opt/aivc_trade/.env.paper` | 通知先だけ（任意・600・git 管理外） |
| `deploy/aivc-paper.service` | `/etc/systemd/system/` | 本体（常駐） |
| `deploy/aivc-paper-report.service` / `.timer` | `/etc/systemd/system/` | 日次レポート |
| `preflight_paper.py` | リポジトリ直下 | 起動前チェック（公開データ疎通。署名なし） |
| `paper_report.py` | リポジトリ直下 | 日次集計（評価指標） |
| `paper_loop_check.py` | リポジトリ直下 | 有限回の通し確認（`--env paper`） |

`config.paper.yaml` は `aivc_trade/config/config.yaml` をコピーし、次の **2 点だけ**
変えたものをリポジトリ直下に置いている（他は main と同一。ドリフトは
`tests/test_g3_activation.py` が検出する）:

```yaml
exchange:
  env: paper                        # ← モードスイッチ（本番公開データ + 模擬約定）
htf_gate:
  g3:
    enabled: true                   # G3（BTC 日足 < SMA200 → 新規停止）を有効化
    fail_closed: true               # 欠損・陳腐化は data_unavailable で新規停止
```

`persistence` / `paper` / `notification` は main と同じ値。VM 上で絶対パスに変える
場合は差分を記録する。意図的な逸脱（fail-closed）の理由は `RUNBOOK.md` §10。

## 手順（要約）

1. クローン + venv + `requirements.txt`
2. `config.paper.yaml` を配置
3. `.env.paper` を配置（通知先のみ・任意。`deploy/env.paper.example` 参照）
4. `preflight_paper.py` で起動前チェック（0 以外なら起動しない）
5. `paper_loop_check.py --env paper --iterations 3` で有限回の通し確認
6. systemd ユニットを配置して `systemctl enable --now aivc-paper`
7. 日次レポート（timer）を有効化し、初回は手動で `paper_report.py` を実行

詳細・停止条件・障害時の手順は `RUNBOOK.md`。
