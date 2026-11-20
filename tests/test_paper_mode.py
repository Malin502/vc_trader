"""実板ペーパー（``exchange.env: paper``）のテスト.

* env 解決（paper は本番の公開ホストを指す）
* 安全ガード（資格情報を読まない / 署名クライアントを作らない / 署名面が無い）
* 実板の板を消費する模擬約定（スリッページ・板不足・板なし fallback）
* 模擬 STOP（価格監視 → トリガー）とジャーナル上の実 STOP との区別
* 仮想口座の再起動維持
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import aivc_trade.data.public_client as public_client_mod
import aivc_trade.main_live as main_live
from aivc_trade.config.loader import is_paper_env, load_config, resolve_base_url
from aivc_trade.core.types import Order, OrderType, Position, Side
from aivc_trade.data.orderbook import simulate_market_fill
from aivc_trade.data.public_client import PublicMarketClient
from aivc_trade.execution.paper_broker import PaperBroker
from aivc_trade.execution.paper_stop import PaperStopManager
from aivc_trade.ops.signal_journal import SignalJournal

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "aivc_trade" / "config" / "config.yaml"


def _paper_cfg(tmp_path: Path, **paper_over):
    cfg = load_config(CONFIG)
    cfg["exchange"]["env"] = "paper"
    cfg["persistence"].update(
        {
            "state_file": str(tmp_path / "state.json"),
            "candles_dir": str(tmp_path / "candles"),
            "signal_journal_path": str(tmp_path / "journal.jsonl"),
        }
    )
    cfg.setdefault("paper", {})
    cfg["paper"].update(
        {
            "initial_equity": 100_000.0,
            "fallback_slippage_bps": 5.0,
            "account_file": str(tmp_path / "paper_account.json"),
        }
    )
    cfg["paper"].update(paper_over)
    return cfg


class FakePublicClient:
    """板と現在値だけを返す公開クライアント（HTTP なし）。"""

    def __init__(self, depth=None, ticker=1000.0, depth_error=None):
        self.depth = depth
        self.ticker = ticker
        self.depth_error = depth_error
        self.depth_calls = 0

    def fetch_order_book(self, symbol, limit=100):
        self.depth_calls += 1
        if self.depth_error is not None:
            raise self.depth_error
        return self.depth

    def get_ticker_price(self, symbol):
        return self.ticker


BOOK = {
    "lastUpdateId": 1,
    "bids": [[999.0, 1.0], [998.0, 2.0]],
    "asks": [[1000.0, 1.0], [1001.0, 2.0]],
}


# ----------------------------------------------------------------------
# env resolution
# ----------------------------------------------------------------------
class TestEnvResolution:
    def test_paper_env_uses_prod_public_host(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        assert is_paper_env(cfg) is True
        assert resolve_base_url(cfg) == "https://api.binance.com"

    def test_testnet_still_resolves_testnet_host(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        cfg["exchange"]["env"] = "testnet"
        assert is_paper_env(cfg) is False
        assert resolve_base_url(cfg) == "https://testnet.binance.vision"

    def test_prod_unchanged(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        cfg["exchange"]["env"] = "prod"
        assert resolve_base_url(cfg) == "https://api.binance.com"


# ----------------------------------------------------------------------
# safety guard: no signed surface in paper mode
# ----------------------------------------------------------------------
class TestPaperSafetyGuard:
    def test_public_client_has_no_signed_surface(self):
        client = PublicMarketClient("https://api.binance.com")
        for attr in (
            "api_key",
            "api_secret",
            "_signed_request",
            "place_market_order",
            "place_limit_order",
            "place_stop_order",
            "cancel_order",
            "cancel_replace_order",
            "get_order",
            "get_open_orders",
            "get_account_balance",
            "get_spot_balance",
        ):
            assert not hasattr(client, attr), f"unexpected signed surface: {attr}"

    def test_public_client_module_has_no_signing_code(self):
        src = Path(public_client_mod.__file__).read_text(encoding="utf-8")
        for token in ("import hmac", "import hashlib", "signature=", "X-MBX-APIKEY"):
            assert token not in src, f"signing artefact found: {token}"

    def test_paper_stack_never_builds_signed_client(self, tmp_path, monkeypatch):
        cfg = _paper_cfg(tmp_path)
        # 資格情報が環境にあっても paper は読まない。
        monkeypatch.setenv("BINANCE_API_KEY", "SHOULD-NOT-BE-READ")
        monkeypatch.setenv("BINANCE_API_SECRET", "SHOULD-NOT-BE-READ")

        def _forbidden(*args, **kwargs):
            raise AssertionError("signed client built in paper mode")

        monkeypatch.setattr(main_live, "BinanceClient", _forbidden)
        monkeypatch.setattr(main_live, "StopOrderManager", _forbidden)
        monkeypatch.setattr(main_live, "LiveBroker", _forbidden)

        stack = main_live.build_exchange_stack(cfg)

        assert stack.env == "paper"
        assert stack.is_paper is True
        assert isinstance(stack.client, PublicMarketClient)
        assert isinstance(stack.broker, PaperBroker)
        assert isinstance(stack.stop_om, PaperStopManager)
        assert stack.account is stack.broker
        assert not hasattr(stack.client, "api_key")

    def test_prod_stack_still_uses_signed_client(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        cfg["exchange"]["env"] = "prod"
        stack = main_live.build_exchange_stack(cfg)
        assert stack.is_paper is False
        assert isinstance(stack.client, main_live.BinanceClient)
        assert isinstance(stack.broker, main_live.LiveBroker)
        assert isinstance(stack.stop_om, main_live.StopOrderManager)

    # -- 資格情報そのものを読まない（cfg に載らない）--------------------
    def _secret_cfg(self, tmp_path, env):
        cfg = load_config(CONFIG)
        cfg["exchange"]["env"] = env
        cfg["secrets"] = {"api_key": "FROM-FILE", "api_secret": "FROM-FILE"}
        cfg["persistence"].update(
            {
                "state_file": str(tmp_path / "state.json"),
                "candles_dir": str(tmp_path / "candles"),
                "signal_journal_path": str(tmp_path / "journal.jsonl"),
            }
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return path

    def test_paper_load_config_drops_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "SHOULD-NOT-BE-READ")
        monkeypatch.setenv("BINANCE_API_SECRET", "SHOULD-NOT-BE-READ")
        path = self._secret_cfg(tmp_path, "paper")

        cfg = load_config(path)

        # 環境変数も config.yaml の secrets も、paper では cfg に載らない。
        assert "secrets" not in cfg
        assert is_paper_env(cfg) is True

    def test_env_override_to_paper_also_drops_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "SHOULD-NOT-BE-READ")
        monkeypatch.setenv("AIVC_EXCHANGE_ENV", "paper")
        path = self._secret_cfg(tmp_path, "prod")

        cfg = load_config(path)

        assert cfg["exchange"]["env"] == "paper"
        assert "secrets" not in cfg

    def test_testnet_load_config_still_reads_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "testnet-key")
        monkeypatch.setenv("BINANCE_API_SECRET", "testnet-secret")
        path = self._secret_cfg(tmp_path, "testnet")

        cfg = load_config(path)

        assert cfg["secrets"]["api_key"] == "testnet-key"
        assert cfg["secrets"]["api_secret"] == "testnet-secret"


# ----------------------------------------------------------------------
# simulated fills against the real book
# ----------------------------------------------------------------------
class TestSimulatedFill:
    def test_buy_consumes_multiple_ask_levels(self):
        fill = simulate_market_fill(BOOK, "BUY", 2.5)
        assert fill["ok"] and fill["source"] == "book"
        assert fill["levels_consumed"] == 2
        assert fill["filled_qty"] == pytest.approx(2.5)
        assert fill["avg_price"] == pytest.approx((1000.0 * 1 + 1001.0 * 1.5) / 2.5)
        # mid より不利（買いは高い方へ滑る）
        assert fill["avg_price"] > fill["mid"]

    def test_sell_consumes_multiple_bid_levels(self):
        fill = simulate_market_fill(BOOK, "SELL", 2.5)
        assert fill["avg_price"] == pytest.approx((999.0 * 1 + 998.0 * 1.5) / 2.5)
        assert fill["avg_price"] < fill["mid"]

    def test_thin_book_records_shortfall(self):
        thin = {"bids": [[999.0, 0.1]], "asks": [[1000.0, 0.1]]}
        fill = simulate_market_fill(thin, "BUY", 1.0)
        assert fill["filled_qty"] == pytest.approx(1.0)
        assert fill["shortfall_qty"] == pytest.approx(0.9)
        assert fill["avg_price"] == pytest.approx(1000.0)

    def test_empty_book_uses_ticker_fallback(self):
        fill = simulate_market_fill({}, "BUY", 1.0, fallback_price=1000.0, fallback_slippage_bps=5.0)
        assert fill["source"] == "ticker_fallback"
        assert fill["avg_price"] == pytest.approx(1000.5)

    def test_fill_refused_without_book_or_ticker(self):
        fill = simulate_market_fill({}, "BUY", 1.0)
        assert fill["ok"] is False


class TestPaperBroker:
    def _broker(self, tmp_path, client, cfg=None):
        cfg = cfg or _paper_cfg(tmp_path)
        return PaperBroker(client, cfg)

    def test_fills_from_book_and_updates_account(self, tmp_path):
        broker = self._broker(tmp_path, FakePublicClient(BOOK))
        order = Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=2.5)

        filled = broker.execute(order)

        assert filled.status == "FILLED"
        assert filled.filled_price == pytest.approx(1000.6)
        assert filled.filled_qty == pytest.approx(2.5)
        assert filled.fee == pytest.approx(2501.5 * 10 / 10_000)
        assert filled.fee_asset == "JPY"
        assert broker.get_account_balance("JPY") == pytest.approx(
            100_000.0 - 2501.5 - filled.fee
        )
        assert broker.get_spot_balance("BTC") == pytest.approx(2.5)
        assert broker.last_fill["levels_consumed"] == 2
        assert broker.last_fill["source"] == "book"

    def test_sell_reduces_base_balance(self, tmp_path):
        broker = self._broker(tmp_path, FakePublicClient(BOOK))
        broker.execute(
            Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=2.5)
        )
        broker.execute(
            Order(symbol="BTCJPY", side=Side.SELL, order_type=OrderType.MARKET, qty=2.5)
        )
        assert broker.get_spot_balance("BTC") == pytest.approx(0.0)
        assert broker.get_account_balance("JPY") > 0

    def test_book_failure_falls_back_to_ticker(self, tmp_path):
        broker = self._broker(
            tmp_path, FakePublicClient(ticker=1000.0, depth_error=RuntimeError("429"))
        )
        filled = broker.execute(
            Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=1.0)
        )
        assert filled.filled_price == pytest.approx(1000.5)
        assert broker.last_fill["source"] == "ticker_fallback"
        assert broker.last_fill["book_available"] is False

    def test_account_survives_restart(self, tmp_path):
        client = FakePublicClient(BOOK)
        cfg = _paper_cfg(tmp_path)
        broker = PaperBroker(client, cfg)
        broker.execute(
            Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=1.0)
        )
        quote = broker.get_account_balance("JPY")
        base = broker.get_spot_balance("BTC")

        restored = PaperBroker(client, cfg)
        assert restored.get_account_balance("JPY") == pytest.approx(quote)
        assert restored.get_spot_balance("BTC") == pytest.approx(base)
        assert restored.fills == 1

    def test_realized_pnl_tracks_round_trip(self, tmp_path):
        client = FakePublicClient({"bids": [[990.0, 5.0]], "asks": [[1000.0, 5.0]]})
        cfg = _paper_cfg(tmp_path)
        broker = PaperBroker(client, cfg)
        broker.execute(
            Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=1.0)
        )
        assert broker.realized_pnl == pytest.approx(0.0)

        # 取得単価 1000 → 990 で決済: 1.0 * (990 - 1000) = -10
        client.depth = {"bids": [[990.0, 5.0]], "asks": [[1000.0, 5.0]]}
        broker.execute(
            Order(symbol="BTCJPY", side=Side.SELL, order_type=OrderType.MARKET, qty=1.0)
        )
        assert broker.realized_pnl == pytest.approx(-10.0)

        restored = PaperBroker(client, cfg)
        assert restored.realized_pnl == pytest.approx(-10.0)
        assert restored.cost_basis == {}


# ----------------------------------------------------------------------
# simulated stop
# ----------------------------------------------------------------------
class TestPaperStop:
    def _position(self):
        return Position(
            symbol="BTCJPY",
            qty=0.01,
            entry_price=1000.0,
            stop_price=950.0,
        )

    def test_safety_net_price_applies_buffer(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        stop = PaperStopManager(cfg)
        buffer_pct = float(cfg["stop_order"]["safety_buffer_pct"])
        assert stop.safety_net_price(1000.0) == pytest.approx(1000.0 * (1 - buffer_pct / 100))

    def test_ratchet_only_raises(self, tmp_path):
        stop = PaperStopManager(_paper_cfg(tmp_path))
        pos = self._position()
        stop.place_initial_stop(pos, 950.0)
        stop.update_stop(pos, 940.0)  # 下げない
        assert stop.get_state("BTCJPY").stop_price == pytest.approx(950.0)
        stop.update_stop(pos, 970.0)
        assert stop.get_state("BTCJPY").stop_price == pytest.approx(970.0)

    def test_trigger_fires_at_or_below_stop(self, tmp_path):
        stop = PaperStopManager(_paper_cfg(tmp_path))
        pos = self._position()
        stop.place_initial_stop(pos, 950.0)
        assert stop.poll_trigger("BTCJPY", 951.0) is None
        record = stop.poll_trigger("BTCJPY", 949.0)
        assert record is not None
        assert record["stop_price"] == pytest.approx(950.0)
        assert record["qty"] == pytest.approx(0.01)
        assert stop.is_protected("BTCJPY") is False
        # 二重トリガーしない
        assert stop.poll_trigger("BTCJPY", 940.0) is None

    def test_resize_and_cancel(self, tmp_path):
        stop = PaperStopManager(_paper_cfg(tmp_path))
        pos = self._position()
        stop.place_initial_stop(pos, 950.0)
        assert stop.resize_stop(pos, 0.005).qty == pytest.approx(0.005)
        assert stop.is_protected("BTCJPY")
        assert stop.cancel("BTCJPY") is True
        assert stop.is_protected("BTCJPY") is False

    def test_reconcile_rebuilds_stop_after_restart(self, tmp_path):
        cfg = _paper_cfg(tmp_path)
        pos = self._position()
        restored = PaperStopManager(cfg)  # 再起動＝状態なし
        report = restored.reconcile([pos])
        assert report.has_errors is False
        assert report.placed == ["BTCJPY"]
        assert restored.is_protected("BTCJPY")

    def test_reconcile_accepts_mapping_like_main_live(self, tmp_path):
        """main_live / StopOrderManager と同じ ``{symbol: Position}`` を受ける."""
        cfg = _paper_cfg(tmp_path)
        pos = self._position()
        restored = PaperStopManager(cfg)

        report = restored.reconcile({"BTCJPY": pos})  # リストではなくマッピング

        assert report.positions == 1
        assert report.placed == ["BTCJPY"]
        assert restored.is_protected("BTCJPY")
        # 復元値は安全網（bot 側 stop の少し下）であること
        buffer_pct = float(cfg["stop_order"]["safety_buffer_pct"])
        assert restored.get_state("BTCJPY").stop_price == pytest.approx(
            950.0 * (1 - buffer_pct / 100)
        )

    def test_reconcile_reports_unbuildable_position(self, tmp_path):
        stop = PaperStopManager(_paper_cfg(tmp_path))
        broken = Position(symbol="BTCJPY", qty=0.0, entry_price=0.0, stop_price=0.0)
        report = stop.reconcile([broken])
        assert report.has_errors is True
        assert "cannot rebuild" in report.details()[0]


# ----------------------------------------------------------------------
# journal: distinguish real stops from simulated ones
# ----------------------------------------------------------------------
class TestJournalDistinction:
    def test_paper_records_are_tagged(self, tmp_path):
        journal = SignalJournal(
            tmp_path / "paper.jsonl",
            mode="paper",
            defaults={"venue": "paper_mock", "execution": "simulated"},
        )
        journal.record_stop(
            symbol="BTCJPY", kind="TRIGGERED", ok=True, stop_price=950.0, qty=0.01
        )
        row = json.loads((tmp_path / "paper.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert row["kind"] == "TRIGGERED"
        assert row["venue"] == "paper_mock"
        assert row["execution"] == "simulated"

    def test_live_records_stay_untagged(self, tmp_path):
        journal = SignalJournal(tmp_path / "live.jsonl", mode="live")
        journal.record_stop(symbol="BTCJPY", kind="PLACED", ok=True, stop_price=950.0)
        row = json.loads((tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert "venue" not in row
        assert "execution" not in row


# ----------------------------------------------------------------------
# one iteration of the real main loop in paper mode (fake public data)
# ----------------------------------------------------------------------
class TestPaperLoopEntry:
    """``main_live.main()`` を paper で 1 回転させ、entry → 記録まで通す."""

    def test_entry_fills_from_book_and_journal_is_tagged(self, monkeypatch, tmp_path):
        import types
        from datetime import datetime, timezone

        import pandas as pd

        from aivc_trade.core.types import Regime, Side, Signal
        from aivc_trade.ops.persistence import StateManager

        now = datetime(2026, 11, 20, 3, 1, tzinfo=timezone.utc)
        price = 5_000_000.0

        cfg = _paper_cfg(tmp_path)
        cfg["exchange"]["symbols"] = ["BTCJPY"]
        cfg["exchange"]["shadow_symbols"] = []
        cfg["shadow"]["enabled"] = False
        cfg["notification"]["discord_webhook_url"] = ""
        monkeypatch.setattr(main_live, "load_config", lambda: cfg)

        class FakePublic:
            def __init__(self, base_url, **kwargs):
                self.base_url = base_url

            def fetch_klines(self, symbol, interval="1h", limit=250):
                df = pd.DataFrame(
                    [{
                        "ts": now,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 1.0,
                    }]
                )
                df.attrs["symbol"] = symbol
                return df

            def get_ticker_price(self, symbol):
                return price * 1.0005

            def get_lot_size(self, symbol):
                return {"minQty": 1e-05, "maxQty": 1e6, "stepSize": 1e-05, "minNotional": 0.0}

            def fetch_order_book(self, symbol, limit=100):
                return {
                    "lastUpdateId": 1,
                    "bids": [[str(price * 0.999), "1.0"], [str(price * 0.995), "5.0"]],
                    "asks": [[str(price * 1.001), "1.0"], [str(price * 1.005), "5.0"]],
                }

        def fake_features(df, cfg):
            symbol = df.attrs["symbol"]
            return pd.DataFrame(
                [{
                    "ts": now,
                    "close": price,
                    "atr": price * 0.01,
                    "atrp": 0.012,
                    "adx": 40.0,
                    "trend_ma": price * 0.95,
                    "high": price,
                    "low": price,
                    "ema_slow": price * 0.95,
                    "regime": Regime.TREND_UP,
                    "symbol": symbol,
                }]
            )

        signal = Signal(
            ts=now,
            symbol="BTCJPY",
            side=Side.BUY,
            entry_price=price,
            stop_price=price * 0.98,
        )

        monkeypatch.setattr(main_live, "PublicMarketClient", FakePublic)
        monkeypatch.setattr(main_live, "compute_features_1h", fake_features)
        monkeypatch.setattr(main_live, "compute_features_5m", fake_features)
        monkeypatch.setattr(main_live, "classify_regime", lambda row, cfg: Regime.TREND_UP)
        monkeypatch.setattr(main_live, "generate_signals", lambda *a, **kw: [signal])
        monkeypatch.setattr(main_live.clock, "now", lambda: now)
        monkeypatch.setattr(
            main_live,
            "time",
            types.SimpleNamespace(
                sleep=lambda _s: (_ for _ in ()).throw(KeyboardInterrupt())
            ),
        )

        with pytest.raises(KeyboardInterrupt):
            main_live.main()  # sleep を差し替えて 1 回転で抜ける（状態は回転末で保存済み）

        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.held_symbols == ["BTCJPY"]

        journal = Path(cfg["persistence"]["signal_journal_path"])
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        kinds = {(row.get("event"), row.get("kind")) for row in rows}
        assert ("SIGNAL", None) in kinds
        assert ("FILL", "ENTRY") in kinds
        assert ("STOP", "PLACED") in kinds

        # 実板ペーパーの全レコードに venue が付く（実約定と区別できる）
        assert all(row.get("venue") == "paper_mock" for row in rows)

        entry = next(r for r in rows if r.get("event") == "FILL" and r["kind"] == "ENTRY")
        # 板の ask（price*1.001）で消費される = シグナル価格より不利
        assert entry["price"] == pytest.approx(price * 1.001)
        assert entry["slippage_bps"] > 0

        account = json.loads(Path(cfg["paper"]["account_file"]).read_text(encoding="utf-8"))
        assert account["quote_balance"] < 100_000.0
        assert account["base_balances"]["BTC"] > 0

    def test_simulated_stop_closes_position_and_is_marked(self, monkeypatch, tmp_path):
        """模擬 STOP 到達 → 模擬約定で決済 → ジャーナルに実 STOP と区別できる形で残る."""
        from datetime import datetime, timezone

        import pandas as pd

        from aivc_trade.core.types import Position, Regime, SystemState
        from aivc_trade.ops.persistence import StateManager

        now = datetime(2026, 11, 20, 3, 1, tzinfo=timezone.utc)
        price = 5_000_000.0
        stop = price * 0.98

        cfg = _paper_cfg(tmp_path)
        cfg["exchange"]["symbols"] = ["BTCJPY"]
        cfg["exchange"]["shadow_symbols"] = []
        cfg["shadow"]["enabled"] = False
        cfg["notification"]["discord_webhook_url"] = ""
        monkeypatch.setattr(main_live, "load_config", lambda: cfg)

        # 建玉を state に仕込む → 起動時 reconcile が模擬 STOP を復元する。
        # 模擬口座側にも同じ建玉を持たせる（起動時の照合は口座を正とする）。
        Path(cfg["paper"]["account_file"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["paper"]["account_file"]).write_text(
            json.dumps(
                {
                    "quote_asset": "JPY",
                    "quote_balance": 95_000.0,
                    "base_balances": {"BTC": 0.001},
                    "realized_pnl": 0.0,
                    "cost_basis": {"BTC": price},
                    "fills": 1,
                }
            ),
            encoding="utf-8",
        )
        StateManager(cfg["persistence"]["state_file"]).save(
            SystemState(
                positions=[
                    Position(
                        symbol="BTCJPY",
                        qty=0.001,
                        entry_price=price,
                        stop_price=stop,
                        initial_qty=0.001,
                        entry_ts=now,
                        atr_at_entry=price * 0.01,
                        highest_price=price,
                        lowest_price=price,
                    )
                ]
            )
        )

        class FakePublic:
            def __init__(self, base_url, **kwargs):
                self.base_url = base_url

            def fetch_klines(self, symbol, interval="1h", limit=250):
                df = pd.DataFrame(
                    [{
                        "ts": now,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 1.0,
                    }]
                )
                df.attrs["symbol"] = symbol
                return df

            def get_ticker_price(self, symbol):
                return price * 0.97  # stop (0.98) を下抜けている

            def get_lot_size(self, symbol):
                return {"minQty": 1e-05, "maxQty": 1e6, "stepSize": 1e-05, "minNotional": 0.0}

            def fetch_order_book(self, symbol, limit=100):
                return {
                    "lastUpdateId": 2,
                    "bids": [[str(price * 0.969), "1.0"], [str(price * 0.965), "5.0"]],
                    "asks": [[str(price * 0.971), "1.0"], [str(price * 0.975), "5.0"]],
                }

        def fake_features(df, cfg):
            symbol = df.attrs["symbol"]
            return pd.DataFrame(
                [{
                    "ts": now,
                    "close": price,
                    "atr": price * 0.01,
                    "atrp": 0.012,
                    "adx": 35.0,
                    "trend_ma": price * 1.05,
                    "high": price,
                    "low": price,
                    "ema_slow": price * 1.05,
                    "regime": Regime.RANGE,
                    "symbol": symbol,
                }]
            )

        monkeypatch.setattr(main_live, "PublicMarketClient", FakePublic)
        monkeypatch.setattr(main_live, "compute_features_1h", fake_features)
        monkeypatch.setattr(main_live, "compute_features_5m", fake_features)
        monkeypatch.setattr(main_live, "classify_regime", lambda row, cfg: Regime.RANGE)
        monkeypatch.setattr(main_live, "generate_signals", lambda *a, **kw: [])
        monkeypatch.setattr(main_live.clock, "now", lambda: now)
        monkeypatch.setattr(
            main_live,
            "_sleep_interruptible",
            lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

        with pytest.raises(KeyboardInterrupt):
            main_live.main()

        journal = Path(cfg["persistence"]["signal_journal_path"])
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]

        triggered = [r for r in rows if r.get("event") == "STOP" and r.get("kind") == "TRIGGERED"]
        assert triggered, "simulated stop did not fire"
        assert triggered[0]["stop_kind"] == "PAPER_MOCK"
        assert triggered[0]["venue"] == "paper_mock"
        buffer_pct = float(cfg["stop_order"]["safety_buffer_pct"])
        assert triggered[0]["stop_price"] == pytest.approx(stop * (1 - buffer_pct / 100))

        exits = [r for r in rows if r.get("event") == "FILL" and r.get("kind") == "EXIT"]
        assert exits, "stop did not close the position"
        assert exits[0]["reason"] == "STOP_LOSS"
        assert exits[0]["stop_kind"] == "PAPER_MOCK"
        # 実板の bid で約定している（ticker そのままではない）
        assert exits[0]["price"] == pytest.approx(price * 0.969)

        assert StateManager(cfg["persistence"]["state_file"]).load().positions == []
