"""响应时长单位语义：分钟为唯一单位，历史记录标记并阻止静默猜测。"""

from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone

from portfolio_ops.api import JsonApplication
from portfolio_ops.clock import FrozenClock
from portfolio_ops.errors import InvalidState, ValidationFailed
from portfolio_ops.models import MAX_RESPONSE_MINUTES
from portfolio_ops.service import CollectionLogisticsService


class ResponseDurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = CollectionLogisticsService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"center_id": "collection-east", "name": "北部实验样本事件保藏中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_units": "500000"})
        self.service.create_facility("plan", {"center_id": "receiving-vault-b", "name": "沿海终端", "kind": "receiving-vault", "timezone": "Asia/Shanghai", "capacity_units": "800000"})

    def tearDown(self) -> None:
        self.connection.close()

    def create_route(self, **overrides: object) -> dict[str, object]:
        payload = {
            "corridor_id": "transfer-1",
            "origin_center_id": "collection-east",
            "destination_center_id": "receiving-vault-b",
            "preservation_resource_kind": "preservation-box",
            "hourly_capacity": "100000",
            "delay_basis_points": 25,
            "response_minutes": 45,
        }
        payload.update(overrides)
        return self.service.create_route("plan", payload)

    def insert_legacy_route(self, corridor_id: str = "legacy-1", response_minutes: int = 45) -> None:
        """模拟修复前写入的旧记录：没有单位标记。"""
        self.connection.execute(
            "INSERT INTO road_corridors(corridor_id,origin_center_id,destination_center_id,preservation_resource_kind,"
            "hourly_capacity,delay_basis_points,response_minutes,response_duration_unit,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (corridor_id, "collection-east", "receiving-vault-b", "preservation-box", "100000", 25, response_minutes, None, "2026-09-01T00:00:00Z"),
        )

    def prepare_allocated_dispatch(self, corridor_id: str) -> None:
        self.service.submit_dispatch("dispatch", {"dispatch_id": "nom-1", "corridor_id": corridor_id, "specimen_event_id": "herbarium-room", "duty_date": "2026-09-25", "requested_units": "40000", "priority": 10, "idempotency_key": "key-1"})
        self.service.allocate("dispatch", corridor_id, "2026-09-25")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})

    def audit_payloads(self, event_type: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT payload_json FROM traffic_audit_events WHERE event_type=? ORDER BY event_id", (event_type,)
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def test_create_route_records_explicit_minutes_unit(self) -> None:
        view = self.create_route()
        self.assertEqual(view["response_minutes"], 45)
        self.assertEqual(view["response_duration_unit"], "minutes")
        self.assertTrue(view["response_duration_verified"])
        payload = self.audit_payloads("route.created")[0]
        self.assertEqual(payload["response_minutes"], 45)
        self.assertEqual(payload["response_duration_unit"], "minutes")

    def test_create_route_rejects_invalid_durations_before_write(self) -> None:
        for bad in (0, -30, MAX_RESPONSE_MINUTES + 1, "45", 45.0, True, None):
            with self.subTest(bad=bad), self.assertRaises(ValidationFailed):
                self.create_route(response_minutes=bad)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) c FROM road_corridors").fetchone()["c"], 0)

    def test_create_route_accepts_boundary_durations(self) -> None:
        self.assertEqual(self.create_route(response_minutes=1)["response_minutes"], 1)
        view = self.create_route(corridor_id="transfer-2", response_minutes=MAX_RESPONSE_MINUTES)
        self.assertEqual(view["response_minutes"], MAX_RESPONSE_MINUTES)

    def test_create_route_rejects_unknown_unit_instead_of_guessing(self) -> None:
        for bad in ("hours", "hour", "seconds", "天", ""):
            with self.subTest(bad=bad), self.assertRaises(ValidationFailed):
                self.create_route(response_duration_unit=bad)
        view = self.create_route(response_duration_unit="minutes")
        self.assertTrue(view["response_duration_verified"])

    def test_expected_arrival_uses_minutes_and_crosses_day_boundary(self) -> None:
        self.create_route()
        self.prepare_allocated_dispatch("transfer-1")
        self.clock.current = datetime(2026, 9, 24, 23, 50, tzinfo=timezone.utc)
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["response_minutes"], 45)
        self.assertEqual(deployment["response_duration_unit"], "minutes")
        # 45 分钟（而非 45 小时）：从 23:50 UTC 跨日到次日 00:35 UTC。
        self.assertEqual(deployment["expected_arrival"], "2026-09-25T00:35:00Z")
        payload = self.audit_payloads("deployment.dispatched")[0]
        self.assertEqual(payload["response_minutes"], 45)
        self.assertEqual(payload["response_duration_unit"], "minutes")
        self.assertEqual(payload["expected_arrival"], "2026-09-25T00:35:00Z")

    def test_legacy_route_is_flagged_and_blocked_from_scheduling(self) -> None:
        self.insert_legacy_route()
        view = self.service.route("legacy-1")
        self.assertEqual(view["response_minutes"], 45)
        self.assertIsNone(view["response_duration_unit"])
        self.assertFalse(view["response_duration_verified"])
        self.prepare_allocated_dispatch("legacy-1")
        with self.assertRaises(InvalidState):
            self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)

    def test_confirm_unit_unblocks_legacy_route(self) -> None:
        self.insert_legacy_route()
        self.prepare_allocated_dispatch("legacy-1")
        with self.assertRaises(ValidationFailed):
            self.service.confirm_route_response_unit("plan", "legacy-1", "hours", 1)
        with self.assertRaises(InvalidState):
            self.service.confirm_route_response_unit("plan", "legacy-1", "minutes", 9)
        view = self.service.confirm_route_response_unit("plan", "legacy-1", "minutes", 1)
        self.assertTrue(view["response_duration_verified"])
        self.assertEqual(view["revision"], 2)
        payload = self.audit_payloads("route.response_unit_confirmed")[0]
        self.assertEqual(payload["response_duration_unit"], "minutes")
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["expected_arrival"], "2026-09-24T08:45:00Z")

    def test_confirm_rejects_out_of_bounds_stored_value(self) -> None:
        self.insert_legacy_route(corridor_id="legacy-huge", response_minutes=MAX_RESPONSE_MINUTES + 1)
        with self.assertRaises(ValidationFailed):
            self.service.confirm_route_response_unit("plan", "legacy-huge", "minutes", 1)

    def test_migration_marks_existing_rows_unverified(self) -> None:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                "CREATE TABLE road_corridors (corridor_id TEXT PRIMARY KEY, origin_center_id TEXT NOT NULL, "
                "destination_center_id TEXT NOT NULL, preservation_resource_kind TEXT NOT NULL, "
                "hourly_capacity TEXT NOT NULL, delay_basis_points INTEGER NOT NULL, "
                "response_minutes INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 1, "
                "state TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO road_corridors(corridor_id,origin_center_id,destination_center_id,preservation_resource_kind,"
                "hourly_capacity,delay_basis_points,response_minutes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                ("legacy-1", "collection-east", "receiving-vault-b", "preservation-box", "100000", 25, 45, "2026-09-01T00:00:00Z"),
            )
            service = CollectionLogisticsService(connection, self.clock)
            view = service.route("legacy-1")
            self.assertIsNone(view["response_duration_unit"])
            self.assertFalse(view["response_duration_verified"])
        finally:
            connection.close()

    def test_api_uses_same_duration_semantics(self) -> None:
        app = JsonApplication(self.service)
        headers = {"X-Actor-Id": "plan"}
        created = app.handle("POST", "/road_corridors", headers, json.dumps({
            "corridor_id": "transfer-1", "origin_center_id": "collection-east", "destination_center_id": "receiving-vault-b",
            "preservation_resource_kind": "preservation-box", "hourly_capacity": "100000",
            "delay_basis_points": 25, "response_minutes": 45,
        }).encode())
        self.assertEqual(created.status, 201)
        self.assertEqual(created.body["response_duration_unit"], "minutes")
        self.assertTrue(created.body["response_duration_verified"])
        shown = app.handle("GET", "/road_corridors/transfer-1", headers)
        self.assertEqual(shown.status, 200)
        self.assertEqual(shown.body["response_minutes"], 45)
        self.assertTrue(shown.body["response_duration_verified"])

        rejected = app.handle("POST", "/road_corridors", headers, json.dumps({
            "corridor_id": "transfer-2", "origin_center_id": "collection-east", "destination_center_id": "receiving-vault-b",
            "preservation_resource_kind": "preservation-box", "hourly_capacity": "100000",
            "delay_basis_points": 25, "response_minutes": 45, "response_duration_unit": "hours",
        }).encode())
        self.assertEqual(rejected.status, 422)

        self.insert_legacy_route()
        legacy = app.handle("GET", "/road_corridors/legacy-1", headers)
        self.assertFalse(legacy.body["response_duration_verified"])
        confirmed = app.handle("POST", "/road_corridors/legacy-1/confirm_response_unit", headers, json.dumps({
            "response_duration_unit": "minutes", "expected_revision": 1,
        }).encode())
        self.assertEqual(confirmed.status, 200)
        self.assertTrue(confirmed.body["response_duration_verified"])


if __name__ == "__main__":
    unittest.main()
