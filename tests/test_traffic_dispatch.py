from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from portfolio_ops.api import JsonApplication
from portfolio_ops.clock import FrozenClock
from portfolio_ops.errors import Conflict, Forbidden, InvalidState, ValidationFailed
from portfolio_ops.planning import AllocationRequest, RiskPoint, allocate_capacity, expected_arrival, latest_streak
from portfolio_ops.service import CollectionLogisticsService
from portfolio_ops.risk import DemandBucket, inventory_coverage, mark_to_risk, traffic_gap


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_uses_first_close_as_base(self) -> None:
        streak = latest_streak([
            RiskPoint("2026-09-18", Decimal("108")),
            RiskPoint("2026-09-19", Decimal("105")),
            RiskPoint("2026-09-20", Decimal("102")),
            RiskPoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 4)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_close, Decimal("98"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["dispatch_id"], "first")
        self.assertEqual(rows[0]["allocated_units"], "70.000")
        self.assertEqual(rows[1]["allocated_units"], "30.000")

    def test_inventory_coverage_and_traffic_gap(self) -> None:
        coverage = inventory_coverage(
            [{"center_id": "receiving-vault", "preservation_resource_kind": "tow-truck", "available_units": "250"}],
            [DemandBucket("receiving-vault", "tow-truck", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = traffic_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["traffic_gap"], "30.000")

    def test_mark_to_risk_groups_deterministically(self) -> None:
        result = mark_to_risk(
            [{"position_id": "p1", "risk_index": "HUMIDITY", "quantity_units": "100", "baseline_value": "105"}],
            {"HUMIDITY": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_cny"], "-700.00")

    def test_expected_arrival_counts_minutes_across_midnight(self) -> None:
        start = datetime(2026, 9, 24, 23, 50, tzinfo=timezone.utc)
        self.assertEqual(
            expected_arrival(start, 36),
            datetime(2026, 9, 25, 0, 26, tzinfo=timezone.utc),
        )

    def test_expected_arrival_requires_timezone_and_valid_minutes(self) -> None:
        start = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            expected_arrival(datetime(2026, 9, 24, 8, 0), 36)
        with self.assertRaises(ValueError):
            expected_arrival(start, 0)
        with self.assertRaises(ValueError):
            expected_arrival(start, -30)
        with self.assertRaises(ValueError):
            expected_arrival(start, 24 * 60 + 1)


class CollectionLogisticsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = CollectionLogisticsService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"center_id": "collection-east", "name": "北部实验样本事件保藏中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_units": "500000"})
        self.service.create_facility("plan", {"center_id": "receiving-vault-b", "name": "沿海终端", "kind": "receiving-vault", "timezone": "Asia/Shanghai", "capacity_units": "800000"})
        self.service.create_route("plan", {"corridor_id": "transfer-east-1", "origin_center_id": "collection-east", "destination_center_id": "receiving-vault-b", "preservation_resource_kind": "preservation-box", "hourly_capacity": "100000", "delay_basis_points": 25, "response_minutes": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def risk_record(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_risk_record("plan", {"risk_index": "HUMIDITY", "duty_date": f"2026-09-{day}", "index_value": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_risk_record_revisions_preserve_history(self) -> None:
        first = self.risk_record(23, "98")
        second = self.service.record_risk_record("plan", {"risk_index": "HUMIDITY", "duty_date": "2026-09-23", "index_value": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["risk_record_id"], second["risk_record_id"])
        rows = self.connection.execute("SELECT * FROM risk_index_risk_records ORDER BY risk_record_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_risk_record_id"], rows[0]["risk_record_id"])

    def test_dispatch_request_replay_and_payload_conflict(self) -> None:
        payload = {"dispatch_id": "nom-1", "corridor_id": "transfer-east-1", "specimen_event_id": "herbarium-room", "duty_date": "2026-09-25", "requested_units": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_dispatch("dispatch", payload)
        self.assertEqual(first, self.service.submit_dispatch("dispatch", payload))
        changed = dict(payload, requested_units="81000")
        with self.assertRaises(Conflict):
            self.service.submit_dispatch("dispatch", changed)

    def test_outage_reduces_allocation_and_deployment_consumes_inventory(self) -> None:
        self.service.announce_restriction("risk", "transfer-east-1", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_dispatch("dispatch", {"dispatch_id": f"nom-{number}", "corridor_id": "transfer-east-1", "specimen_event_id": f"specimen_event-{number}", "duty_date": "2026-09-25", "requested_units": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "transfer-east-1", "2026-09-25")
        self.assertEqual(allocation["available_units"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_units"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["deployed_units"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_units"], "20000.000")

    def prepare_allocated_dispatch(self) -> None:
        self.service.submit_dispatch("dispatch", {"dispatch_id": "nom-1", "corridor_id": "transfer-east-1", "specimen_event_id": "herbarium-room", "duty_date": "2026-09-25", "requested_units": "40000", "priority": 10, "idempotency_key": "key-1"})
        self.service.allocate("dispatch", "transfer-east-1", "2026-09-25")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})

    def test_expected_arrival_uses_registered_minutes(self) -> None:
        self.prepare_allocated_dispatch()
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["response_minutes"], 36)
        self.assertEqual(deployment["expected_arrival"], "2026-09-24T08:36:00Z")

    def test_expected_arrival_crosses_day_boundary(self) -> None:
        self.prepare_allocated_dispatch()
        self.clock.advance(hours=15, minutes=50)
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["expected_arrival"], "2026-09-25T00:26:00Z")

    def test_response_minutes_write_boundaries(self) -> None:
        base = {"origin_center_id": "collection-east", "destination_center_id": "receiving-vault-b", "preservation_resource_kind": "preservation-box", "hourly_capacity": "1000", "delay_basis_points": 0}
        for bad in (0, -5, 1441, "45", 45.0, True, None):
            with self.assertRaises(ValidationFailed, msg=repr(bad)):
                self.service.create_route("plan", {**base, "corridor_id": "route-bad", "response_minutes": bad})
        self.assertEqual(self.service.create_route("plan", {**base, "corridor_id": "route-min", "response_minutes": 1})["response_minutes"], 1)
        self.assertEqual(self.service.create_route("plan", {**base, "corridor_id": "route-max", "response_minutes": 1440})["response_minutes"], 1440)

    def test_route_view_and_audit_share_minutes_semantics(self) -> None:
        view = self.service.route("transfer-east-1")
        self.assertEqual(view["response_minutes"], 36)
        self.assertEqual(view["response_duration_state"], "minutes")
        row = self.connection.execute(
            "SELECT payload_json FROM traffic_audit_events WHERE event_type='route.created'"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["response_minutes"], 36)
        self.assertEqual(payload["response_duration_state"], "minutes")

    def test_deployment_audit_records_minutes_and_arrival(self) -> None:
        self.prepare_allocated_dispatch()
        self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        row = self.connection.execute(
            "SELECT payload_json FROM traffic_audit_events WHERE event_type='deployment.dispatched'"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["response_minutes"], 36)
        self.assertEqual(payload["expected_arrival"], "2026-09-24T08:36:00Z")

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.risk_record(23, "98")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "库房环境恢复", "risk_index_drop_percent": "9", "route_capacity_changes": {"transfer-east-1": "20"}, "demand_changes": {"collection-east:preservation-box": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE traffic_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/risk_records/summary/HUMIDITY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")


class LegacyDurationTests(unittest.TestCase):
    """旧版库表中响应时长单位的迁移、标记与阻断。"""

    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE road_corridors (
                corridor_id TEXT PRIMARY KEY,
                origin_center_id TEXT NOT NULL,
                destination_center_id TEXT NOT NULL,
                preservation_resource_kind TEXT NOT NULL,
                hourly_capacity TEXT NOT NULL,
                delay_basis_points INTEGER NOT NULL,
                response_hours INTEGER,
                revision INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "INSERT INTO road_corridors(corridor_id,origin_center_id,destination_center_id,preservation_resource_kind,"
            "hourly_capacity,delay_basis_points,response_hours,created_at) "
            "VALUES('legacy-known','collection-east','receiving-vault-b','preservation-box','1000',0,2,'2026-09-20T00:00:00Z')"
        )
        self.connection.execute(
            "INSERT INTO road_corridors(corridor_id,origin_center_id,destination_center_id,preservation_resource_kind,"
            "hourly_capacity,delay_basis_points,response_hours,created_at) "
            "VALUES('legacy-unknown','collection-east','receiving-vault-b','preservation-box','1000',0,NULL,'2026-09-20T00:00:00Z')"
        )
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = CollectionLogisticsService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"center_id": "collection-east", "name": "北部实验样本事件保藏中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_units": "500000"})
        self.service.create_facility("plan", {"center_id": "receiving-vault-b", "name": "沿海终端", "kind": "receiving-vault", "timezone": "Asia/Shanghai", "capacity_units": "800000"})

    def tearDown(self) -> None:
        self.connection.close()

    def deploy(self, corridor_id: str) -> dict[str, object]:
        self.service.submit_dispatch("dispatch", {"dispatch_id": f"nom-{corridor_id}", "corridor_id": corridor_id, "specimen_event_id": "herbarium-room", "duty_date": "2026-09-25", "requested_units": "100", "priority": 10, "idempotency_key": f"key-{corridor_id}"})
        self.service.allocate("dispatch", corridor_id, "2026-09-25")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": f"lot-{corridor_id}", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "500", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        return self.service.dispatch_deployment("dispatch", f"deployment-{corridor_id}", f"nom-{corridor_id}", f"lot-{corridor_id}", 2)

    def test_identifiable_hours_backfilled_to_minutes(self) -> None:
        view = self.service.route("legacy-known")
        self.assertEqual(view["response_minutes"], 120)
        self.assertEqual(view["response_duration_state"], "minutes")
        deployment = self.deploy("legacy-known")
        self.assertEqual(deployment["response_minutes"], 120)
        self.assertEqual(deployment["expected_arrival"], "2026-09-24T10:00:00Z")

    def test_unidentifiable_legacy_duration_flagged_and_blocked(self) -> None:
        view = self.service.route("legacy-unknown")
        self.assertIsNone(view["response_minutes"])
        self.assertEqual(view["response_duration_state"], "ambiguous")
        with self.assertRaises(InvalidState):
            self.deploy("legacy-unknown")
        lot = self.service.inventory_lot("lot-legacy-unknown")
        self.assertEqual(lot["available_units"], "500")


if __name__ == "__main__":
    unittest.main()
