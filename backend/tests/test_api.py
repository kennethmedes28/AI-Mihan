"""Tests for the HTTP API. A fake chat graph and a fake database are used: no network, model or real database.

Run from the backend folder:  python -m unittest tests.test_api -v
"""

import os
import sys
import types
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))  # optional search library, not needed here

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.api import startup
from src.api.app import create_app
from src.api.rate_limit import RateLimiter
from src.api.routers import advisories as advisories_router
from src.api.routers import evacuation as evacuation_router
from src.api.routers import help_requests as requests_router
from src.api.routers import reports as reports_router
from src.api.routers import staff as staff_router
from src.api.settings import Settings, load_settings

MARIKINA_POINT = {"lat": 14.6507, "lng": 121.1029}
TANZA_POINT = {"lat": 14.3925, "lng": 120.8497}  # Cavite, outside Metro Manila
LONG_BULLETIN = "Rainfall warning bulletin for Metro Manila. " * 12  # long enough to count as a real bulletin


class FakeGraph:
    """Stands in for the compiled chat graph: records what it was given and returns a fixed reply."""

    def __init__(self, text="the reply", safe_answer=False, error=None):
        self.text, self.safe_answer, self.error = text, safe_answer, error
        self.inputs, self.configs = [], []

    async def ainvoke(self, graph_input, config):
        self.inputs.append(graph_input)
        self.configs.append(config)
        if self.error:
            raise self.error
        return {"messages": [SimpleNamespace(content=self.text)], "safe_answer": self.safe_answer}


class FakeConnection:
    """Stands in for a database connection; ``execute`` only has to succeed or raise."""

    def __init__(self, error=None):
        self.error = error

    def execute(self, sql, values=None):
        if self.error:
            raise self.error


def make_client(graph=None, connection=None, settings=None) -> TestClient:
    """A client for an app whose services are fakes. Use it in a ``with`` block so the lifespan runs."""
    graph = graph if graph is not None else FakeGraph()
    connection = connection if connection is not None else FakeConnection()

    @asynccontextmanager
    async def fake_services(app):
        app.state.graph = graph
        app.state.get_connection = lambda: connection
        app.state.llm_ready = True
        app.state.retriever_ready = False
        yield

    return TestClient(create_app(settings or Settings(), lifespan=fake_services))


class ChatTests(unittest.TestCase):
    def test_new_conversation_gets_a_thread_id(self):
        graph = FakeGraph(text="Hello")
        with make_client(graph) as client:
            response = client.post("/chat", json={"message": "hi"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        UUID(body["thread_id"])  # raises if it is not a UUID
        self.assertEqual((body["reply"], body["safe_answer"]), ("Hello", False))
        self.assertEqual(graph.inputs[0]["messages"], [("user", "hi")])
        self.assertEqual(graph.configs[0]["configurable"]["thread_id"], body["thread_id"])

    def test_given_thread_id_is_kept(self):
        thread_id = "0b6f0c1e-6f7e-4d6a-9b1a-3f5f2f6d7a10"
        with make_client() as client:
            response = client.post("/chat", json={"message": "hi", "thread_id": thread_id})
        self.assertEqual(response.json()["thread_id"], thread_id)

    def test_gps_location_is_passed_to_the_graph(self):
        graph = FakeGraph()
        with make_client(graph) as client:
            client.post("/chat", json={"message": "saan ang evacuation center?", "location": MARIKINA_POINT})
        self.assertEqual(graph.inputs[0]["location"]["city"], "marikina")

    def test_safe_answer_flag_is_returned(self):
        with make_client(FakeGraph(safe_answer=True)) as client:
            self.assertTrue(client.post("/chat", json={"message": "hi"}).json()["safe_answer"])

    def test_bad_input_is_rejected(self):
        bad_bodies = [
            {"message": "   "},
            {"message": "x" * 1001},
            {"message": "hi", "thread_id": "not-a-uuid"},
            {"message": "hi", "unknown_field": 1},
        ]
        with make_client() as client:
            for body in bad_bodies:
                self.assertEqual(client.post("/chat", json=body).status_code, 422, body)

            # Strict JSON has no NaN, so the test client cannot send one; a hand-written client can.
            nan_body = '{"message": "hi", "location": {"lat": NaN, "lng": 121.0}}'
            response = client.post("/chat", content=nan_body, headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)

    def test_location_outside_metro_manila_is_rejected_with_safety_text(self):
        with make_client() as client:
            response = client.post("/chat", json={"message": "hi", "location": TANZA_POINT})
        self.assertEqual(response.status_code, 422)
        self.assertIn("911", response.text)

    def test_graph_failure_gives_a_safe_error_without_details(self):
        graph = FakeGraph(error=RuntimeError("password=hunter2 in the stack"))
        with make_client(graph) as client:
            response = client.post("/chat", json={"message": "hi"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("911", response.text)
        self.assertNotIn("hunter2", response.text)

    def test_rate_limit(self):
        with make_client(settings=Settings(chat_per_minute=2)) as client:
            statuses = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(3)]
            blocked = client.post("/chat", json={"message": "hi"})
        self.assertEqual(statuses, [200, 200, 429])
        self.assertIn("Retry-After", blocked.headers)
        self.assertIn("911", blocked.text)


class ValidationErrorTests(unittest.TestCase):
    def test_error_does_not_echo_what_was_sent(self):
        payload = {"kind": "supply", "city": "Marikina", "details": "Water", "contact_phone": "call me maybe"}
        with make_client() as client:
            response = client.post("/requests", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("call me maybe", response.text)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "contact_phone"])


class ReportTests(unittest.TestCase):
    def test_city_name_is_turned_into_the_city_key(self):
        with make_client() as client, mock.patch.object(reports_router, "insert_report") as insert:
            response = client.post("/reports", json={"city": "Parañaque City", "body": "Baha sa kanto"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "unverified")
        arguments = insert.call_args.kwargs
        self.assertEqual((arguments["city"], arguments["lat"], arguments["lng"]), ("paranaque", None, None))

    def test_gps_point_without_city_uses_the_nearest_city_and_keeps_the_point(self):
        payload = {"location": MARIKINA_POINT, "barangay": "Nangka", "body": "Knee-deep water"}
        with make_client() as client, mock.patch.object(reports_router, "insert_report") as insert:
            client.post("/reports", json=payload)
        arguments = insert.call_args.kwargs
        self.assertEqual((arguments["city"], arguments["barangay"]), ("marikina", "Nangka"))
        self.assertEqual((arguments["lat"], arguments["lng"]), (14.6507, 121.1029))

    def test_bad_reports_are_rejected(self):
        bad_bodies = [
            {"city": "Tanza", "body": "Baha"},  # not a Metro Manila city
            {"body": "Baha sa kanto"},  # no place at all
            {"city": "Pasig", "body": "ab"},  # too short
            {"city": "Pasig", "body": "Baha", "hazard": "earthquake"},
            {"location": TANZA_POINT, "body": "Baha"},  # outside Metro Manila
        ]
        with make_client() as client, mock.patch.object(reports_router, "insert_report") as insert:
            for body in bad_bodies:
                self.assertEqual(client.post("/reports", json=body).status_code, 422, body)
        insert.assert_not_called()

    def test_database_failure_gives_503_without_the_error_text(self):
        with make_client() as client, mock.patch.object(
            reports_router, "insert_report", side_effect=RuntimeError('relation "reports" does not exist')
        ):
            response = client.post("/reports", json={"city": "Pasig", "body": "Baha"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("relation", response.text)
        self.assertIn("911", response.text)

    def test_listing_hides_rejected_reports_and_phone_numbers(self):
        rows = [
            {"hazard": "flood", "city": "pasig", "barangay": None, "body": "Call 0917 123 4567 for help",
             "status": "unverified", "trust_level": "community", "created_at": "2026-10-30T10:00:00+08:00"},
            {"hazard": "flood", "city": "pasig", "barangay": None, "body": "spam",
             "status": "rejected", "trust_level": "community", "created_at": "2026-10-30T09:00:00+08:00"},
        ]
        with make_client() as client, mock.patch.object(reports_router, "fetch_recent_reports", return_value=rows) as fetch:
            response = client.get("/reports", params={"city": "Pasig", "limit": 5})
        reports = response.json()
        self.assertEqual(len(reports), 1)
        self.assertNotIn("0917", reports[0]["body"])
        self.assertIn("[number]", reports[0]["body"])
        self.assertEqual(fetch.call_args.kwargs["city"], "pasig")

    def test_listing_checks_its_parameters(self):
        with make_client() as client:
            self.assertEqual(client.get("/reports", params={"city": "Tanza"}).status_code, 422)
            self.assertEqual(client.get("/reports", params={"limit": 51}).status_code, 422)
            self.assertEqual(client.get("/reports", params={"hazard": "fire"}).status_code, 422)


class RequestTests(unittest.TestCase):
    PAYLOAD = {"kind": "supply", "city": "Marikina", "details": "Water and medicine for 12 families",
               "people_count": 40, "contact_phone": "+63 917 123 4567"}

    def test_request_is_saved_unverified_and_the_phone_is_not_echoed(self):
        with make_client() as client, mock.patch.object(requests_router, "insert_request") as insert:
            response = client.post("/requests", json=self.PAYLOAD)
        self.assertEqual(response.status_code, 201)
        self.assertNotIn("917", response.text)
        self.assertIn("911", response.json()["message"])
        self.assertEqual(insert.call_args.kwargs["city"], "marikina")
        self.assertEqual(insert.call_args.kwargs["contact_phone"], "+63 917 123 4567")

    def test_bad_requests_are_rejected(self):
        bad_changes = [
            {"kind": "weapons"},
            {"people_count": 0},
            {"contact_phone": "call me"},
            {"city": None},
        ]
        with make_client() as client, mock.patch.object(requests_router, "insert_request") as insert:
            for change in bad_changes:
                self.assertEqual(client.post("/requests", json={**self.PAYLOAD, **change}).status_code, 422, change)
        insert.assert_not_called()

    def test_requests_cannot_be_read_back(self):
        with make_client() as client:
            self.assertEqual(client.get("/requests").status_code, 405)


STAFF_KEY = "staff-key-for-tests-0123456789abcdef"
STAFF_HEADERS = {"Authorization": f"Bearer {STAFF_KEY}"}
REPORT_ID = "11111111-2222-3333-4444-555555555555"


def make_staff_client(**settings) -> TestClient:
    """A client for an app that has the staff key set, so the /staff endpoints exist."""
    return make_client(settings=Settings(staff_api_key=STAFF_KEY, **settings))


class StaffAccessTests(unittest.TestCase):
    def test_staff_endpoints_do_not_exist_without_a_key(self):
        with make_client() as client:  # no staff key configured
            self.assertEqual(client.get("/staff/summary", headers=STAFF_HEADERS).status_code, 404)
            self.assertEqual(client.get("/staff/requests", headers=STAFF_HEADERS).status_code, 404)

    def test_missing_wrong_and_malformed_keys_are_refused_the_same_way(self):
        attempts = [
            {},
            {"Authorization": "Bearer wrong-key-wrong-key-wrong-key-12"},
            {"Authorization": STAFF_KEY},  # no "Bearer"
            {"Authorization": f"Basic {STAFF_KEY}"},
            {"Authorization": "Bearer "},
        ]
        with make_staff_client() as client:
            responses = [client.get("/staff/summary", headers=headers) for headers in attempts]
        for response in responses:
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual({response.text for response in responses}, {responses[0].text})  # nothing to tell them apart

    def test_nothing_is_read_from_the_database_without_the_key(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "fetch_requests_for_staff") as fetch:
            client.get("/staff/requests")
        fetch.assert_not_called()

    def test_repeated_calls_are_rate_limited_even_with_wrong_keys(self):
        with make_staff_client(staff_per_minute=2) as client:
            statuses = [client.get("/staff/summary", headers={"Authorization": "Bearer nope"}).status_code for _ in range(3)]
        self.assertEqual(statuses, [401, 401, 429])

    def test_a_short_key_stops_start_up(self):
        with mock.patch("src.api.settings.load_dotenv"), mock.patch.dict(os.environ, {"STAFF_API_KEY": "short"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "at least 24"):
                load_settings()

    def test_public_endpoints_do_not_expose_staff_data(self):
        with make_staff_client() as client:
            self.assertEqual(client.get("/requests").status_code, 405)  # still write only
            self.assertEqual(client.get("/reports", headers=STAFF_HEADERS).status_code, 503)  # public list, not the staff one


class StaffReviewTests(unittest.TestCase):
    REPORT = {"id": REPORT_ID, "hazard": "flood", "city": "pasig", "barangay": "Kapitolyo",
              "body": "Call 0917 123 4567, water is waist-deep", "status": "unverified", "trust_level": "community",
              "lat": 14.5764, "lng": 121.0851, "created_at": "2026-10-30T10:00:00+08:00"}
    REQUEST = {"id": "99999999-8888-7777-6666-555555555555", "kind": "supply", "city": "marikina", "barangay": None,
               "details": "Water for 12 families", "people_count": 40, "contact_phone": "+63 917 123 4567",
               "lat": None, "lng": None, "status": "unverified", "created_at": "2026-10-30T10:05:00+08:00"}

    def test_staff_see_the_full_report_unredacted(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "fetch_reports_for_staff", return_value=[self.REPORT]) as fetch:
            response = client.get("/staff/reports", params={"status": "unverified", "city": "Pasig", "limit": 5}, headers=STAFF_HEADERS)
        report = response.json()[0]
        self.assertIn("0917 123 4567", report["body"])  # the public list hides this; staff need it
        self.assertEqual((report["id"], report["lat"]), (REPORT_ID, 14.5764))
        self.assertEqual((fetch.call_args.kwargs["status"], fetch.call_args.kwargs["city"]), ("unverified", "pasig"))

    def test_staff_see_request_contact_numbers(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "fetch_requests_for_staff", return_value=[self.REQUEST]):
            response = client.get("/staff/requests", headers=STAFF_HEADERS)
        self.assertEqual(response.json()[0]["contact_phone"], "+63 917 123 4567")

    def test_filters_are_checked(self):
        with make_staff_client() as client:
            for path, params in (
                ("/staff/reports", {"status": "assigned"}),  # only requests can be assigned
                ("/staff/requests", {"status": "rejected"}),  # only reports can be rejected
                ("/staff/reports", {"city": "Tanza"}),
                ("/staff/requests", {"kind": "weapons"}),
                ("/staff/reports", {"limit": 51}),
            ):
                self.assertEqual(client.get(path, params=params, headers=STAFF_HEADERS).status_code, 422, (path, params))

    def test_status_change(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "update_status", return_value=True) as update:
            response = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "verified"}, headers=STAFF_HEADERS)
            assigned = client.patch(f"/staff/requests/{REPORT_ID}", json={"status": "assigned"}, headers=STAFF_HEADERS)
        self.assertEqual(response.json(), {"id": REPORT_ID, "status": "verified"})
        self.assertEqual(assigned.status_code, 200)
        self.assertEqual([call.args[1] for call in update.call_args_list], ["reports", "requests"])

    def test_unknown_id_is_404_and_bad_input_is_422(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "update_status", return_value=False) as update:
            missing = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "rejected"}, headers=STAFF_HEADERS)
            bad_status = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "assigned"}, headers=STAFF_HEADERS)
            bad_id = client.patch("/staff/reports/not-a-uuid", json={"status": "verified"}, headers=STAFF_HEADERS)
            extra = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "verified", "body": "edited"}, headers=STAFF_HEADERS)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual((bad_status.status_code, bad_id.status_code, extra.status_code), (422, 422, 422))
        update.assert_called_once()  # only the first call got as far as the database

    def test_status_change_needs_the_key(self):
        with make_staff_client() as client, mock.patch.object(staff_router, "update_status") as update:
            response = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "verified"})
        self.assertEqual(response.status_code, 401)
        update.assert_not_called()

    def test_summary(self):
        counts = {"reports": {"total": 3, "by_status": {"unverified": 2, "verified": 1}, "by_city": {"pasig": 3},
                              "oldest_unverified_at": "2026-10-30T08:00:00+08:00"},
                  "requests": {"total": 0, "by_status": {}, "by_city": {}, "oldest_unverified_at": None}}
        with make_staff_client() as client, mock.patch.object(staff_router, "summarize_table", side_effect=lambda conn, table: counts[table]):
            body = client.get("/staff/summary", headers=STAFF_HEADERS).json()
        self.assertEqual(body["reports"]["by_status"], {"unverified": 2, "verified": 1})
        self.assertIsNone(body["requests"]["oldest_unverified_at"])

    def test_database_failure_gives_503_without_details(self):
        with make_staff_client() as client, mock.patch.object(
            staff_router, "update_status", side_effect=RuntimeError('relation "reports" does not exist')
        ):
            response = client.patch(f"/staff/reports/{REPORT_ID}", json={"status": "verified"}, headers=STAFF_HEADERS)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("relation", response.text)


class StaffQueryTests(unittest.TestCase):
    """The SQL helpers, with a recording connection (no real database)."""

    class Recorder:
        def __init__(self, rows=None, rowcount=1):
            self.calls, self.rows, self.rowcount = [], rows or [], rowcount

        def execute(self, sql, values=None):
            self.calls.append((sql, values))
            return self

        def fetchall(self):
            return self.rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

    def test_only_fixed_column_names_reach_the_sql_and_values_are_parameters(self):
        from src.services import fetch_postgre_database as database

        recorder = self.Recorder()
        database.fetch_reports_for_staff(recorder, status="x'; DROP TABLE reports; --", city="pasig", limit=500)
        sql, values = recorder.calls[0]
        self.assertNotIn("DROP", sql)
        self.assertIn("status ILIKE %s", sql)
        self.assertEqual(values[0], "x'; DROP TABLE reports; --")
        self.assertTrue(sql.endswith("LIMIT 50"))  # never more than 50 rows

    def test_update_and_summary_refuse_other_tables(self):
        from src.services import fetch_postgre_database as database

        for call in (lambda: database.update_status(self.Recorder(), "users", REPORT_ID, "verified"),
                     lambda: database.summarize_table(self.Recorder(), "ncr_locations")):
            with self.assertRaises(ValueError):
                call()

    def test_update_status_reports_whether_a_row_changed(self):
        from src.services import fetch_postgre_database as database

        self.assertTrue(database.update_status(self.Recorder(rowcount=1), "reports", REPORT_ID, "verified"))
        self.assertFalse(database.update_status(self.Recorder(rowcount=0), "reports", REPORT_ID, "verified"))


class EvacuationCenterTests(unittest.TestCase):
    ROW = {"name": "Near Gym", "city": "Marikina City", "barangay": "Nangka", "type": "gym",
           "lat": 14.65, "lng": 121.10, "notes": None}

    def test_search_by_city_uses_the_city_key(self):
        with make_client() as client, mock.patch.object(
            evacuation_router, "fetch_evacuation_centers", return_value=[self.ROW]
        ) as fetch:
            response = client.get("/evacuation-centers", params={"city": "Parañaque", "limit": 5})
        self.assertEqual(response.json()[0]["name"], "Near Gym")
        self.assertEqual(fetch.call_args.kwargs["city"], "paranaque")
        self.assertEqual(fetch.call_args.kwargs["limit"], 5)

    def test_bad_parameters_are_rejected(self):
        with make_client() as client, mock.patch.object(evacuation_router, "fetch_evacuation_centers") as fetch:
            self.assertEqual(client.get("/evacuation-centers", params={"city": "Tanza"}).status_code, 422)
            self.assertEqual(client.get("/evacuation-centers", params={"barangay": "%"}).status_code, 422)
            self.assertEqual(client.get("/evacuation-centers", params={"barangay": "a_b"}).status_code, 422)
            self.assertEqual(client.get("/evacuation-centers", params={"limit": 0}).status_code, 422)
        fetch.assert_not_called()

    def test_database_failure_gives_503(self):
        with make_client() as client, mock.patch.object(
            evacuation_router, "fetch_evacuation_centers", side_effect=RuntimeError("down")
        ):
            self.assertEqual(client.get("/evacuation-centers").status_code, 503)


class AdvisoryTests(unittest.TestCase):
    def page(self, text=LONG_BULLETIN):
        return {"page": "advisory", "source": "https://www.pagasa.dost.gov.ph/weather/weather-advisory",
                "title": "Weather Advisory", "text": text}

    def get(self, pages=None, error=None, params=None):
        with make_client() as client, mock.patch.object(
            advisories_router, "fetch_weather_pages", return_value=pages, side_effect=error
        ):
            return client.get("/advisories/latest", params=params)

    def test_reliable_pages_are_returned_shortened(self):
        body = self.get([self.page("Marikina " + "x" * 5000)] if False else [self.page(LONG_BULLETIN * 20)]).json()
        self.assertTrue(body["available"])
        self.assertLessEqual(len(body["advisories"][0]["excerpt"]), advisories_router.EXCERPT_CHARS)
        self.assertIsNone(body["advisories"][0]["mentions_city"])

    def test_city_marks_which_pages_mention_it(self):
        pages = [self.page(LONG_BULLETIN + " Heavy rain over Marikina."), self.page(LONG_BULLETIN)]
        body = self.get(pages, params={"city": "Marikina"}).json()
        self.assertEqual([item["mentions_city"] for item in body["advisories"]], [True, False])

    def test_block_page_empty_and_failure_all_say_unavailable(self):
        block_page = self.page("Just a moment... " * 30)
        for pages, error in (([block_page], None), ([], None), (None, RuntimeError("down"))):
            body = self.get(pages, error).json()
            self.assertFalse(body["available"])
            self.assertEqual(body["advisories"], [])
            self.assertIn("pagasa.dost.gov.ph", body["note"])

    def test_unknown_city_is_rejected(self):
        self.assertEqual(self.get([self.page()], params={"city": "Tanza"}).status_code, 422)


class HealthTests(unittest.TestCase):
    def test_everything_works(self):
        with make_client() as client:
            body = client.get("/health").json()
            self.assertEqual(client.get("/health/ready").status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checks"]["database"], "ok")

    def test_database_down_is_degraded_and_not_ready(self):
        with make_client(connection=FakeConnection(error=RuntimeError("host=secret.example.com"))) as client:
            health = client.get("/health")
            ready = client.get("/health/ready")
        self.assertEqual(health.status_code, 200)  # the process is alive
        self.assertEqual(health.json()["status"], "degraded")
        self.assertEqual(ready.status_code, 503)
        self.assertNotIn("secret.example.com", health.text + ready.text)


class CorsTests(unittest.TestCase):
    def preflight(self, origin):
        with make_client(settings=Settings(cors_origins=("https://app.example.org",))) as client:
            return client.options(
                "/chat",
                headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
            )

    def test_listed_origin_is_allowed(self):
        self.assertEqual(self.preflight("https://app.example.org").headers.get("access-control-allow-origin"), "https://app.example.org")

    def test_other_origin_is_not(self):
        self.assertIsNone(self.preflight("https://evil.example.com").headers.get("access-control-allow-origin"))


class RateLimiterTests(unittest.TestCase):
    def test_blocks_after_the_limit_and_frees_up_later(self):
        now = [0.0]
        limiter = RateLimiter(max_calls=2, window_seconds=60, clock=lambda: now[0])
        self.assertIsNone(limiter.retry_after("a"))
        self.assertIsNone(limiter.retry_after("a"))
        self.assertEqual(limiter.retry_after("a"), 60)
        now[0] = 59
        self.assertEqual(limiter.retry_after("a"), 1)  # a turned-away call is not counted
        now[0] = 61
        self.assertIsNone(limiter.retry_after("a"))

    def test_clients_are_counted_separately(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60)
        self.assertIsNone(limiter.retry_after("a"))
        self.assertIsNone(limiter.retry_after("b"))
        self.assertIsNotNone(limiter.retry_after("a"))

    def test_idle_clients_are_forgotten(self):
        now = [0.0]
        limiter = RateLimiter(max_calls=1, window_seconds=10, clock=lambda: now[0])
        with mock.patch("src.api.rate_limit.MAX_TRACKED_CLIENTS", 3):
            for number in range(4):
                limiter.retry_after(f"old-{number}")
            now[0] = 100
            limiter.retry_after("new")
        self.assertEqual(list(limiter._calls), ["new"])

    def test_forwarded_header_is_only_trusted_when_enabled(self):
        calls = {"X-Forwarded-For": "203.0.113.9, 10.0.0.1"}
        for trust, expected in ((False, "testclient"), (True, "203.0.113.9")):
            seen = []
            with make_client(settings=Settings(trust_proxy_headers=trust)) as client:
                client.app.state.limiters["chat"].retry_after = lambda who, seen=seen: seen.append(who)
                client.post("/chat", json={"message": "hi"}, headers=calls)
            self.assertEqual(seen, [expected], trust)


class SettingsTests(unittest.TestCase):
    def load(self, **environment):
        with mock.patch("src.api.settings.load_dotenv"), mock.patch.dict(os.environ, environment, clear=True):
            return load_settings()

    def test_defaults(self):
        settings = self.load()
        self.assertEqual(settings.cors_origins, ("http://localhost:3000",))
        self.assertEqual((settings.checkpointer, settings.enable_rag, settings.trust_proxy_headers), ("postgres", True, False))

    def test_values_are_read(self):
        settings = self.load(CORS_ORIGINS="https://a.org, https://b.org", CHECKPOINTER="MEMORY", ENABLE_RAG="no", CHAT_PER_MINUTE="5")
        self.assertEqual(settings.cors_origins, ("https://a.org", "https://b.org"))
        self.assertEqual((settings.checkpointer, settings.enable_rag, settings.chat_per_minute), ("memory", False, 5))

    def test_wrong_checkpointer_stops_start_up(self):
        with self.assertRaises(RuntimeError):
            self.load(CHECKPOINTER="redis")


class StartupTests(unittest.IsolatedAsyncioTestCase):
    def test_without_a_model_key_a_stand_in_is_used(self):
        with mock.patch.object(startup, "build_main_llm", side_effect=RuntimeError("MUSE_API is not set")):
            llm, configured = startup.build_llm()
        self.assertFalse(configured)
        with self.assertRaises(RuntimeError):
            llm.invoke("hello")

    def test_failing_document_search_is_switched_off(self):
        def broken_connection():
            raise RuntimeError("database down")

        self.assertIsNone(startup.build_retriever(broken_connection))

    async def test_memory_checkpointer(self):
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            self.assertIsInstance(await startup.open_checkpointer(stack, "memory"), MemorySaver)

    async def test_missing_postgres_package_gives_a_clear_message(self):
        from contextlib import AsyncExitStack

        # A None entry in sys.modules makes the import fail, as if the package were not installed.
        with mock.patch.dict(sys.modules, {"checkpointer.postgres_checkpointer": None}):
            async with AsyncExitStack() as stack:
                with self.assertRaisesRegex(RuntimeError, "uv sync"):
                    await startup.open_checkpointer(stack, "postgres")


if __name__ == "__main__":
    unittest.main()
