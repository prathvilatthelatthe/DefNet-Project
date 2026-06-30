"""
Comprehensive API endpoint test - finds all bugs in one pass.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app

def test_all():
    app = create_app(testing=True)
    client = app.test_client()
    results = {}

    def test(name, method, url, json_data=None):
        try:
            if method == 'GET':
                r = client.get(url)
            elif method == 'POST':
                r = client.post(url, json=json_data or {}, content_type='application/json')
            elif method == 'PUT':
                r = client.put(url, json=json_data or {}, content_type='application/json')

            data = r.get_json(silent=True)
            status = "PASS" if r.status_code < 400 else f"FAIL({r.status_code})"
            results[name] = {"status": status, "code": r.status_code}
            print(f"  [{status}] {method} {url} -> {r.status_code}")
            if r.status_code >= 400:
                print(f"         Response: {json.dumps(data, indent=2)[:200] if data else 'None'}")
            return data, r.status_code
        except Exception as e:
            results[name] = {"status": f"ERROR: {e}", "code": 0}
            print(f"  [ERROR] {method} {url} -> {e}")
            return None, 0

    print("=" * 60)
    print("COMPREHENSIVE API TEST")
    print("=" * 60)

    # Health
    print("\n--- Health & Root ---")
    test("health", "GET", "/api/health")
    test("root", "GET", "/api")
    test("dashboard_page", "GET", "/")
    test("dashboard_page2", "GET", "/dashboard")

    # Static files
    print("\n--- Static Files ---")
    test("css", "GET", "/static/css/dashboard.css")
    test("js", "GET", "/static/js/dashboard.js")

    # Officers (test first since alerts may need them)
    print("\n--- Officers ---")
    test("officers_list", "GET", "/api/officers")
    data, _ = test("officers_setup_demo", "POST", "/api/officers/setup-demo")
    test("officers_list_after", "GET", "/api/officers")
    test("officers_create", "POST", "/api/officers", {
        "name": "Test Officer", "email": "test@test.com", "region": "Test Region"
    })
    test("officers_by_region", "GET", "/api/officers/by-region/Western Ghats")

    # Alerts (should be empty initially)
    print("\n--- Alerts ---")
    test("alerts_list", "GET", "/api/alerts")
    test("alerts_statistics", "GET", "/api/alerts/statistics")
    test("alerts_active", "GET", "/api/alerts/active")
    test("alerts_pending", "GET", "/api/alerts/pending")

    # Predictions (generates alerts)
    print("\n--- Predictions ---")
    data, code = test("pred_demo", "POST", "/api/predictions/demo", {
        "cause": "Mining", "latitude": 10.5, "longitude": 76.3,
        "region": "Western Ghats", "area_fraction": 0.3
    })
    alert_id = None
    if data and data.get("alert"):
        alert_id = data["alert"]["alert_id"]
        print(f"         Generated alert: {alert_id}")

    data, code = test("pred_demo2", "POST", "/api/predictions/demo", {
        "cause": "Fire", "latitude": 22.1, "longitude": 80.5,
        "region": "Central India", "area_fraction": 0.5
    })
    if data and data.get("alert"):
        print(f"         Generated alert: {data['alert']['alert_id']}")

    data, code = test("pred_demo3", "POST", "/api/predictions/demo", {
        "cause": "Logging", "latitude": 26.5, "longitude": 93.2,
        "region": "Northeast India", "area_fraction": 0.2
    })

    test("pred_recent", "GET", "/api/predictions/recent")
    test("pred_analyze", "POST", "/api/predictions/analyze", {"demo": True})

    # Alerts after predictions
    print("\n--- Alerts (after predictions) ---")
    data, _ = test("alerts_list_after", "GET", "/api/alerts")
    test("alerts_stats_after", "GET", "/api/alerts/statistics")
    test("alerts_active_after", "GET", "/api/alerts/active")

    if alert_id:
        test("alert_get", "GET", f"/api/alerts/{alert_id}")
        test("alert_ack", "POST", f"/api/alerts/{alert_id}/acknowledge")
        test("alert_status", "PUT", f"/api/alerts/{alert_id}/status", {"status": "investigating"})

    # Notifications
    print("\n--- Notifications ---")
    test("notif_status", "GET", "/api/notifications/status")
    test("notif_history", "GET", "/api/notifications/history")
    test("notif_test", "POST", "/api/notifications/test")
    if alert_id:
        test("notif_send", "POST", f"/api/notifications/send/{alert_id}")

    # Dashboard API
    print("\n--- Dashboard API ---")
    test("dash_overview", "GET", "/api/dashboard")
    test("dash_stats", "GET", "/api/dashboard/stats")
    test("dash_by_cause", "GET", "/api/dashboard/alerts-by-cause")
    test("dash_by_severity", "GET", "/api/dashboard/alerts-by-severity")
    test("dash_by_status", "GET", "/api/dashboard/alerts-by-status")
    test("dash_regions", "GET", "/api/dashboard/regions")
    test("dash_timeline", "GET", "/api/dashboard/timeline")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v["status"] == "PASS")
    failed = sum(1 for v in results.values() if v["status"] != "PASS")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\n  FAILURES:")
        for name, v in results.items():
            if v["status"] != "PASS":
                print(f"    - {name}: {v['status']}")

    return failed == 0

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
