"""
Google Maps integration — used ONLY for what Google Maps can actually do:
road-network driving routes (last-mile / inland delivery legs) and address
geocoding. It is deliberately NOT used for the ocean/air trunk routes between
ports (that's route_optimizer.py's job, via Dijkstra + historical risk data).

WHY NOT: Google's routing APIs compute routes over roads, walking paths,
bike lanes, and transit networks. There is no road between Shanghai and
Rotterdam — a container ship doesn't follow a road, and Google Maps has no
concept of a shipping lane or a flight path. Asking it to route an ocean leg
would either fail outright or (worse) silently return a nonsensical result.
This module exists for the genuinely Google-Maps-appropriate part of the
logistics chain: the truck from the arrival port to the final delivery
address.

API VERSION NOTE: Google deprecated the legacy Directions API and Distance
Matrix API (moved to Legacy status 1 March 2025; JS API versions face a hard
removal in the May 2026 deprecation wave). This module targets the CURRENT
Routes API (`routes.googleapis.com`, POST + JSON body + field mask header),
not the old GET-based Directions API endpoint that a lot of older sample
code still shows.

REQUIRES A BILLING-ENABLED GOOGLE CLOUD API KEY — unlike Open-Meteo elsewhere
in this project, Google Maps Platform is not free/keyless. You need a Google
Cloud project with billing enabled and the "Routes API" and "Geocoding API"
enabled for your key.

NOTE ON TESTING: built and logic-tested against Google's documented current
request/response schema (verified via their docs as of this writing), but
the dev sandbox this was built in cannot reach routes.googleapis.com or
maps.googleapis.com (confirmed via direct curl — 403 from the sandbox's own
egress proxy), so the live HTTP round-trip itself could not be executed here.
Test with a real API key after deploying.
"""
import requests

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GEOCODE_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"
REQUEST_TIMEOUT = 15


def geocode_address(address: str, api_key: str) -> dict:
    """Resolve a free-text address/city to coordinates via the Geocoding API
    (a separate, still-current product from the Routes API migration)."""
    if not api_key:
        return {"ok": False, "error": "No Google Maps API key provided."}

    try:
        resp = requests.get(
            GEOCODE_API_URL,
            params={"address": address, "key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request failed: {e}"}

    status = data.get("status")
    if status != "OK":
        error_map = {
            "ZERO_RESULTS": f"No location found for '{address}'.",
            "REQUEST_DENIED": "Request denied — check that your API key is valid and the Geocoding API is enabled for it.",
            "OVER_QUERY_LIMIT": "Google Maps quota exceeded for this key.",
            "INVALID_REQUEST": "Invalid geocoding request.",
        }
        return {"ok": False, "error": error_map.get(status, f"Geocoding failed: {status}")}

    try:
        result = data["results"][0]
        loc = result["geometry"]["location"]
        return {
            "ok": True,
            "lat": loc["lat"],
            "lon": loc["lng"],
            "formatted_address": result.get("formatted_address", address),
        }
    except (KeyError, IndexError) as e:
        return {"ok": False, "error": f"Unexpected geocoding response shape: {e}"}


def compute_driving_route(origin_coords: tuple, dest_coords: tuple, api_key: str) -> dict:
    """Real road-network driving route between two points, via the CURRENT
    Google Routes API (computeRoutes, POST, not the deprecated GET Directions
    API). Intended for last-mile/inland legs only — see module docstring."""
    if not api_key:
        return {"ok": False, "error": "No Google Maps API key provided."}

    body = {
        "origin": {"location": {"latLng": {"latitude": origin_coords[0], "longitude": origin_coords[1]}}},
        "destination": {"location": {"latLng": {"latitude": dest_coords[0], "longitude": dest_coords[1]}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_UNAWARE",
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # Explicit field list per Google's guidance — avoid the "*" wildcard.
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.warnings",
    }

    try:
        resp = requests.post(ROUTES_API_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request failed: {e}"}

    if resp.status_code == 403:
        return {"ok": False, "error": "403 from Google — check that your API key is valid, billing is enabled on the project, and the 'Routes API' is enabled for this key."}
    if resp.status_code == 429:
        return {"ok": False, "error": "429 Too Many Requests — Google Maps quota exceeded for this key."}
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    try:
        data = resp.json()
        routes = data.get("routes", [])
        if not routes:
            return {"ok": False, "error": "No driving route found between these points (they may not be road-connected — e.g. separated by water with no bridge/ferry, or one point isn't reachable by road)."}

        route = routes[0]
        distance_km = route["distanceMeters"] / 1000.0
        duration_str = route["duration"]  # e.g. "5024s"
        duration_minutes = float(duration_str.rstrip("s")) / 60.0

        return {
            "ok": True,
            "distance_km": distance_km,
            "duration_minutes": duration_minutes,
            "duration_hours": duration_minutes / 60.0,
        }
    except (KeyError, ValueError, TypeError) as e:
        return {"ok": False, "error": f"Unexpected response shape: {e}"}
