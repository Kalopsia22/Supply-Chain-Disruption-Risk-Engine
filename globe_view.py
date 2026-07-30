"""
Real 3D interactive globe for the port network, using globe.gl (Three.js/WebGL),
loaded client-side from a CDN inside a Streamlit HTML component.

Why this works despite this project's various server-side network restrictions
(Open-Meteo, Google Maps, etc. all needed careful handling elsewhere in this
codebase): this is fundamentally different. The globe.gl library is fetched by
the END USER'S BROWSER when they view the page, not by the Streamlit server
process. It has no dependency on the server's outbound network access at all —
only on the browser's, which is essentially never restricted.

Ports are clickable (not just hoverable) — clicking one populates a detail
panel with real, precomputed statistics: for the 6 historical dataset lanes,
actual delay rate / risk score / volume; for the other 25 ports, region and
coordinates plus a note that they're live-tracking/route-optimizer-only.
"""
import json
import pandas as pd

from data_utils import CITY_COORDS, PORT_REGIONS

REGION_LABELS = {
    "east_asia": "East Asia", "sea_asia": "Southeast Asia", "south_asia": "South Asia",
    "middle_east": "Middle East", "n_europe": "Northern Europe", "med_europe": "Mediterranean Europe",
    "us_east": "US East Coast", "us_west": "US West Coast", "latam": "Latin America", "africa": "Africa",
}


def _build_port_info_html(port: str, df: pd.DataFrame) -> str:
    """Pre-render an HTML snippet with this port's real statistics, shown in
    the detail panel when the point is clicked."""
    lat, lon = CITY_COORDS[port]
    region = REGION_LABELS.get(PORT_REGIONS.get(port), "Unknown region")

    as_origin = df[df["Origin_City"] == port]
    as_dest = df[df["Destination_City"] == port]
    in_dataset = len(as_origin) > 0 or len(as_dest) > 0

    header = f"<div style='font-size:16px;font-weight:700;color:#e5e7eb;margin-bottom:4px;'>{port}</div>"
    sub = f"<div style='font-size:12px;color:#8b93a7;margin-bottom:10px;'>{region} &middot; {lat:.2f}, {lon:.2f}</div>"

    if not in_dataset:
        body = (
            "<div style='font-size:12.5px;color:#bfc8d6;line-height:1.5;'>"
            "Not one of the 6 historical shipment lanes — tracked here for "
            "<b>live weather</b> and the <b>route optimizer</b> only. "
            "No historical delay statistics available for this port."
            "</div>"
        )
    else:
        rows = ""
        if len(as_origin):
            rate = as_origin["Is_Delayed"].mean() * 100
            risk = as_origin["Shipment_Risk_Score"].mean()
            rows += (
                f"<div style='margin-bottom:8px;'>"
                f"<div style='font-size:11px;color:#8b93a7;'>AS ORIGIN &middot; {len(as_origin):,} shipments</div>"
                f"<div style='font-size:13px;color:#e5e7eb;'>Delay rate: <b style='color:#f5b942;'>{rate:.1f}%</b> "
                f"&middot; Avg risk score: <b style='color:#2dd4bf;'>{risk:.1f}</b></div></div>"
            )
        if len(as_dest):
            rate = as_dest["Is_Delayed"].mean() * 100
            risk = as_dest["Shipment_Risk_Score"].mean()
            rows += (
                f"<div>"
                f"<div style='font-size:11px;color:#8b93a7;'>AS DESTINATION &middot; {len(as_dest):,} shipments</div>"
                f"<div style='font-size:13px;color:#e5e7eb;'>Delay rate: <b style='color:#f5b942;'>{rate:.1f}%</b> "
                f"&middot; Avg risk score: <b style='color:#2dd4bf;'>{risk:.1f}</b></div></div>"
            )
        body = rows

    return header + sub + body


def build_ports_payload(df: pd.DataFrame) -> list:
    """One dict per port: coordinates, color, size, and precomputed info HTML."""
    dataset_ports = set(df["Origin_City"].unique()) | set(df["Destination_City"].unique())

    payload = []
    for port, (lat, lon) in CITY_COORDS.items():
        in_dataset = port in dataset_ports
        payload.append({
            "name": port,
            "lat": lat,
            "lng": lon,
            "color": "#2dd4bf" if in_dataset else "#5b8def",
            "size": 0.45 if in_dataset else 0.3,
            "info": _build_port_info_html(port, df),
        })
    return payload


def build_lanes_payload(df: pd.DataFrame) -> list:
    """Arcs for the real historical lanes, colored by average risk score."""
    lane_agg = (
        df.groupby(["Origin_City", "Destination_City"])
        .agg(avg_score=("Shipment_Risk_Score", "mean"), volume=("Shipment_Risk_Score", "count"),
             delay_rate=("Is_Delayed", "mean"))
        .reset_index()
    )

    lanes = []
    for _, row in lane_agg.iterrows():
        o, d = row["Origin_City"], row["Destination_City"]
        if o not in CITY_COORDS or d not in CITY_COORDS:
            continue
        score = row["avg_score"]
        color = "#2dd4bf" if score < 20 else "#f5b942" if score < 40 else "#ff8c42" if score < 65 else "#ef4444"
        lanes.append({
            "startLat": CITY_COORDS[o][0], "startLng": CITY_COORDS[o][1],
            "endLat": CITY_COORDS[d][0], "endLng": CITY_COORDS[d][1],
            "color": color,
            "label": f"{o} &rarr; {d}: {row['delay_rate']*100:.1f}% delayed, {row['volume']:,} shipments",
        })
    return lanes


def build_globe_html(df: pd.DataFrame, height: int = 620) -> str:
    """Full standalone HTML/JS for the 3D globe, ready for st.components.v1.html()."""
    points = build_ports_payload(df)
    lanes = build_lanes_payload(df)
    points_json = json.dumps(points)
    lanes_json = json.dumps(lanes)

    return f"""
<div id="globe-wrapper" style="position:relative; width:100%; height:{height}px; background:#0b0f19; border-radius:10px; overflow:hidden;">
  <div id="globeViz" style="width:100%; height:100%;"></div>

  <div id="info-panel" style="
      position:absolute; top:16px; right:16px; width:280px; max-width:38%;
      background:rgba(17,23,35,0.95); border:1px solid #232b3e; border-radius:10px;
      padding:14px 16px; backdrop-filter: blur(4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      font-family: 'Segoe UI', sans-serif; z-index:10;">
    <div style="font-size:13px; color:#8b93a7;">Click any port to see its stats</div>
  </div>

  <div style="position:absolute; bottom:12px; left:16px; z-index:10; font-family:'Segoe UI',sans-serif;">
    <label style="font-size:12px; color:#8b93a7; display:flex; align-items:center; gap:6px; cursor:pointer;">
      <input type="checkbox" id="rotate-toggle" checked style="cursor:pointer;"> Auto-rotate
    </label>
  </div>

  <div style="position:absolute; bottom:12px; right:16px; z-index:10; font-family:'Segoe UI',sans-serif; font-size:11px; color:#5a6b7a;">
    <span style="color:#2dd4bf;">&#9679;</span> In historical dataset &nbsp;
    <span style="color:#5b8def;">&#9679;</span> Live-tracking only
  </div>
</div>

<script src="https://unpkg.com/globe.gl"></script>
<script>
(function() {{
  const pointsData = {points_json};
  const lanesData = {lanes_json};

  const infoPanel = document.getElementById('info-panel');

  const globe = Globe()(document.getElementById('globeViz'))
    .backgroundColor('rgba(0,0,0,0)')
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .showAtmosphere(true)
    .atmosphereColor('#2dd4bf')
    .atmosphereAltitude(0.18)
    .pointsData(pointsData)
    .pointLat('lat')
    .pointLng('lng')
    .pointColor('color')
    .pointRadius('size')
    .pointAltitude(0.012)
    .pointsMerge(false)
    .pointLabel(d => `<div style="font-family:'Segoe UI',sans-serif;font-size:12px;background:rgba(11,15,25,0.9);padding:4px 8px;border-radius:4px;color:#e5e7eb;border:1px solid #2dd4bf;">${{d.name}}</div>`)
    .onPointClick(d => {{
      infoPanel.innerHTML = d.info;
    }})
    .arcsData(lanesData)
    .arcStartLat('startLat').arcStartLng('startLng')
    .arcEndLat('endLat').arcEndLng('endLng')
    .arcColor('color')
    .arcLabel('label')
    .arcStroke(0.5)
    .arcDashLength(0.4)
    .arcDashGap(0.15)
    .arcDashAnimateTime(2500)
    .arcAltitudeAutoScale(0.3)
    .width(document.getElementById('globe-wrapper').clientWidth)
    .height({height});

  globe.controls().autoRotate = true;
  globe.controls().autoRotateSpeed = 0.6;
  globe.controls().enableZoom = true;

  document.getElementById('rotate-toggle').addEventListener('change', (e) => {{
    globe.controls().autoRotate = e.target.checked;
  }});

  window.addEventListener('resize', () => {{
    const wrapper = document.getElementById('globe-wrapper');
    globe.width(wrapper.clientWidth).height({height});
  }});
}})();
</script>
"""
