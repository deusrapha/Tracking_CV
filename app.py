import os
import sys
import time
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add simulation and simulation/src to path for tracking imports
sys.path.append(os.path.join(os.path.dirname(__file__), "simulation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "simulation", "src"))

from models.baseline_tracker import BoundingBoxTracker, AppearanceExtractor
from models.projection import GroundPlaneProjector
from models.amodal_anchor import OcclusionAwareTracker
from run_simulation import HILSafetyWatchdog

# 1. Page Configuration
st.set_page_config(
    page_title="Beyond the Line of Sight - UAV HIL Simulation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme Toggle State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Design System
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
accent = "#2563eb"
accent_muted = "#1d4ed8"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"

css = f"""
<style>
:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: {accent};
    --accent-muted: {accent_muted};
    --green: {green};
    --green-muted: {green_muted};
    --red: {red};
    --red-muted: {red_muted};
    --amber: {amber};
    --amber-muted: {amber_muted};
    --shadow: {"none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"};
    --radius: 10px;
}}

/* Hide standard Streamlit header and decorations */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1360px !important;
}}

/* Custom pill-style tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px;
}}

/* Card container design */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow);
}}
.metric-label {{
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-top: 2px;
}}
.metric-delta {{
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.4rem;
    padding: 2px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-warn {{ color: var(--amber); background: var(--amber-muted); }}

.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.chart-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
}}
.chart-subtitle {{
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}}

.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
}}
.data-table td {{
    padding: 0.65rem 0.8rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}

.badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    text-align: center;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.brand-name {{
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}}
[data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stHorizontalBlock"]) {{
    margin-bottom: 0.5rem !important;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# 4. Helpers and Re-ID model loader
@st.cache_resource
def get_shared_extractor():
    # Cache the ONNX model loader so it only initializes once
    return AppearanceExtractor(onnx_path="src/models/reid_model.onnx")

# Load dataset frame paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "Dataset", "processed_frames", "DJI_0004")
frame_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))

if not frame_paths:
    st.error(f"Error: Calibration sequence folder '{dataset_dir}' not found or contains no frames.")
    st.stop()

# 5. Core Metric and Chart helpers
def metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "⚠")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# Initialize Session States
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "fps" not in st.session_state:
    st.session_state.fps = 5

# Custom User Fault Toggles
if "fault_gps_loss" not in st.session_state:
    st.session_state.fault_gps_loss = False
if "fault_geofence" not in st.session_state:
    st.session_state.fault_geofence = False
if "fault_low_conf" not in st.session_state:
    st.session_state.fault_low_conf = False
if "fault_yaw_spike" not in st.session_state:
    st.session_state.fault_yaw_spike = False

# Latency and Alarm Histories
if "latency_history" not in st.session_state:
    st.session_state.latency_history = []
if "breakdown_history" not in st.session_state:
    st.session_state.breakdown_history = []
if "watchdog_events" not in st.session_state:
    st.session_state.watchdog_events = []

def reset_simulation():
    st.session_state.frame_idx = 0
    st.session_state.playing = False
    
    # Re-initialize tracker models
    shared_extractor = get_shared_extractor()
    projector = GroundPlaneProjector()
    baseline = BoundingBoxTracker(iou_threshold=0.3, max_lost_frames=30, extractor=shared_extractor)
    st.session_state.tracker = OcclusionAwareTracker(baseline, projector)
    
    # Re-initialize safety watchdog
    geofence = [[100, 100], [900, 100], [900, 900], [100, 900]]
    st.session_state.watchdog = HILSafetyWatchdog(geofence)
    st.session_state.geofence = geofence
    
    # Reset history
    st.session_state.last_processed_frame_idx = -1
    st.session_state.latency_history = []
    st.session_state.breakdown_history = []
    st.session_state.watchdog_events = []

if "tracker" not in st.session_state or "watchdog" not in st.session_state:
    reset_simulation()

# 6. Single Simulation Step Executor
def run_single_sim_step(frame_idx):
    t_start = time.perf_counter()
    
    # Load frame
    f_path = frame_paths[frame_idx]
    t0 = time.perf_counter()
    frame = cv2.imread(f_path)
    if frame is None:
        return
    load_time = (time.perf_counter() - t0) * 1000.0
    
    # Step A: Vegetation/canopy segmentation
    t1 = time.perf_counter()
    # Runs the segmentation on the model projector
    occ_mask = st.session_state.tracker.projector.segment_vegetation(frame)
    segment_time = (time.perf_counter() - t1) * 1000.0
    
    # Step B: Telemetry inputs & homography
    t2 = time.perf_counter()
    # UAV coordinates (normal path moves from [500, 500])
    uav_pos = [500.0 + frame_idx * 5.0, 500.0 - frame_idx * 2.0]
    
    # Inject Geofence Fault
    # Frame 8 is default, but user toggle takes precedence
    is_geofence_fault = st.session_state.fault_geofence or (frame_idx == 8)
    if is_geofence_fault:
        uav_pos = [950.0, 950.0]
        
    yaw, pitch, roll = 0.05 * frame_idx, -0.02, 0.01
    
    # Inject Camera Yaw Spike
    # Frame 6 is default, but user toggle takes precedence
    is_yaw_fault = st.session_state.fault_yaw_spike or (frame_idx == 6)
    if is_yaw_fault:
        yaw = 15.5
        
    # Apply camera motion compensation (CMC) homography warping
    st.session_state.tracker.baseline_tracker.apply_camera_motion_compensation(frame)
    cmc_time = (time.perf_counter() - t2) * 1000.0
    
    # Step C: Target detection
    t3 = time.perf_counter()
    # Mock detection bounding box representation of cows moving
    detections = [[200.0 + frame_idx * 1.5, 200.0 + frame_idx * 0.8, 250.0 + frame_idx * 1.5, 250.0 + frame_idx * 0.8]]
    
    # Inject Low confidence detection
    # Frame 12 is default, but user toggle takes precedence
    is_low_conf_fault = st.session_state.fault_low_conf or (frame_idx == 12)
    det_score = 0.15 if is_low_conf_fault else 0.92
    
    # Execute step on tracking model
    st.session_state.tracker.step(frame, detections, altitude=30.0, pitch=-90.0, roll=0.0)
    tracking_time = (time.perf_counter() - t3) * 1000.0
    
    # Step D: Safety Watchdogs
    t4 = time.perf_counter()
    
    # Inject GPS Loss
    # Frame 4 is default, but user toggle takes precedence
    is_gps_fault = st.session_state.fault_gps_loss or (frame_idx == 4)
    gps_status = 0.0 if is_gps_fault else 1.0
    link_status = 1.0
    
    sig_ok, sig_action = st.session_state.watchdog.check_signals(gps_status, link_status)
    geo_ok, geo_action = st.session_state.watchdog.check_geofence(uav_pos)
    conf_ok, conf_action = st.session_state.watchdog.check_confidence(det_score)
    
    # Parse new logs added in this step
    new_logs = st.session_state.watchdog.logs[len(st.session_state.watchdog_events):]
    for log in new_logs:
        st.session_state.watchdog_events.append({
            "frame": frame_idx,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(log["timestamp"])),
            "event": log["event"],
            "details": log["details"],
            "status": "CRITICAL" if "FAULT" in log["details"] else "WARNING"
        })
        
    watchdog_time = (time.perf_counter() - t4) * 1000.0
    e2e_time = (time.perf_counter() - t_start) * 1000.0
    
    # Save latency stats
    st.session_state.latency_history.append(e2e_time)
    st.session_state.breakdown_history.append({
        "frame": frame_idx,
        "load": load_time,
        "segment": segment_time,
        "cmc": cmc_time,
        "tracking": tracking_time,
        "watchdog": watchdog_time
    })

# Main replay logic
target_idx = st.session_state.frame_idx

# Handle scrubbing backwards: Reset tracker and play up to target frame
if target_idx < st.session_state.last_processed_frame_idx:
    f_gps = st.session_state.fault_gps_loss
    f_geo = st.session_state.fault_geofence
    f_conf = st.session_state.fault_low_conf
    f_yaw = st.session_state.fault_yaw_spike
    
    reset_simulation()
    
    st.session_state.fault_gps_loss = f_gps
    st.session_state.fault_geofence = f_geo
    st.session_state.fault_low_conf = f_conf
    st.session_state.fault_yaw_spike = f_yaw
    st.session_state.frame_idx = target_idx

# Fast-forward / step sequentially up to target frame
while st.session_state.last_processed_frame_idx < target_idx:
    next_step_idx = st.session_state.last_processed_frame_idx + 1
    run_single_sim_step(next_step_idx)
    st.session_state.last_processed_frame_idx = next_step_idx

# 7. Rendering Header
head_left, head_right = st.columns([7, 3])
with head_left:
    st.markdown("""
    <div class="brand">
        <span style="font-size: 1.6rem;">◆</span>
        <span class="brand-name">Beyond the Line of Sight: UAV Tracking & Watchdog Simulator</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)
    with btn_col2:
        st.button("⚙ Reset", on_click=reset_simulation, use_container_width=True)
    with btn_col3:
        if st.session_state.playing:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.playing = False
                st.rerun()
        else:
            if st.button("▶ Play", use_container_width=True):
                st.session_state.playing = True
                st.rerun()

# Determine overall flight status based on watchdog logs
current_status = "NORMAL"
status_class = "badge-green"
last_fault_msg = "All watchdogs clear. Tracking normal."

if st.session_state.watchdog_events:
    last_evt = st.session_state.watchdog_events[-1]
    # Check if last event matches current frame or is active
    if last_evt["frame"] == st.session_state.frame_idx:
        if "RTL" in last_evt["details"]:
            current_status = "RTL (RETURN TO LAND)"
            status_class = "badge-amber"
            last_fault_msg = last_evt["details"]
        elif "ABORT" in last_evt["details"]:
            current_status = "ABORT / FLIGHT SUSPENDED"
            status_class = "badge-red"
            last_fault_msg = last_evt["details"]
        elif "TAKEOVER" in last_evt["details"]:
            current_status = "OPERATOR TAKEOVER"
            status_class = "badge-blue"
            last_fault_msg = last_evt["details"]

# Compute canopy coverage percentage for the current frame mask
f_path = frame_paths[st.session_state.frame_idx]
img_test = cv2.imread(f_path)
if img_test is not None:
    test_mask = st.session_state.tracker.projector.segment_vegetation(img_test)
    canopy_coverage = (test_mask > 0).mean() * 100.0
else:
    canopy_coverage = 0.0

active_tracks = len([t for t in st.session_state.tracker.baseline_tracker.tracks if t.time_since_update == 0])
active_anchors = len(st.session_state.tracker.amodal_anchors)

# Metrics Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">UAV Flight Status</div>
        <div style="margin-top:8px;"><span class="badge {status_class}" style="font-size:0.9rem; padding: 4px 12px;">{current_status}</span></div>
    </div>
    """, unsafe_allow_html=True)
with m_col2:
    e2e_ms = st.session_state.latency_history[-1] if st.session_state.latency_history else 0.0
    metric_card("E2E Process Latency", f"{e2e_ms:.1f} ms", delta="Tracking Feed active", delta_type="up")
with m_col3:
    metric_card("Canopy Segmentation", f"{canopy_coverage:.1f}%", delta="Veg Area Coverage", delta_type="warn" if canopy_coverage > 10.0 else "up")
with m_col4:
    metric_card("Tracking State", f"{active_tracks} Active / {active_anchors} Anchored", delta="Occlusion Anchors seeded", delta_type="up" if active_anchors > 0 else "warn")

# Main Content Layout
col_main, col_sidebar = st.columns([7, 5])

# Frame drawer function
def draw_visualization_frame(frame_idx):
    f_path = frame_paths[frame_idx]
    img = cv2.imread(f_path)
    if img is None:
        return None
        
    # Draw vegetation mask overlay (green)
    occ_mask = st.session_state.tracker.projector.segment_vegetation(img)
    overlay = img.copy()
    overlay[occ_mask > 0] = [34, 197, 94]
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
    
    # Draw detection (cow) bounding box in gray
    det_box = [200.0 + frame_idx * 1.5, 200.0 + frame_idx * 0.8, 250.0 + frame_idx * 1.5, 250.0 + frame_idx * 0.8]
    x1_d, y1_d, x2_d, y2_d = map(int, det_box)
    cv2.rectangle(img, (x1_d, y1_d), (x2_d, y2_d), (220, 220, 220), 2)
    cv2.putText(img, "DETECTION (COW)", (x1_d, y1_d - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    
    # Draw active tracks in blue
    for track in st.session_state.tracker.baseline_tracker.tracks:
        if track.time_since_update == 0:
            x1, y1, x2, y2 = map(int, track.last_bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (235, 99, 37), 3) # BGR Orange
            cv2.putText(img, f"COW ID {track.track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 99, 37), 2)
            
    # Draw active amodal anchors (occluded targets) in orange/red
    for track_id, anchor in st.session_state.tracker.amodal_anchors.items():
        x1, y1, x2, y2 = map(int, anchor.last_bbox)
        col = (68, 68, 239) if anchor.frames_occluded > 15 else (11, 158, 245) # BGR Red/Amber
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"ANCHOR {track_id} (OCCLUDED)", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        # Uncertainty ellipse
        var_x = anchor.Sigma[0, 0]
        var_y = anchor.Sigma[1, 1]
        r = int(np.sqrt(var_x + var_y) * 2.5)
        cv2.circle(img, (int(anchor.cx), int(anchor.cy)), r, col, 1)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with col_main:
    # 8. Video Replay Window
    vis_frame = draw_visualization_frame(st.session_state.frame_idx)
    st.image(vis_frame, use_container_width=True, caption=f"UAV Downlink Stream - Frame {st.session_state.frame_idx + 1} / {len(frame_paths)} (DJI_0004 Calibration)")
    
    # 9. Performance Analytics Tab View
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Real-Time Performance Diagnostics</div>
        <div class="chart-subtitle">Execution latency metrics per computational sub-step</div>
    """, unsafe_allow_html=True)
    
    if st.session_state.breakdown_history:
        df = pd.DataFrame(st.session_state.breakdown_history)
        
        # Stacked bar chart showing step breakdowns
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['frame'], y=df['load'], name='Frame Load', marker_color='#6b7280'))
        fig.add_trace(go.Bar(x=df['frame'], y=df['segment'], name='Segmentation', marker_color='#10b981'))
        fig.add_trace(go.Bar(x=df['frame'], y=df['cmc'], name='CMC Homography', marker_color='#6366f1'))
        fig.add_trace(go.Bar(x=df['frame'], y=df['tracking'], name='Amodal Tracking', marker_color='#f59e0b'))
        fig.add_trace(go.Bar(x=df['frame'], y=df['watchdog'], name='Watchdog Check', marker_color='#ef4444'))
        
        fig.update_layout(
            barmode='stack',
            height=200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Play or step the simulation to populate performance charts.")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_sidebar:
    # 10. Geofence Arena View
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Hardware-in-the-Loop Geofence Monitor</div>
        <div class="chart-subtitle">2D Ground Projection coordinate plane (UAV bounds mapping)</div>
    """, unsafe_allow_html=True)
    
    # Generate Map figure
    geofence = np.array(st.session_state.geofence)
    geo_closed = np.vstack([geofence, geofence[0]])
    
    path_x = [500.0 + idx * 5.0 for idx in range(st.session_state.frame_idx + 1)]
    path_y = [500.0 - idx * 2.0 for idx in range(st.session_state.frame_idx + 1)]
    
    uav_x = path_x[-1]
    uav_y = path_y[-1]
    
    is_breached = st.session_state.fault_geofence or (st.session_state.frame_idx == 8)
    if is_breached:
        uav_x, uav_y = 950.0, 950.0
        path_x[-1] = 950.0
        path_y[-1] = 950.0
        
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=geo_closed[:, 0], y=geo_closed[:, 1],
        mode='lines',
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.05)' if not is_breached else 'rgba(239, 68, 68, 0.05)',
        line=dict(color='#2563eb' if not is_breached else '#ef4444', width=2, dash='dash'),
        name='Geofence Boundary'
    ))
    fig_map.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=4),
        name='Flight Path'
    ))
    
    uav_color = '#ef4444' if is_breached else '#10b981'
    uav_symbol = 'x' if is_breached else 'circle'
    fig_map.add_trace(go.Scatter(
        x=[uav_x], y=[uav_y],
        mode='markers',
        marker=dict(color=uav_color, size=12, symbol=uav_symbol, line=dict(color='#ffffff', width=1.5)),
        name='UAV Current Pos'
    ))
    
    # Merge PLOT_LAYOUT to prevent duplicate keyword arguments (xaxis, yaxis) in update_layout
    map_layout = PLOT_LAYOUT.copy()
    map_layout['xaxis'] = {**map_layout['xaxis'], 'range': [0, 1000], 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.04)' if IS_DARK else 'rgba(0,0,0,0.04)'}
    map_layout['yaxis'] = {**map_layout['yaxis'], 'range': [0, 1000], 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.04)' if IS_DARK else 'rgba(0,0,0,0.04)'}
    
    fig_map.update_layout(
        showlegend=False,
        height=190,
        **map_layout
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 11. Interactive Fault Injection Controls
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Dynamic Fault Injection Panel</div>
        <div class="chart-subtitle">Click to force watchdogs to test fail-safe routines (RTL, Abort, Takeover)</div>
    """, unsafe_allow_html=True)
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.checkbox("Force GPS Loss (RTL)", key="fault_gps_loss")
        st.checkbox("Force Low Target Confidence", key="fault_low_conf")
    with c_f2:
        st.checkbox("Force Geofence Breach", key="fault_geofence")
        st.checkbox("Force Cam Yaw Spike", key="fault_yaw_spike")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 12. Real-Time Safety Watchdog Log
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Safety Watchdog Alarm System</div>
        <div class="chart-subtitle">Active flight computer events and warning indicators</div>
    """, unsafe_allow_html=True)
    
    if st.session_state.watchdog_events:
        # Convert events list to table rows
        rows_html = ""
        for evt in reversed(st.session_state.watchdog_events):
            badge_class = "badge-red" if evt["status"] == "CRITICAL" else "badge-amber"
            rows_html += f"""
            <tr>
                <td><span class="badge {badge_class}">{evt["status"]}</span></td>
                <td><b>Frame {evt["frame"] + 1}</b></td>
                <td>{evt["event"]}</td>
                <td>{evt["details"]}</td>
            </tr>
            """
            
        st.markdown(f"""
        <div style="max-height: 220px; overflow-y: auto;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Frame</th>
                        <th>Alarm Type</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 1rem; border-radius: var(--radius); background: var(--bg-subtle); border: 1px solid var(--border); text-align: center; color: var(--text-muted); font-size: 0.8rem;">
            🟢 All watchdogs normal. No active faults detected.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# 13. Replay loop auto-stepping
# Run animation if playing is True and we're not at the end
if st.session_state.playing:
    if st.session_state.frame_idx < len(frame_paths) - 1:
        st.session_state.frame_idx += 1
        time.sleep(1.0 / st.session_state.fps)
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()

# 14. Bottom Slider Control
st.markdown("---")
slider_col1, slider_col2 = st.columns([1, 11])
with slider_col1:
    st.markdown("<p style='margin-top: 10px; font-weight: 500; font-size:0.835rem; color:var(--text-muted);'>Timeline</p>", unsafe_allow_html=True)
with slider_col2:
    st.slider(
        "Simulation Frames",
        min_value=0,
        max_value=len(frame_paths) - 1,
        key="frame_idx",
        label_visibility="collapsed"
    )
