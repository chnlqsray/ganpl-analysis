"""
GaN PL 智能分析平台
Local Streamlit App — Phase 1: SPF2 Parsing + Spectrum Correction + Interactive Viewer
"""

import streamlit as st
import numpy as np
import os
import re
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, differential_evolution, root
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GaN PL Analysis",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
SPF2_HEADER_BYTES = 1084
SPF2_N_PIXELS     = 3648
MAIN_CORR_FILE    = "ccs_correction_curve_10 smooth.txt"
DIV_CORR_FILES    = [
    "UV flat beamsplitter.txt",
    "UV x15 lens.txt",
    "Vis cube beamsplitter.txt",
]
WL_MIN, WL_MAX = 350, 800
EPS = 1e-8

DEFAULT_DEAD_PIXELS = [557.48]   # nm — add more as needed

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for key, default in [
    ("spectrum_dir", ""),
    ("last_processed", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────

# Byte offset of the integration time field in the SPF2 header (float64, milliseconds)
SPF2_INTEGRATION_TIME_OFFSET = 900

def parse_spf2(filepath: str):
    """
    Parse a Thorlabs CCS200 SPF2 file.

    Binary layout:
        Bytes 0–1083   : header (metadata; integration time at byte 900 as float64 ms)
        Bytes 1084–15675: 3648 × float32  wavelengths (nm)
        Bytes 15676–30267: 3648 × float32  intensities (raw accumulated)

    Integration time is read directly from the header (byte 900, float64, milliseconds).
    Filename parsing is used as fallback only if the header value is invalid.

    Returns: wl (nm), intensity_per_second, exposure_s
    """
    import struct as _struct

    with open(filepath, "rb") as f:
        data = f.read()

    expected = SPF2_HEADER_BYTES + SPF2_N_PIXELS * 8
    if len(data) < expected:
        raise ValueError(f"File too small ({len(data)} bytes, expected ≥{expected}).")

    # --- Read integration time from header (byte 900, float64, milliseconds) ---
    try:
        integration_ms = _struct.unpack('<d', data[SPF2_INTEGRATION_TIME_OFFSET:
                                                    SPF2_INTEGRATION_TIME_OFFSET + 8])[0]
        if not (0.001 <= integration_ms <= 60000):
            raise ValueError(f"Implausible integration time value: {integration_ms} ms")
        exposure_s = integration_ms / 1000.0
    except Exception as e:
        raise ValueError(
            f"Could not read integration time from SPF2 header of "
            f"'{os.path.basename(filepath)}': {e}"
        )

    wl = np.frombuffer(
        data[SPF2_HEADER_BYTES : SPF2_HEADER_BYTES + SPF2_N_PIXELS * 4],
        dtype="<f4"
    ).copy().astype(float)

    raw = np.frombuffer(
        data[SPF2_HEADER_BYTES + SPF2_N_PIXELS * 4 : SPF2_HEADER_BYTES + SPF2_N_PIXELS * 8],
        dtype="<f4"
    ).copy().astype(float)

    return wl, raw / exposure_s, exposure_s


def load_correction_interp(filepath: str):
    """Load two-column correction file → interp1d (returns 1 everywhere on failure)."""
    if not os.path.isfile(filepath):
        return lambda wl: np.ones_like(wl, dtype=float)
    try:
        d = np.loadtxt(filepath)
        return interp1d(d[:, 0], d[:, 1], kind="linear",
                        bounds_error=False, fill_value=(float(d[:, 1][0]), float(d[:, 1][-1])), assume_sorted=False,)
    except Exception:
        return lambda wl: np.ones_like(wl, dtype=float)


def apply_correction(wl: np.ndarray, intensity: np.ndarray, calib_folder: str,
                     file_roles: dict | None = None):
    """
    Apply spectral correction and clip to [WL_MIN, WL_MAX].
    file_roles: {filename: "Multiply"|"Divide"|"Disabled"}
    Falls back to the hardcoded MAIN_CORR_FILE/DIV_CORR_FILES if file_roles is None.
    Returns: wl_clipped, corrected, normalised
    """
    # Build lists from roles dict (or legacy constants)
    if file_roles:
        mul_files = [fn for fn, role in file_roles.items() if role == "Multiply"]
        div_files = [fn for fn, role in file_roles.items() if role == "Divide"]
    else:
        mul_files = [MAIN_CORR_FILE]
        div_files = DIV_CORR_FILES

    mul_interps = [load_correction_interp(os.path.join(calib_folder, fn)) for fn in mul_files]
    div_interps = [load_correction_interp(os.path.join(calib_folder, fn)) for fn in div_files]

    mask = (wl >= WL_MIN) & (wl <= WL_MAX)
    wl_c, int_c = wl[mask], intensity[mask]

    corrected = int_c.copy()
    for m in mul_interps:
        corrected *= np.array(m(wl_c), dtype=float)
    for d in div_interps:
        dv = np.array(d(wl_c), dtype=float)
        corrected /= np.where(np.abs(dv) < EPS, EPS, dv)

    max_val    = np.max(np.abs(corrected)) or 1.0
    normalised = corrected / max_val
    return wl_c, corrected, normalised


def fix_dead_pixels(wl: np.ndarray, intensity: np.ndarray,
                    dead_wl_list: list, window: int = 2) -> np.ndarray:
    """
    Replace dead-pixel spikes with linear interpolation from neighbouring points.
    For each wavelength in dead_wl_list, finds the nearest index, then replaces
    it with the mean of `window` points on each side.
    """
    intensity = intensity.copy()
    for dead_nm in dead_wl_list:
        idx = int(np.argmin(np.abs(wl - dead_nm)))
        left_idx  = max(0, idx - window)
        right_idx = min(len(intensity) - 1, idx + window)
        neighbours = np.concatenate([
            intensity[left_idx : idx],
            intensity[idx + 1 : right_idx + 1],
        ])
        if len(neighbours) > 0:
            intensity[idx] = float(np.mean(neighbours))
    return intensity


def process_spf2_file(spf2_path: str, calib_folder: str,
                      out_dir: str, dead_pixels: list,
                      file_roles: dict | None = None) -> dict:
    """Full pipeline: parse → correct → fix dead pixels → save two output files."""
    wl_raw, intensity_ps, exposure_s = parse_spf2(spf2_path)
    wl, corrected, normalised = apply_correction(wl_raw, intensity_ps, calib_folder, file_roles)
    corrected  = fix_dead_pixels(wl, corrected,  dead_pixels)
    normalised = fix_dead_pixels(wl, normalised, dead_pixels)

    stem     = Path(spf2_path).stem
    out_corr = os.path.join(out_dir, stem + "_corrected.txt")
    out_norm = os.path.join(out_dir, stem + "_corrected_normalize.txt")

    np.savetxt(out_corr, np.column_stack((wl, corrected)),
               fmt="%.6f", header="Wavelength Intensity",            comments="")
    np.savetxt(out_norm, np.column_stack((wl, normalised)),
               fmt="%.6f", header="Wavelength Intensity_Normalized", comments="")

    return dict(name=stem, exposure_s=exposure_s,
                wl=wl, corrected=corrected, normalised=normalised,
                out_corr=out_corr, out_norm=out_norm)


def load_corrected_txt(filepath: str):
    """Load _corrected.txt or _corrected_normalize.txt → (wl, intensity)."""
    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]


def wavelength_to_eV(wl_nm: np.ndarray) -> np.ndarray:
    return 1239.84 / wl_nm


def clean_label(filename: str) -> str:
    return (filename
            .replace("_corrected_normalize.txt", "")
            .replace("_corrected.txt", ""))


def _load_intensity_txt(path: str):
    """
    Parse intensity_with_axes.txt (Mapping format).
    First row: tab-separated, first cell ignored, remaining = X coords.
    Subsequent rows: first cell = Y coord, remaining = intensity values.
    Returns x_coords (sorted asc), y_coords (sorted asc), data_2d (n_y × n_x).
    """
    with open(path, "r") as fh:
        lines = [l.rstrip("\n\r") for l in fh if l.strip()]
    x_raw = np.array(list(map(float, lines[0].split("\t")[1:])))
    y_raw, rows = [], []
    for line in lines[1:]:
        parts = line.split("\t")
        y_raw.append(float(parts[0]))
        rows.append(list(map(float, parts[1:])))
    y_raw    = np.array(y_raw)
    data_raw = np.array(rows)
    xi = np.argsort(x_raw);  x_c = x_raw[xi];  data_raw = data_raw[:, xi]
    yi = np.argsort(y_raw);  y_c = y_raw[yi];  data_out = data_raw[yi, :]
    return x_c, y_c, data_out


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 GaN PL Analysis")
    st.markdown("---")

    # ── Calibration ──
    st.markdown("### Calibration files")
    st.caption(
        "**Multiply**: multiplies the spectrum by this file's curve "
        "(e.g. a detector/system response correction that boosts sensitivity).  \n"
        "**Divide**: divides the spectrum by this file's curve "
        "(e.g. removes wavelength-dependent transmission/reflection of an optical element).  \n"
        "To add a new file: place the `.txt` in the calibration folder, then restart the app."
    )
    default_calib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    calib_folder = st.text_input(
        "Calibration folder",
        value=default_calib,
        help="Folder containing spectral correction files.",
    ).strip()

    # Determine roles for each .txt found in the folder
    # Default role is inferred from the hard-coded constants; new files default to Divide
    _ROLE_OPTIONS = ["Multiply", "Divide", "Disabled"]
    calib_file_roles: dict[str, str] = {}   # filename → role string

    if os.path.isdir(calib_folder):
        _all_calib_txt = sorted([
            f for f in os.listdir(calib_folder) if f.lower().endswith(".txt")
        ])
        if _all_calib_txt:
            for _fn in _all_calib_txt:
                if _fn == MAIN_CORR_FILE:
                    _default_role = "Multiply"
                elif _fn in DIV_CORR_FILES:
                    _default_role = "Divide"
                else:
                    _default_role = "Divide"   # sensible default for new files
                _role_key = f"calib_role__{_fn}"
                if _role_key not in st.session_state:
                    st.session_state[_role_key] = _default_role
                _exists = os.path.isfile(os.path.join(calib_folder, _fn))
                _icon = "✅" if _exists else "❌"
                _col_lbl, _col_sel = st.columns([3, 2])
                with _col_lbl:
                    st.caption(f"{_icon} {_fn}")
                with _col_sel:
                    calib_file_roles[_fn] = st.selectbox(
                        label=_fn,
                        options=_ROLE_OPTIONS,
                        key=_role_key,          # value driven by session_state only
                        label_visibility="collapsed",
                    )
        else:
            st.warning("No `.txt` files found in calibration folder.")
    else:
        st.warning("Calibration folder not found.")

    st.markdown("---")

    # ── Dead pixels ──
    st.markdown("### Dead pixel correction")
    dead_px_raw = st.text_area(
        "Dead pixel wavelengths (nm)",
        value="\n".join(str(x) for x in DEFAULT_DEAD_PIXELS),
        height=90,
        key="dead_pixel_input",
        help="One wavelength per line. Each flagged point is replaced by the "
             "mean of its 2 nearest neighbours on each side.",
    )
    dead_pixels = []
    for line in dead_px_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dead_pixels.append(float(line))
        except ValueError:
            st.warning(f"Ignored invalid value: '{line}'")

    st.markdown("---")

    # ── Plot appearance ──
    st.markdown("### Plot appearance")
    font_size  = st.slider("Font size",   min_value=10, max_value=28, value=16, step=1)
    font_bold  = st.checkbox("Bold text", value=True)
    line_width = st.slider("Line width",  min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    tick_len   = st.slider("Tick length", min_value=2,   max_value=20,  value=6,   step=1)
    tick_width = st.slider("Tick width",  min_value=1,   max_value=5,   value=1,   step=1)
    axis_lw    = st.slider("Axis line width", min_value=1, max_value=5, value=1,   step=1)

    st.markdown("---")


# ─────────────────────────────────────────────
#  Global plot font dicts (depend on sidebar sliders)
# ─────────────────────────────────────────────
font_family = "Arial Black" if font_bold else "Arial"
_axis_font  = dict(size=font_size, family=font_family, color="black")
_tick_font  = dict(size=max(font_size - 2, 10), family=font_family, color="black")

# Global axis/tick line style (applied to every figure via _style_axes)
_ax_style = dict(
    showline=True, linecolor="black", linewidth=axis_lw,
    ticks="outside", tickwidth=tick_width, ticklen=tick_len, tickcolor="black",
)

def _style_axes(fig: go.Figure) -> go.Figure:
    """Apply global tick/axis-line style to all axes of a Plotly figure."""
    fig.update_xaxes(**_ax_style)
    fig.update_yaxes(**_ax_style)
    return fig

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📂 Batch Processor",
    "📈 Spectrum Viewer",
    "🔎 Peak Fitting",
    "⚡ Power Series",
    "📊 Lifetime Compare",
    "🎨 CIE Diagram",
    "🗺️ Mapping",
    "🤖 AI Copilot",
])


# ══════════════════════════════════════════════
#  TAB 1 — Batch Processor
# ══════════════════════════════════════════════
with tab1:
    st.header("Batch SPF2 Processor")
    st.markdown(
        "Point to a sample folder containing `.spf2` files. "
        "Corrected spectra are saved to a `spectrum/` subfolder and the "
        "**Spectrum Viewer** tab will load them automatically."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_folder = st.text_input(
            "Sample folder path",
            placeholder=r"e.g.  C:\data\unimplanted",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("🔍 Scan folder", width="stretch")

    # ── Scan ──
    spf2_files = []
    if sample_folder and os.path.isdir(sample_folder):
        spf2_files = sorted([
            f for f in os.listdir(sample_folder) if f.lower().endswith(".spf2")
        ])

    if scan_btn or spf2_files:
        if not os.path.isdir(sample_folder):
            st.error("Folder not found. Please check the path.")
        elif not spf2_files:
            st.warning("No `.spf2` files found in this folder.")
        else:
            rows = []
            import struct as _s
            for fname in spf2_files:
                fpath = os.path.join(sample_folder, fname)
                exp_label = "⚠️ unreadable"
                try:
                    with open(fpath, "rb") as _f:
                        _f.seek(SPF2_INTEGRATION_TIME_OFFSET)
                        val_ms = _s.unpack('<d', _f.read(8))[0]
                    if 0.001 <= val_ms <= 60000:
                        exp_label = f"{val_ms/1000:.3g} s"
                except Exception:
                    pass
                rows.append({"File": fname, "Exposure time": exp_label})
            st.success(f"Found **{len(spf2_files)}** SPF2 file(s):")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("---")

    process_btn = st.button(
        "▶️ Start processing",
        disabled=(not spf2_files),
        type="primary",
        width="stretch",
    )

    if process_btn and spf2_files:
        out_dir = os.path.join(sample_folder, "spectrum")
        os.makedirs(out_dir, exist_ok=True)

        progress_bar = st.progress(0, text="Initialising…")
        log_area     = st.empty()
        log_lines    = []
        results      = []

        for i, fname in enumerate(spf2_files):
            fpath = os.path.join(sample_folder, fname)
            progress_bar.progress(
                i / len(spf2_files),
                text=f"Processing {i + 1}/{len(spf2_files)}: {fname}"
            )
            try:
                res = process_spf2_file(fpath, calib_folder, out_dir, dead_pixels,
                                        calib_file_roles if calib_file_roles else None)
                results.append(res)
                dp_note = f", {len(dead_pixels)} dead pixel(s) corrected" if dead_pixels else ""
                log_lines.append(
                    f"✅  {fname}  →  {res['exposure_s']:.0f} s"
                    f"  |  peak: {res['corrected'].max():.4g}{dp_note}"
                )
            except Exception as e:
                log_lines.append(f"❌  {fname}  →  ERROR: {e}")
            log_area.code("\n".join(log_lines), language=None)

        progress_bar.progress(1.0, text="Done!")

        if results:
            st.session_state.spectrum_dir = out_dir
            st.session_state.last_processed = [r["out_corr"] for r in results]
            # Also update the viewer widget's own key so it renders with the new path
            st.session_state["viewer_path_input"] = out_dir
            st.success(
                f"✅ Processed **{len(results)}/{len(spf2_files)}** file(s). "
                f"Output → `{out_dir}`  \n"
                f"Switch to the **Spectrum Viewer** tab — results are loaded automatically."
            )


# ══════════════════════════════════════════════
#  TAB 2 — Spectrum Viewer
# ══════════════════════════════════════════════
with tab2:
    st.header("Interactive Spectrum Viewer")

    # ── Path input — driven entirely by key to avoid session-state conflict ──
    # Initialise the widget key from spectrum_dir if not yet set (cold start).
    # Tab 1 writes st.session_state["viewer_path_input"] = out_dir directly
    # to hand off the processed path; we must NOT also supply value= here or
    # Streamlit raises a "default value + session state" warning.
    if "viewer_path_input" not in st.session_state:
        st.session_state["viewer_path_input"] = st.session_state.spectrum_dir
    new_dir = st.text_input(
        "Spectrum folder path",
        placeholder=r"e.g.  C:\data\unimplanted\spectrum",
        key="viewer_path_input",
    )
    st.session_state.spectrum_dir = new_dir
    current_dir = new_dir

    # ── Discover files ──
    corr_files, norm_files = [], []
    if current_dir and os.path.isdir(current_dir):
        all_txt    = sorted(os.listdir(current_dir))
        corr_files = [f for f in all_txt if f.endswith("_corrected.txt")]
        norm_files = [f for f in all_txt if f.endswith("_corrected_normalize.txt")]

    if not corr_files and not norm_files:
        st.info(
            "No corrected spectra found. "
            "Run the **Batch Processor** first, or enter a valid spectrum folder path."
        )

    else:
        # ── Display options ──
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            data_type = st.radio("Data type", ["Corrected", "Normalised"], horizontal=True)
        with col_b:
            x_axis = st.radio("X axis", ["Wavelength (nm)", "Energy (eV)"], horizontal=True)
        with col_c:
            show_legend = st.checkbox("Show legend", value=True)
        with col_d:
            legend_pos = st.radio(
                "Legend position",
                ["Outside (right)", "Inside top-right", "Inside top-left",
                 "Inside bottom-right", "Inside bottom-left"],
                horizontal=False,
                disabled=not show_legend,
            )

        file_pool = corr_files if data_type == "Corrected" else norm_files
        if not file_pool:
            st.warning(f"No {'corrected' if data_type == 'Corrected' else 'normalised'} files found.")

        # ── File selection ──
        col_s1, col_s2 = st.columns([4, 1])
        with col_s1:
            selected = st.multiselect(
                "Select spectra to display",
                options=file_pool,
                default=file_pool,
            )
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Select all", width="stretch"):
                selected = file_pool

        if not selected:
            st.info("Select at least one file to display.")

        # ── Custom legend labels ──
        with st.expander("✏️ Edit legend labels", expanded=False):
            st.caption(
                "Each box corresponds to one selected spectrum. "
                "Edit the text to change its legend entry."
            )
            custom_labels: dict[str, str] = {}
            n_cols = max(min(len(selected), 3), 1)
            lbl_cols = st.columns(n_cols)
            for i, fname in enumerate(selected):
                default_lbl = clean_label(fname)
                custom_labels[fname] = (
                    lbl_cols[i % n_cols].text_input(
                        label=fname,
                        value=default_lbl,
                        key=f"lbl_{fname}",
                        label_visibility="collapsed",
                    ) or default_lbl
                )

        # ── Build Plotly figure ──
        PALETTE = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#f7b6d2",
        ]
        axis_font   = _axis_font
        tick_font   = _tick_font

        fig = go.Figure()
        x_label = "Wavelength (nm)" if x_axis == "Wavelength (nm)" else "Energy (eV)"
        y_label = ("Intensity (corrected, a.u.)"
                   if data_type == "Corrected" else "Normalised Intensity (a.u.)")

        for idx, fname in enumerate(selected):
            fpath = os.path.join(current_dir, fname)
            try:
                wl, intensity = load_corrected_txt(fpath)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
                continue

            # Apply dead pixel fix at display time so sidebar changes take effect immediately
            if dead_pixels:
                intensity = fix_dead_pixels(wl, intensity, dead_pixels)

            x_vals = wl if x_axis == "Wavelength (nm)" else wavelength_to_eV(wl)
            label  = custom_labels.get(fname, clean_label(fname))
            color  = PALETTE[idx % len(PALETTE)]
            hover_x = "λ = %{x:.2f} nm" if x_axis == "Wavelength (nm)" else "E = %{x:.4f} eV"

            fig.add_trace(go.Scatter(
                x=x_vals, y=intensity,
                mode="lines", name=label,
                line=dict(color=color, width=line_width),
                hovertemplate=(
                    f"<b>{label}</b><br>{hover_x}<br>I = %{{y:.4g}}<extra></extra>"
                ),
            ))

        # Fixed x-axis range: 350–800 nm (or equivalent in eV)
        if x_axis == "Wavelength (nm)":
            x_range = [350, 800]
        else:
            x_range = [
                float(wavelength_to_eV(np.array([800.0]))[0]),
                float(wavelength_to_eV(np.array([350.0]))[0]),
            ]

        fig.update_layout(
            xaxis=dict(
                title=dict(text=x_label, font=axis_font),
                tickfont=tick_font,
                range=x_range,
                showgrid=True, gridcolor="#eeeeee", zeroline=False,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=axis_font),
                tickfont=tick_font,
                showgrid=True, gridcolor="#eeeeee", zeroline=False,
            ),
            legend=dict(
                visible=show_legend,
                font=dict(size=max(font_size - 2, 10), family=font_family),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#cccccc",
                borderwidth=1,
                **({} if legend_pos == "Outside (right)" else {
                    "x": 0.99 if "right" in legend_pos else 0.01,
                    "y": 0.99 if "top"   in legend_pos else 0.01,
                    "xanchor": "right"  if "right" in legend_pos else "left",
                    "yanchor": "top"    if "top"   in legend_pos else "bottom",
                }),
            ),
            hovermode="x unified",
            height=560,
            margin=dict(l=70, r=30, t=30, b=70),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        _style_axes(fig)
        st.plotly_chart(fig, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

        # ── Apply dead pixels to files ──
        with st.expander("💾 Apply dead pixel correction to files", expanded=False):
            st.caption(
                "**How the dead pixel correction works in this viewer:** "
                "Any wavelengths listed in the sidebar are applied *in real-time* when rendering the plot above — "
                "the displayed curve already reflects the correction. "
                "However, the underlying `.txt` files are unchanged unless you click the button below.\n\n"
                "**Note:** Files produced by the Batch Processor already have the default "
                f"dead pixel wavelength(s) ({', '.join(str(w) for w in DEFAULT_DEAD_PIXELS)} nm) "
                "corrected and saved. Adding those same wavelengths here has no visible effect on the plot "
                "because the data is already clean. Add new wavelengths to the sidebar list to correct "
                "additional spikes that were not handled during batch processing."
            )

            # Show which wavelengths will be applied
            if dead_pixels:
                st.write(f"Dead pixels to apply: **{', '.join(f'{w} nm' for w in dead_pixels)}**")
            else:
                st.warning("No dead pixel wavelengths defined in the sidebar. Nothing to apply.")

            apply_btn = st.button(
                f"▶️ Apply to {len(selected)} selected file(s)",
                disabled=(not dead_pixels),
                type="primary",
            )

            if apply_btn and dead_pixels:
                apply_log = []
                for fname in selected:
                    fpath = os.path.join(current_dir, fname)
                    try:
                        wl, intensity = load_corrected_txt(fpath)
                        intensity_fixed = fix_dead_pixels(wl, intensity, dead_pixels)

                        # Determine header line from filename
                        if fname.endswith("_corrected_normalize.txt"):
                            header = "Wavelength Intensity_Normalized"
                        else:
                            header = "Wavelength Intensity"

                        np.savetxt(fpath,
                                   np.column_stack((wl, intensity_fixed)),
                                   fmt="%.6f", header=header, comments="")
                        apply_log.append(f"✅  {fname}")
                    except Exception as e:
                        apply_log.append(f"❌  {fname}  →  {e}")

                st.code("\n".join(apply_log), language=None)
                if all(l.startswith("✅") for l in apply_log):
                    st.success(f"Done. {len(apply_log)} file(s) updated on disk.")

        # ── Summary table ──
        with st.expander("📋 File summary", expanded=False):
            rows = []
            for fname in selected:
                fpath = os.path.join(current_dir, fname)
                try:
                    wl, intensity = load_corrected_txt(fpath)
                    if dead_pixels:
                        intensity = fix_dead_pixels(wl, intensity, dead_pixels)
                    peak_wl = wl[np.argmax(intensity)]
                    rows.append({
                        "Label":         custom_labels.get(fname, clean_label(fname)),
                        "File":          clean_label(fname),
                        "Points":        len(wl),
                        "WL range (nm)": f"{wl[0]:.1f} – {wl[-1]:.1f}",
                        "Peak WL (nm)":  f"{peak_wl:.2f}",
                        "Peak E (eV)":   f"{wavelength_to_eV(np.array([peak_wl]))[0]:.4f}",
                        "Max intensity": f"{intensity.max():.4g}",
                    })
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


    # ══════════════════════════════════════════════
    #  TAB 3 — Peak Fitting
    # ══════════════════════════════════════════════

    # ── Helper functions (peak fitting) ──────────

    def n_gaussian(x, *params):
        """Sum of N Gaussians. params = [A1, mu1, sigma1, A2, mu2, sigma2, ...]"""
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            A, mu, sigma = params[i], params[i+1], params[i+2]
            y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        return y


    def gaussian_area(A, sigma):
        return A * np.abs(sigma) * np.sqrt(2 * np.pi)


    def gaussian_fwhm(sigma):
        return 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma)


    def auto_p0_bounds(energy, intensity_lam2, peak_ev_list, sigma_init=0.05,
                       mu_window=0.15, sigma_max=0.5):
        """
        Derive p0 and bounds for N-Gaussian fit from user-supplied peak centres.

        Parameters
        ----------
        energy         : eV array (sorted ascending)
        intensity_lam2 : λ²-corrected intensity array
        peak_ev_list   : list of float, approximate peak centres in eV
        sigma_init     : initial sigma guess (eV)
        mu_window      : ±window around each peak centre for mu bounds (eV)
        sigma_max      : upper bound for sigma

        Returns
        -------
        p0     : list [A1, mu1, sigma1, A2, mu2, sigma2, ...]
        bounds : (lower_list, upper_list)
        """
        p0, lo, hi = [], [], []
        max_int = np.max(intensity_lam2)

        for mu_guess in peak_ev_list:
            # Amplitude: value of spectrum at closest point to mu_guess
            idx = int(np.argmin(np.abs(energy - mu_guess)))
            A_guess = max(float(intensity_lam2[idx]), max_int * 0.01)

            p0 += [A_guess, mu_guess, sigma_init]
            lo += [0.0,             mu_guess - mu_window, 0.001]
            hi += [max_int * 10,   mu_guess + mu_window, sigma_max]

        return p0, (lo, hi)


with tab3:
    st.header("Peak Fitting")
    st.markdown(
        "Select a corrected spectrum, enter approximate peak wavelengths "
        "(readable from the Spectrum Viewer), and the app will fit N Gaussians "
        "in energy (eV) space with λ² intensity correction."
    )

    # ── File selection ─────────────────────────
    if "fit_spectrum_dir" not in st.session_state:
        st.session_state.fit_spectrum_dir = ""
    fit_dir = st.text_input(
        "Spectrum folder path",
        value=st.session_state.fit_spectrum_dir,
        placeholder=r"e.g.  C:\data\unimplanted\spectrum",
        key="fit_dir_input",
    )
    st.session_state.fit_spectrum_dir = fit_dir

    fit_corr_files = []
    if fit_dir and os.path.isdir(fit_dir):
        fit_corr_files = sorted([
            f for f in os.listdir(fit_dir) if f.endswith("_corrected.txt")
        ])

    if not fit_corr_files:
        st.info("No corrected spectra found. Run the Batch Processor first.")

    else:
        selected_fit_file = st.selectbox(
            "Select spectrum to fit",
            options=fit_corr_files,
            format_func=clean_label,
        )

        # ── Load and prepare spectrum ───────────────
        fit_fpath = os.path.join(fit_dir, selected_fit_file)
        wl_fit, int_fit = load_corrected_txt(fit_fpath)
        if dead_pixels:
            int_fit = fix_dead_pixels(wl_fit, int_fit, dead_pixels)

        # Convert to eV, sort ascending, apply λ² correction
        ev_fit      = wavelength_to_eV(wl_fit)               # eV
        sort_idx    = np.argsort(ev_fit)
        ev_fit      = ev_fit[sort_idx]
        wl_fit_sort = wl_fit[sort_idx]
        int_lam2    = int_fit[sort_idx] * wl_fit_sort**2     # λ² correction

        # ── Peak input ─────────────────────────────
        st.markdown("---")
        st.markdown("#### Step 1 — Enter approximate peak wavelengths")
        st.caption(
            "Use the Spectrum Viewer to visually identify peaks, "
            "then enter their approximate wavelengths in nm (comma-separated)."
        )

        col_peaks, col_sigma = st.columns([3, 1])
        with col_peaks:
            peaks_input = st.text_input(
                "Approximate peak wavelengths (nm)",
                placeholder="e.g.  362, 460, 565",
                key="peaks_input",
            )
        with col_sigma:
            sigma_init = st.number_input(
                "Initial sigma (eV)",
                min_value=0.001, max_value=0.5,
                value=0.05, step=0.005, format="%.3f",
                help="Starting width guess for each Gaussian.",
            )

        mu_window = st.slider(
            "Peak centre search window ± (eV)",
            min_value=0.02, max_value=0.40,
            value=0.15, step=0.01,
            help="Bounds on how far the fitted centre can move from your initial guess.",
        )

        # Parse user input
        peak_nm_list = []
        if peaks_input.strip():
            for tok in re.split(r'[,\s]+', peaks_input.strip()):
                try:
                    peak_nm_list.append(float(tok))
                except ValueError:
                    pass
        peak_ev_list = [float(wavelength_to_eV(np.array([nm]))[0]) for nm in peak_nm_list]

        # ── Preview plot (always visible) ──────────
        st.markdown("---")
        st.markdown("#### Step 2 — Preview spectrum & peak positions")

        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(
            x=ev_fit, y=int_lam2,
            mode="lines", name="Spectrum (λ²-corrected)",
                line=dict(color="#1f77b4", width=line_width),
            hovertemplate="E = %{x:.4f} eV<br>I·λ² = %{y:.4g}<extra></extra>",
        ))

        # Mark user-supplied peak positions as vertical dashed lines
        for nm, ev in zip(peak_nm_list, peak_ev_list):
            fig_prev.add_vline(
                x=ev, line_dash="dash", line_color="red", line_width=1.5,
                annotation_text=f"{nm:.0f} nm<br>{ev:.3f} eV",
                annotation_position="top",
                annotation_font_size=11,
            )

        # (font dicts defined globally before tabs)

        fig_prev.update_layout(
            xaxis=dict(title=dict(text="Energy (eV)", font=_axis_font), tickfont=_tick_font,
                       showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title=dict(text="Intensity · λ² (a.u.)", font=_axis_font), tickfont=_tick_font,
                       showgrid=True, gridcolor="#eeeeee"),
            legend=dict(font=dict(size=max(font_size-2,10))),
            height=400, margin=dict(l=70, r=30, t=30, b=60),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        _style_axes(fig_prev)
        st.plotly_chart(fig_prev, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

        with st.expander("ℹ️ Why is intensity multiplied by λ²?", expanded=False):
            st.markdown(
                "The corrected spectra record intensity as a function of wavelength $I(\\lambda)$. "
                "To correctly compare spectral shapes in **energy space**, the Jacobian of the "
                "coordinate transformation must be applied:"
            )
            st.latex(r"I(\hbar\omega) \propto I(\lambda)\cdot\lambda^2")
            st.markdown(
                "This converts $I(\\lambda)$ — which represents intensity **per unit wavelength** — "
                "into $I(\\hbar\\omega)$ — intensity **per unit photon energy**. Without this correction, "
                "peaks at longer wavelengths (lower energy) would appear artificially broader and taller "
                "when plotted on an energy axis."
            )
            st.caption(
                "This λ² correction is applied here (Tab 3) and in Tab 4 Steps 1–2. "
                "Tab 4 Step 3 (rate equation fitting) applies an additional factor of λ, "
                "giving λ³ total, to further convert from radiated power to photon numbers."
            )
        st.markdown("---")
        st.markdown("#### Step 3 — Run fit")

        if not peak_ev_list:
            st.info("Enter at least one peak wavelength above to enable fitting.")

        # Derive p0 and bounds
        p0_auto, bounds_auto = auto_p0_bounds(
            ev_fit, int_lam2, peak_ev_list,
            sigma_init=sigma_init, mu_window=mu_window
        )

        # Advanced: manual override of p0
        with st.expander("⚙️ Advanced: view / edit initial parameters (p0)", expanded=False):
            st.caption(
                "Each row = one Gaussian component. "
                "Edit values if auto-derived p0 needs adjustment. "
                "Click **Reset to auto** to restore defaults."
            )
            # Version counter: incrementing this changes the editor key → forces full re-render
            p0_key      = f"p0__{selected_fit_file}__{len(peak_ev_list)}"
            ver_key     = f"p0_ver__{p0_key}"
            if ver_key not in st.session_state:
                st.session_state[ver_key] = 0

            def _make_p0_rows():
                return [
                    {"Peak": f"Peak {i+1} (~{peak_nm_list[i]:.0f} nm)",
                     "A (amplitude)":  round(p0_auto[i*3],   4),
                     "μ (centre, eV)": round(p0_auto[i*3+1], 4),
                     "σ (width, eV)":  round(p0_auto[i*3+2], 4)}
                    for i in range(len(peak_ev_list))
                ]

            reset_btn = st.button("↺ Reset to auto", key="reset_p0")
            if reset_btn:
                st.session_state[p0_key] = _make_p0_rows()
                st.session_state[ver_key] += 1   # change key → Streamlit treats editor as brand new

            if p0_key not in st.session_state:
                st.session_state[p0_key] = _make_p0_rows()

            # Editor key includes version number: changes on reset, forcing fresh widget
            p0_editor_key = f"p0_editor__{p0_key}_v{st.session_state[ver_key]}"
            p0_df  = pd.DataFrame(st.session_state[p0_key])
            edited = st.data_editor(
                p0_df,
                width="stretch",
                hide_index=True,
                disabled=["Peak"],
                key=p0_editor_key,
            )

            # Read back edited values
            p0_use = []
            for _, row in edited.iterrows():
                p0_use += [float(row["A (amplitude)"]),
                           float(row["μ (centre, eV)"]),
                           float(row["σ (width, eV)"])]
        # If expander was never opened, use auto p0
        if not p0_use:
            p0_use = p0_auto

        fit_btn = st.button("▶️ Run fit", type="primary", width="stretch")

        if fit_btn:
            with st.spinner("Fitting…"):
                try:
                    popt, pcov = curve_fit(
                        n_gaussian, ev_fit, int_lam2,
                        p0=p0_use, bounds=bounds_auto,
                        maxfev=8000,
                    )
                    fit_ok = True
                except Exception as fit_err:
                    st.error(
                        f"Fitting failed: {fit_err}\n\n"
                        "Try adjusting the peak positions or widening the search window."
                    )
                    fit_ok = False

            if fit_ok:
                st.session_state[f"fit_result__{selected_fit_file}"] = {
                    "popt": popt, "peak_nm": peak_nm_list,
                    "ev_fit": ev_fit, "int_lam2": int_lam2,
                }

        # ── Show fit result (persists after clicking other widgets) ──
        fit_result_key = f"fit_result__{selected_fit_file}"
        if fit_result_key in st.session_state:
            res      = st.session_state[fit_result_key]
            popt     = res["popt"]
            ev_plot  = res["ev_fit"]
            il2_plot = res["int_lam2"]

            # Legend position control
            fit_legend_pos = st.radio(
                "Legend position",
                ["Outside (right)", "Inside top-right", "Inside top-left",
                 "Inside bottom-right", "Inside bottom-left"],
                horizontal=True, key="fit_legend_pos",
            )
            COMP_COLORS = ["#d62728","#ff7f0e","#2ca02c","#9467bd",
                           "#8c564b","#e377c2","#17becf","#bcbd22"]

            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(
                x=ev_plot, y=il2_plot,
                mode="lines", name="Data",
                line=dict(color="#1f77b4", width=line_width),
                hovertemplate="E=%{x:.4f} eV  I·λ²=%{y:.4g}<extra></extra>",
            ))

            # Total fit
            y_total = n_gaussian(ev_plot, *popt)
            fig_fit.add_trace(go.Scatter(
                x=ev_plot, y=y_total,
                mode="lines", name="Total fit",
                line=dict(color="black", width=line_width, dash="dash"),
                hovertemplate="Fit: %{y:.4g}<extra></extra>",
            ))

            # Individual components + results table
            result_rows = []
            n_peaks = len(popt) // 3
            for i in range(n_peaks):
                A, mu, sigma = popt[i*3], popt[i*3+1], popt[i*3+2]
                comp = A * np.exp(-(ev_plot - mu)**2 / (2 * sigma**2))
                color = COMP_COLORS[i % len(COMP_COLORS)]
                label = f"Peak {i+1} ({1239.84/mu:.0f} nm)"
                # Convert hex color to rgba for fill
                if color.startswith("#"):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    fill_color = f"rgba({r},{g},{b},0.15)"
                else:
                    fill_color = color
                fig_fit.add_trace(go.Scatter(
                    x=ev_plot, y=comp,
                    mode="lines", name=label,
                    line=dict(color=color, width=line_width),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    hovertemplate=f"{label}: %{{y:.4g}}<extra></extra>",
                ))
                result_rows.append({
                    "Peak":           f"Peak {i+1}",
                    "Centre (eV)":    f"{mu:.4f}",
                    "Centre (nm)":    f"{1239.84/mu:.2f}",
                    "sigma (eV)":     f"{sigma:.4f}",
                    "FWHM (eV)":      f"{gaussian_fwhm(sigma):.4f}",
                    "Area (a.u.)":    f"{gaussian_area(A, sigma):.4g}",
                    "Amplitude":      f"{A:.4g}",
                })

            fig_fit.update_layout(
                xaxis=dict(title=dict(text="Energy (eV)", font=_axis_font), tickfont=_tick_font,
                           showgrid=True, gridcolor="#eeeeee"),
                yaxis=dict(title=dict(text="Intensity · λ² (a.u.)", font=_axis_font), tickfont=_tick_font,
                           showgrid=True, gridcolor="#eeeeee"),
                legend=dict(
                    font=dict(size=max(font_size-2, 10), family=font_family),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#cccccc", borderwidth=1,
                    **({} if fit_legend_pos == "Outside (right)" else {
                        "x":       0.99 if "right" in fit_legend_pos else 0.01,
                        "y":       0.99 if "top"   in fit_legend_pos else 0.01,
                        "xanchor": "right"  if "right" in fit_legend_pos else "left",
                        "yanchor": "top"    if "top"   in fit_legend_pos else "bottom",
                    }),
                ),
                height=500,
                margin=dict(l=70, r=30 if fit_legend_pos != "Outside (right)" else 30, t=30, b=60),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            _style_axes(fig_fit)
            st.plotly_chart(fig_fit, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

            # Formula reference
            with st.expander("📐 Formula reference", expanded=False):
                st.markdown("**N-Gaussian model** (fit is performed in energy space with λ² intensity correction):")
                st.latex(
                    r"I_{\text{fit}}(E) = \sum_{i=1}^{N} A_i \exp\!\left(-\frac{(E - \mu_i)^2}{2\sigma_i^2}\right)"
                )
                st.markdown("""
    | Parameter | Symbol | Meaning |
    |-----------|--------|---------|
    | Amplitude | *A* | Peak height of the Gaussian component |
    | Centre | *μ* | Peak position in eV (also shown in nm) |
    | Width | *σ* | Standard deviation of the Gaussian (eV) |
    | FWHM | — | Full width at half maximum = 2√(2 ln 2) · σ ≈ 2.355 σ |
    | Area | — | Integrated intensity = A · σ · √(2π) |
    """)
                st.markdown("**Coefficient of determination R²:**")
                st.latex(
                    r"R^2 = 1 - \frac{\displaystyle\sum_{j}(y_j - \hat{y}_j)^2}{\displaystyle\sum_{j}(y_j - \bar{y})^2}"
                )
                st.caption("y = measured I·λ²,  ŷ = fitted value,  ȳ = mean of measured values.  "
                           "R² = 1 indicates a perfect fit.")

            # Compute R² here so it's available for both table and residuals
            residuals_all = il2_plot - n_gaussian(ev_plot, *popt)
            ss_res_all = np.sum(residuals_all**2)
            ss_tot_all = np.sum((il2_plot - np.mean(il2_plot))**2)
            r2 = 1 - ss_res_all / ss_tot_all

            # Results table — append R² as a summary row
            results_df = pd.DataFrame(result_rows)
            st.markdown("##### Fit parameters")
            st.dataframe(results_df, width="stretch", hide_index=True)
            st.caption(f"Overall fit quality:  **R² = {r2:.6f}**")

            # Residuals
            with st.expander("📉 Residuals", expanded=False):
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=ev_plot, y=residuals_all, mode="lines",
                    line=dict(color="#7f7f7f", width=line_width),
                    hovertemplate="E=%{x:.4f} eV  Δ=%{y:.4g}<extra></extra>",
                    name="Residual",
                ))
                fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
                fig_res.update_layout(
                    xaxis=dict(title="Energy (eV)", tickfont=_tick_font),
                    yaxis=dict(title="Residual", tickfont=_tick_font),
                    height=250, margin=dict(l=70, r=30, t=20, b=50),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                _style_axes(fig_res)
                st.plotly_chart(fig_res, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

            # Excel download (.xlsx — avoids CSV Unicode encoding issues in Excel)
            st.markdown("---")
            import io
            xlsx_stem = clean_label(selected_fit_file)
            # Add R² as a separate summary row at the bottom
            summary_row = pd.DataFrame([{
                "Peak": "—", "Centre (eV)": "—", "Centre (nm)": "—",
                "sigma (eV)": "—", "FWHM (eV)": "—",
                "Area (a.u.)": "—", "Amplitude": f"R² = {r2:.6f}",
            }])
            export_df = pd.concat([results_df, summary_row], ignore_index=True)
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Fit Parameters")
            xlsx_bytes = xlsx_buf.getvalue()
            st.download_button(
                label="⬇️ Download fit parameters (.xlsx)",
                data=xlsx_bytes,
                file_name=f"{xlsx_stem}_fit_params.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


    # ══════════════════════════════════════════════
    #  TAB 4 — Power Series Analysis
    # ══════════════════════════════════════════════

    # ── Power series helper functions ────────────

    def load_laser_power_txt(filepath: str) -> dict:
        """
        Load laser power.txt → {label: power_percent}.
        Format: header line, then rows like "ND100  100"
        """
        data = np.loadtxt(filepath, skiprows=1, dtype=str)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return {row[0]: float(row[1]) for row in data}


    def find_laser_power_file(folder: str) -> str | None:
        """Search for laser power.txt in folder, then parent folder."""
        for search_dir in [folder, os.path.dirname(folder)]:
            candidate = os.path.join(search_dir, "laser power.txt")
            if os.path.isfile(candidate):
                return candidate
        return None


    def extract_label(filename: str) -> str:
        """Extract the first whitespace-delimited token of the filename stem."""
        stem = Path(filename).stem
        # Remove _corrected or _corrected_normalize suffix
        stem = stem.replace("_corrected_normalize", "").replace("_corrected", "")
        return stem.split()[0] if stem.split() else stem


    def batch_peak_fit_lam2(filepaths: list, peak_ev_list: list,
                            sigma_init: float, mu_window: float,
                            dead_px: list) -> list:
        """
        Fit N Gaussians on each file using λ² intensity correction (for log-log plot).
        Returns list of dicts: {filename, label, ev, il2, popt, areas, r2, error}
        """
        results = []
        for fpath in filepaths:
            fname = os.path.basename(fpath)
            try:
                wl, intensity = load_corrected_txt(fpath)
                if dead_px:
                    intensity = fix_dead_pixels(wl, intensity, dead_px)
                ev   = wavelength_to_eV(wl)
                sidx = np.argsort(ev)
                ev   = ev[sidx]
                wls  = wl[sidx]
                il2  = intensity[sidx] * wls**2   # λ² correction

                p0, bounds_fit = auto_p0_bounds(ev, il2, peak_ev_list,
                                                sigma_init=sigma_init,
                                                mu_window=mu_window)
                popt, _ = curve_fit(n_gaussian, ev, il2,
                                    p0=p0, bounds=bounds_fit, maxfev=8000)

                areas = [gaussian_area(popt[i*3], popt[i*3+2])
                         for i in range(len(peak_ev_list))]

                y_pred = n_gaussian(ev, *popt)
                ss_res = np.sum((il2 - y_pred)**2)
                ss_tot = np.sum((il2 - np.mean(il2))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                results.append(dict(filename=fname, label=extract_label(fname),
                                    ev=ev, il2=il2, popt=popt,
                                    areas=areas, r2=r2, error=None))
            except Exception as e:
                results.append(dict(filename=fname, label=extract_label(fname),
                                    ev=None, il2=None, popt=None,
                                    areas=None, r2=None, error=str(e)))
        return results


    def loglog_linfit(log_x, log_y):
        """Linear fit in log-log space, returns (slope, intercept, r2)."""
        from scipy.stats import linregress
        slope, intercept, r, _, _ = linregress(log_x, log_y)
        return slope, intercept, r**2


with tab4:
    st.header("Power Series Analysis")
    st.markdown(
        "Analyse a set of corrected spectra collected at different laser powers. "
        "Steps: batch peak fitting → log-log plot → rate equation fitting."
    )

    # ── Path + laser power file ─────────────────
    if "ps_spectrum_dir" not in st.session_state:
        st.session_state.ps_spectrum_dir = ""
    ps_dir = st.text_input(
        "Spectrum folder (power series)",
        value=st.session_state.ps_spectrum_dir,
        placeholder=r"e.g.  C:\data\unimplanted\spectrum",
        key="ps_dir_input",
    )
    st.session_state.ps_spectrum_dir = ps_dir

    # Laser power file detection
    laser_power_path = None
    laser_power_map  = {}
    if ps_dir and os.path.isdir(ps_dir):
        detected = find_laser_power_file(ps_dir)
        if detected:
            laser_power_path = detected
            try:
                laser_power_map = load_laser_power_txt(laser_power_path)
            except Exception as lp_err:
                st.warning(f"Found laser power.txt but could not read it: {lp_err}")

    col_lp1, col_lp2 = st.columns([3, 1])
    with col_lp1:
        if laser_power_path:
            st.success(f"✅ laser power.txt detected: `{laser_power_path}` "
                       f"({len(laser_power_map)} entries)")
        else:
            manual_lp = st.text_input(
                "laser power.txt path (not auto-detected — enter manually)",
                placeholder=r"e.g.  C:\data\unimplanted\laser power.txt",
                key="manual_lp_path",
            )
            if manual_lp and os.path.isfile(manual_lp):
                try:
                    laser_power_map = load_laser_power_txt(manual_lp)
                    laser_power_path = manual_lp
                    st.success(f"✅ Loaded: {len(laser_power_map)} entries")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    # Discover corrected files
    ps_corr_files = []
    if ps_dir and os.path.isdir(ps_dir):
        ps_corr_files = sorted([
            f for f in os.listdir(ps_dir) if f.endswith("_corrected.txt")
        ])

    if not ps_corr_files:
        st.info("No `_corrected.txt` files found in the specified folder.")

    else:
        st.caption(f"Found **{len(ps_corr_files)}** corrected spectrum file(s).")
        st.markdown("---")

        # ── Step 1 — Batch Peak Fitting ─────────────
        st.markdown("### Step 1 — Batch Peak Fitting")

        col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
        with col_p1:
            ps_peaks_input = st.text_input(
                "Approximate peak wavelengths (nm, comma-separated)",
                placeholder="e.g.  362, 460, 560  or  460, 560",
                key="ps_peaks_input",
            )
        with col_p2:
            ps_sigma = st.number_input("Initial σ (eV)", min_value=0.001,
                                       max_value=0.5, value=0.05, step=0.005,
                                       format="%.3f", key="ps_sigma")
        with col_p3:
            ps_mu_window = st.number_input("Search window ± (eV)", min_value=0.02,
                                           max_value=0.40, value=0.15, step=0.01,
                                           format="%.2f", key="ps_mu_window")

        ps_peak_nm_list, ps_peak_ev_list = [], []
        if ps_peaks_input.strip():
            for tok in re.split(r'[,\s]+', ps_peaks_input.strip()):
                try:
                    nm = float(tok)
                    ps_peak_nm_list.append(nm)
                    ps_peak_ev_list.append(float(wavelength_to_eV(np.array([nm]))[0]))
                except ValueError:
                    pass

        # Peak mode label (for display)
        if len(ps_peak_nm_list) == 3:
            ps_mode = "Triple peak (UVB + BB + YB)"
            ps_band_names = ["UVB", "BB", "YB"]
        elif len(ps_peak_nm_list) == 2:
            ps_mode = "Twin peak (BB + YB)"
            ps_band_names = ["BB", "YB"]
        elif len(ps_peak_nm_list) == 1:
            ps_mode = "Single peak"
            ps_band_names = ["Peak 1"]
        else:
            ps_band_names = [f"Peak {i+1}" for i in range(len(ps_peak_nm_list))]
            ps_mode = f"{len(ps_peak_nm_list)}-peak"

        if ps_peak_nm_list:
            st.caption(f"Mode: **{ps_mode}** — peaks at "
                       f"{', '.join(f'{nm:.0f} nm ({ev:.3f} eV)' for nm, ev in zip(ps_peak_nm_list, ps_peak_ev_list))}")

        run_batch_btn = st.button(
            "▶️ Run batch peak fitting",
            disabled=(not ps_peak_ev_list),
            type="primary",
            width="stretch",
            key="run_batch_ps",
        )

        ps_fit_key = f"ps_fit_results__{ps_dir}"

        if run_batch_btn and ps_peak_ev_list:
            filepaths = [os.path.join(ps_dir, f) for f in ps_corr_files]
            prog = st.progress(0, text="Fitting spectra…")
            ps_results = []
            for i, fpath in enumerate(filepaths):
                prog.progress((i) / len(filepaths),
                              text=f"Fitting {i+1}/{len(filepaths)}: {os.path.basename(fpath)}")
                res = batch_peak_fit_lam2([fpath], ps_peak_ev_list,
                                          ps_sigma, ps_mu_window, dead_pixels)
                ps_results.extend(res)
            prog.progress(1.0, text="Done!")
            st.session_state[ps_fit_key] = ps_results
            # Store peak info alongside
            st.session_state[f"ps_peaks__{ps_dir}"] = {
                "nm": ps_peak_nm_list, "ev": ps_peak_ev_list,
                "names": ps_band_names, "mode": ps_mode,
            }

        # Show results table and file selector
        if ps_fit_key in st.session_state:
            ps_results   = st.session_state[ps_fit_key]
            ps_peak_info = st.session_state.get(f"ps_peaks__{ps_dir}", {})
            band_names   = ps_peak_info.get("names", ps_band_names)

            # Build display table
            table_rows = []
            for r in ps_results:
                label = r["label"]
                power = laser_power_map.get(label, None)
                power_str = f"{power:.3g}" if power is not None else "—"
                row = {"Include": r["error"] is None,
                       "File": r["filename"],
                       "Label": label,
                       "Power (%)": power_str,
                       "R²": f"{r['r2']:.4f}" if r["r2"] is not None else "—"}
                if r["areas"] is not None:
                    for bn, area in zip(band_names, r["areas"]):
                        row[f"Area {bn}"] = f"{area:.4g}"
                else:
                    row["Error"] = r["error"]
                table_rows.append(row)

            st.markdown("##### Batch fitting results — uncheck files to exclude from plots")
            result_df = pd.DataFrame(table_rows)
            edited_result = st.data_editor(
                result_df,
                width="stretch",
                hide_index=True,
                disabled=[c for c in result_df.columns if c != "Include"],
                key="ps_result_editor",
            )

            # Warn about low R²
            low_r2 = [r["filename"] for r in ps_results
                      if r["r2"] is not None and r["r2"] < 0.90]
            if low_r2:
                st.warning(f"⚠️ {len(low_r2)} file(s) have R² < 0.90 — consider unchecking them: "
                           + ", ".join(low_r2))

            st.markdown("---")

            # ── Step 2 — Log-log Plot ─────────────────
            st.markdown("### Step 2 — Log-log Plot")

            # Filter to included files with valid fits and known power
            included_mask = edited_result["Include"].values
            included_results = [r for r, inc in zip(ps_results, included_mask) if inc]
            valid_results = []
            for r in included_results:
                if r["areas"] is None:
                    continue
                label = r["label"]
                power = laser_power_map.get(label, None)
                if power is None:
                    # Try numeric parse from label (e.g. "ND100" → 100)
                    try:
                        power = float(label.replace("ND", "").replace("nd", ""))
                    except ValueError:
                        continue
                if power <= 0:
                    continue
                valid_results.append({**r, "power": power})

            if not valid_results:
                st.info("No valid files with known laser power. "
                        "Check that laser power.txt is loaded and labels match.")
            else:
                # Sort by power
                valid_results.sort(key=lambda x: x["power"])

                # Build power and area arrays per band — always plot all bands individually
                powers_arr = np.array([r["power"] for r in valid_results])
                log_p = np.log10(powers_arr)

                plot_bands = {
                    bn: np.array([r["areas"][i] for r in valid_results])
                    for i, bn in enumerate(band_names)
                }

                BAND_COLORS = {
                    "UVB":    "#9467bd",
                    "BB":     "#1f77b4",
                    "UVB+BB": "#1f77b4",
                    "YB":     "#ff7f0e",
                    "Peak 1": "#1f77b4",
                    "Peak 2": "#ff7f0e",
                    "Peak 3": "#2ca02c",
                }

                col_ll_a, col_ll_b, col_ll_c = st.columns(3)
                with col_ll_a:
                    ll_legend_pos = st.radio(
                        "Legend position",
                        ["Outside (right)", "Inside top-right", "Inside top-left",
                         "Inside bottom-right", "Inside bottom-left"],
                        horizontal=False, key="ll_legend_pos",
                    )
                with col_ll_c:
                    ll_label_size = st.slider(
                        "Data label font size", min_value=6, max_value=16,
                        value=10, step=1, key="ll_label_size",
                    )
                fig_ll = go.Figure()
                ll_fit_summary = []

                # Collect band data first, then decide which band gets labels
                band_data = {}
                for bn, areas in plot_bands.items():
                    valid_mask = areas > 0
                    if valid_mask.sum() < 2:
                        continue
                    lp_v     = log_p[valid_mask]
                    la_v     = np.log10(areas[valid_mask])
                    labels_v = [r["label"] for r, vm in zip(valid_results, valid_mask) if vm]
                    band_data[bn] = dict(lp=lp_v, la=la_v, labels=labels_v,
                                         spread=la_v.max() - la_v.min(),
                                         valid_mask=valid_mask)

                # Label only the band with the largest y-spread (most room between points)
                label_band = max(band_data, key=lambda b: band_data[b]["spread"]) if band_data else None

                for bn, bd in band_data.items():
                    lp_v, la_v, labels_v = bd["lp"], bd["la"], bd["labels"]
                    color = BAND_COLORS.get(bn, "#333333")

                    # Scatter
                    fig_ll.add_trace(go.Scatter(
                        x=lp_v, y=la_v, mode="markers",
                        name=f"{bn} Data",
                        marker=dict(color=color, size=9,
                                    symbol="circle" if "BB" in bn or "UVB" in bn else "square"),
                        text=labels_v,
                        hovertemplate=(f"<b>{bn}</b><br>Label: %{{text}}<br>"
                                       "log₁₀(P)=%{x:.3f}<br>log₁₀(I)=%{y:.3f}<extra></extra>"),
                    ))

                    # Linear fit
                    slope, intercept, r2_ll = loglog_linfit(lp_v, la_v)
                    x_fit = np.linspace(lp_v.min(), lp_v.max(), 100)
                    y_fit = slope * x_fit + intercept
                    fig_ll.add_trace(go.Scatter(
                        x=x_fit, y=y_fit, mode="lines",
                        name=f"{bn} Fit (slope={slope:.2f}, R²={r2_ll:.3f})",
                        line=dict(color=color, width=line_width),
                        hovertemplate=f"Fit: %{{y:.3f}}<extra></extra>",
                    ))

                    # Only label the most-spacious band; others share the same x-positions
                    if bn == label_band:
                        for k, (xi, yi, lbl) in enumerate(zip(lp_v, la_v, labels_v)):
                            sign = 1 if k % 2 == 0 else -1
                            yshift = sign * ll_label_size * 1.4
                            fig_ll.add_annotation(
                                x=xi, y=yi,
                                text=lbl, showarrow=False,
                                yshift=yshift,
                                font=dict(size=ll_label_size, color="#444444",
                                          family=font_family),
                            )

                    ll_fit_summary.append({"Band": bn, "Slope": f"{slope:.3f}",
                                           "R² (fit)": f"{r2_ll:.4f}",
                                           "N points": int(bd["valid_mask"].sum())})

                fig_ll.update_layout(
                    xaxis=dict(title=dict(text="log₁₀ (Laser power percent [%])", font=_axis_font),
                               tickfont=_tick_font, showgrid=True, gridcolor="#eeeeee"),
                    yaxis=dict(title=dict(text="log₁₀ (I·λ² integration [a.u.])", font=_axis_font),
                               tickfont=_tick_font, showgrid=True, gridcolor="#eeeeee"),
                    legend=dict(
                        font=dict(size=max(font_size-2,10), family=font_family),
                        bgcolor="rgba(255,255,255,0.88)",
                        bordercolor="#cccccc", borderwidth=1,
                        **({} if ll_legend_pos == "Outside (right)" else {
                            "x":       0.99 if "right" in ll_legend_pos else 0.01,
                            "y":       0.99 if "top"   in ll_legend_pos else 0.01,
                            "xanchor": "right"  if "right" in ll_legend_pos else "left",
                            "yanchor": "top"    if "top"   in ll_legend_pos else "bottom",
                        }),
                    ),
                    height=520, margin=dict(l=70, r=30, t=30, b=60),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                _style_axes(fig_ll)
                st.plotly_chart(fig_ll, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

                if ll_fit_summary:
                    st.markdown("##### Log-log linear fit summary")
                    st.dataframe(pd.DataFrame(ll_fit_summary),
                                 width="stretch", hide_index=True)

                # Save to session state for Step 3
                st.session_state[f"ps_valid_results__{ps_dir}"] = valid_results
                st.session_state[f"ps_ll_summary__{ps_dir}"] = ll_fit_summary  # for Tab 8

                st.markdown("---")

                # ── Step 3 — Rate Equation Fitting ───────────────────────────────
                st.markdown("### Step 3 — Rate Equation Fitting")
                st.markdown(
                    "Fit the dual-channel rate equation model to the power-series PL data. "
                    "The model solves two coupled steady-state rate equations to extract effective "
                    "carrier lifetimes and radiative efficiencies for each emission band."
                )

                # ── Physical formula reference ────────────────────────────────────
                with st.expander("📐 Physical model & formula reference", expanded=False):
                    st.markdown("#### Parameter overview")
                    st.caption(
                        "All symbols used in the model, their physical meaning, units, and "
                        "the search bounds supplied to the differential evolution optimiser. "
                        "'derived' means the quantity is computed from the optimised parameters "
                        "after fitting and is not directly searched."
                    )
                    st.markdown("""
<style>
.param-tbl {border-collapse:collapse; width:100%; font-size:13px;}
.param-tbl th {background:#f0f2f6; text-align:left; padding:5px 10px; border-bottom:2px solid #ccc;}
.param-tbl td {padding:4px 10px; border-bottom:1px solid #eee; vertical-align:top;}
.param-tbl tr:hover td {background:#fafafa;}
</style>
<table class="param-tbl">
<tr><th>Symbol</th><th>Description</th><th>Unit</th><th>Search bounds / value</th></tr>
<tr><td>N<sub>1</sub></td><td>Excited carrier population — BB channel</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>N<sub>2</sub></td><td>Excited carrier population — YB channel</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>N<sub>defect</sub></td><td>Max allowable population of excited carriers at defect states</td><td>dimensionless</td><td>(5×10², 1×10⁴)</td></tr>
<tr><td>k<sub>r1</sub></td><td>Radiative recombination rate — BB channel</td><td>s<sup>−1</sup></td><td>(1×10⁶, 1×10⁹)</td></tr>
<tr><td>k<sub>nr1</sub></td><td>Non-radiative recombination rate — BB channel</td><td>s<sup>−1</sup></td><td>(1×10⁶, 1×10⁹)</td></tr>
<tr><td>k<sub>r2</sub></td><td>Radiative recombination rate — YB channel</td><td>s<sup>−1</sup></td><td>(1×10³, 1×10⁶)</td></tr>
<tr><td>k<sub>nr2</sub></td><td>Non-radiative recombination rate — YB channel</td><td>s<sup>−1</sup></td><td>(1×10³, 1×10⁶)</td></tr>
<tr><td>k<sub>capture</sub></td><td>Carrier capture rate from BB to YB</td><td>s<sup>−1</sup></td><td>(1×10⁴, 2×10⁵)</td></tr>
<tr><td>τ<sub>1</sub></td><td>Effective carrier lifetime — BB channel</td><td>s</td><td>derived</td></tr>
<tr><td>τ<sub>2</sub></td><td>Effective carrier lifetime — YB channel</td><td>s</td><td>derived</td></tr>
<tr><td>η<sub>BB</sub></td><td>Effective radiative efficiency — BB channel</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>η<sub>YB</sub></td><td>Effective radiative efficiency — YB channel</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>G</td><td>Generation rate of photo-generated carriers</td><td>s<sup>−1</sup></td><td>derived (= β·P)</td></tr>
<tr><td>β</td><td>Effective scaling constant relating laser power to G</td><td>s<sup>−1</sup></td><td>(1×10⁷, 5×10⁹)</td></tr>
<tr><td>P</td><td>Laser power percentage</td><td>dimensionless</td><td>input data</td></tr>
<tr><td>I<sub>BB</sub><sup>model</sup></td><td>Modelled BB PL intensity</td><td>a.u.</td><td>derived</td></tr>
<tr><td>I<sub>YB</sub><sup>model</sup></td><td>Modelled YB PL intensity</td><td>a.u.</td><td>derived</td></tr>
<tr><td>I<sub>BB</sub><sup>exp</sup></td><td>Experimental BB PL intensity</td><td>a.u.</td><td>input data</td></tr>
<tr><td>I<sub>YB</sub><sup>exp</sup></td><td>Experimental YB PL intensity</td><td>a.u.</td><td>input data</td></tr>
<tr><td>r<sub>BB</sub>(P<sub>i</sub>)</td><td>Normalised residual at the i<sup>th</sup> power percentage (BB)</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>r<sub>YB</sub>(P<sub>i</sub>)</td><td>Normalised residual at the i<sup>th</sup> power percentage (YB)</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>w<sub>i</sub></td><td>Residual weighting factor (uniform = 1)</td><td>dimensionless</td><td>1</td></tr>
<tr><td>ℒ<sub>data</sub></td><td>Weighted sum of squared residuals</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>τ<sub>1,min</sub></td><td>Lower bound of τ<sub>1</sub> — penalty boundary</td><td>s</td><td>1×10<sup>−8</sup></td></tr>
<tr><td>τ<sub>1,max</sub></td><td>Upper bound of τ<sub>1</sub> — penalty boundary</td><td>s</td><td>5×10<sup>−7</sup></td></tr>
<tr><td>τ<sub>2,min</sub></td><td>Lower bound of τ<sub>2</sub> — penalty boundary</td><td>s</td><td>1×10<sup>−5</sup></td></tr>
<tr><td>τ<sub>2,max</sub></td><td>Upper bound of τ<sub>2</sub> — penalty boundary</td><td>s</td><td>5×10<sup>−4</sup></td></tr>
<tr><td>ℒ<sub>penalty</sub></td><td>Lifetime penalty term</td><td>dimensionless</td><td>derived</td></tr>
<tr><td>α</td><td>Penalty weighting factor</td><td>dimensionless</td><td>0.05</td></tr>
<tr><td>ℒ<sub>total</sub></td><td>Total loss function</td><td>dimensionless</td><td>derived</td></tr>
</table><br>
""", unsafe_allow_html=True)
                    st.markdown("---")

                    st.markdown("#### Dual-channel steady-state rate equation model")
                    st.markdown(
                        "Under laser excitation, photo-generated carriers first populate the "
                        "near-band-edge (NBE) states associated with BB (or UVB+BB) emission. "
                        "These carriers then undergo radiative recombination, non-radiative "
                        "recombination, or are captured by defect states that subsequently "
                        "produce YB emission. The total number of available defect states is "
                        "fixed at N<sub>defect</sub>, leading to saturation of YB at high laser power.",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Rate equations (steady-state, dN/dt = 0):**")
                    st.latex(
                        r"\frac{dN_1}{dt} = G - \bigl(k_{r1} + k_{nr1} "
                        r"+ k_{\text{capture}}\,(N_{\text{defect}} - N_2)\bigr)\,N_1 = 0"
                    )
                    st.latex(
                        r"\frac{dN_2}{dt} = k_{\text{capture}}\,N_1\,(N_{\text{defect}} - N_2) "
                        r"- (k_{r2} + k_{nr2})\,N_2 = 0"
                    )
                    st.markdown(
                        "where $N_1$, $N_2$ are the excited carrier populations in the BB and YB "
                        "channels respectively, and $G = \\beta P$ is the generation rate "
                        "proportional to laser power percentage $P$."
                    )
                    st.markdown("**Modelled PL intensities:**")
                    st.latex(r"I_{\text{BB}}^{\text{model}} = k_{r1}\,N_1 \qquad I_{\text{YB}}^{\text{model}} = k_{r2}\,N_2")
                    st.markdown("**Effective carrier lifetimes:**")
                    st.latex(
                        r"\tau_1 = \frac{1}{k_{r1} + k_{nr1}} \qquad "
                        r"\tau_2 = \frac{1}{k_{r2} + k_{nr2}}"
                    )
                    st.markdown("**Effective radiative efficiencies** (intrinsic channel property, excluding inter-channel transfer):")
                    st.latex(
                        r"\eta_{\text{BB}} = \frac{k_{r1}}{k_{r1} + k_{nr1}} \qquad "
                        r"\eta_{\text{YB}} = \frac{k_{r2}}{k_{r2} + k_{nr2}}"
                    )
                    st.markdown("---")
                    st.markdown("#### Intensity correction: why multiply by λ³?")
                    st.markdown(
                        "The raw corrected spectra give intensity as a function of wavelength $I(\\lambda)$. "
                        "To obtain values proportional to **photon numbers as a function of energy**, two conversions are needed:"
                    )
                    st.latex(
                        r"I(\hbar\omega) \propto I(\lambda)\cdot\lambda^2 "
                        r"\quad\text{(Jacobian: converting }I(\lambda)\to I(\hbar\omega)\text{)}"
                    )
                    st.latex(
                        r"I_{\text{photon}} \propto \frac{I(\hbar\omega)}{\hbar\omega} \propto I(\lambda)\cdot\lambda^3 "
                        r"\quad\text{(dividing power by photon energy }\hbar\omega \propto 1/\lambda\text{)}"
                    )
                    st.caption(
                        "This λ³ correction is applied only in Step 3 (rate equation fitting). "
                        "Step 1/2 use λ² correction, which is appropriate for comparing spectral "
                        "shapes in energy space but does not convert to photon-number units."
                    )
                    st.markdown("---")
                    st.markdown("#### Loss function and optimisation")
                    st.markdown("Normalised residuals for each emission band at laser power $P_i$:")
                    st.latex(
                        r"r_{\text{BB}}(P_i) = \frac{I_{\text{BB}}^{\text{model}}(P_i) - I_{\text{BB}}^{\text{exp}}(P_i)}"
                        r"{\max\!\bigl(I_{\text{BB}}^{\text{exp}}\bigr)}"
                    )
                    st.latex(
                        r"r_{\text{YB}}(P_i) = \frac{I_{\text{YB}}^{\text{model}}(P_i) - I_{\text{YB}}^{\text{exp}}(P_i)}"
                        r"{\max\!\bigl(I_{\text{YB}}^{\text{exp}}\bigr)}"
                    )
                    st.markdown("Data loss (general form with per-point weights $w_i$; uniform weighting $w_i=1$ is used here):")
                    st.latex(
                        r"L_{\text{data}} = \sum_{i=1}^{N} w_i^2\Bigl[r_{\text{BB}}(P_i)^2 + r_{\text{YB}}(P_i)^2\Bigr]"
                    )
                    st.markdown(
                        "Lifetime penalty (prevents physically unreasonable solutions; "
                        "$\\tau_{1}\\in[1\\times10^{-8},\\,5\\times10^{-7}]$ s, "
                        "$\\tau_{2}\\in[1\\times10^{-5},\\,5\\times10^{-4}]$ s):"
                    )
                    st.latex(
                        r"P(\tau;\tau_{\min},\tau_{\max}) = \begin{cases} "
                        r"\!\left(\dfrac{\tau_{\min}-\tau}{\tau_{\min}}\right)^{\!2} & \tau < \tau_{\min} \\ "
                        r"\!\left(\dfrac{\tau-\tau_{\max}}{\tau_{\max}}\right)^{\!2} & \tau > \tau_{\max} \\ "
                        r"0 & \text{otherwise} \end{cases}"
                    )
                    st.latex(
                        r"L_{\text{total}} = L_{\text{data}} + \alpha\,\bigl[P(\tau_1)+P(\tau_2)\bigr], "
                        r"\quad \alpha = 0.05"
                    )
                    st.markdown("**Absolute residuals** (plotted in panel c):")
                    st.latex(
                        r"\Delta_{\text{BB}}(P_i) = I_{\text{BB}}^{\text{model}}(P_i) - I_{\text{BB}}^{\text{exp}}(P_i)"
                    )
                    st.markdown("**Relative residuals** (plotted in panel d):")
                    st.latex(
                        r"\delta_{\text{BB}}(P_i) = \frac{I_{\text{BB}}^{\text{model}}(P_i) - I_{\text{BB}}^{\text{exp}}(P_i)}"
                        r"{I_{\text{BB}}^{\text{exp}}(P_i)} \times 100\%"
                    )
                    st.markdown(
                        "Global optimisation uses **differential evolution** (`best1bin` strategy, "
                        "maxiter=1000, popsize=20, polish=True) repeated **5 times** with "
                        "randomised initial conditions. The parameter set with the minimum "
                        "$L_{\\text{total}}$ across all runs is taken as the optimal solution. "
                        "Mean ± std are computed over all 5 runs."
                    )
                    st.markdown("---")
                    st.markdown("#### Coefficient of determination R²")
                    st.markdown(
                        "Unlike the log-log linear fit in Step 2, the R² here measures how well "
                        "the **rate equation model** (a nonlinear physical model, not a straight line) "
                        "reproduces the experimental integrated PL intensities. It is defined "
                        "identically to the standard R², applied separately to each emission channel:"
                    )
                    st.latex(
                        r"R^2_{\text{ch}} = 1 - "
                        r"\frac{\displaystyle\sum_{i}\!\bigl(I_{\text{ch}}^{\text{exp}}(P_i)"
                        r"- I_{\text{ch}}^{\text{model}}(P_i)\bigr)^2}"
                        r"{\displaystyle\sum_{i}\!\bigl(I_{\text{ch}}^{\text{exp}}(P_i)"
                        r"- \overline{I_{\text{ch}}^{\text{exp}}}\bigr)^2}"
                    )
                    st.markdown(
                        "$R^2 = 1$ indicates a perfect model fit; $R^2 < 0$ means the model is "
                        "worse than simply predicting the mean value, which would indicate a "
                        "convergence failure or an inappropriate model."
                    )

                st.markdown("---")

                # ── λ³ re-fitting ──────────────────────────────────────────────
                st.markdown("#### λ³ intensity correction & peak separation")
                st.caption(
                    "Step 3 re-reads each spectrum from disk and applies I·λ³ correction "
                    "(converting to photon-number units; see formula panel above). "
                    "Gaussian peaks are then re-fitted using the same peak positions as Step 1."
                )

                # Retrieve peak info from Step 1
                ps_peak_info_s3 = st.session_state.get(f"ps_peaks__{ps_dir}", {})
                ps_peak_ev_s3   = ps_peak_info_s3.get("ev", [])
                ps_peak_nm_s3   = ps_peak_info_s3.get("nm", [])
                ps_band_names_s3 = ps_peak_info_s3.get("names", [])
                ps_mode_s3      = ps_peak_info_s3.get("mode", "")

                is_triple = (len(ps_peak_nm_s3) == 3)
                is_twin   = (len(ps_peak_nm_s3) == 2)

                if not ps_peak_ev_s3:
                    st.warning("Peak information not found — please run Step 1 first.")
                else:
                    st.caption(
                        f"Using peak positions from Step 1: "
                        f"{', '.join(f'{nm:.0f} nm' for nm in ps_peak_nm_s3)}  |  "
                        f"Mode: **{ps_mode_s3}**"
                    )

                    # Triple-peak: always merge UVB+BB into channel 1
                    if is_triple:
                        st.caption(
                            "ℹ️ Triple-peak mode: UVB and BB are automatically merged into "
                            "channel 1 (UVB+BB), as they share similar power dependence "
                            "(see Step 2 log-log plot). Channel 2 = YB."
                        )

                    # Determine channel labels for display
                    if is_triple:
                        ch1_label    = "UVB+BB"
                        ch1_file_key = "UV+Blue"   # used in saved txt
                    elif is_twin:
                        ch1_label    = "BB"
                        ch1_file_key = "Blue"
                    else:
                        ch1_label    = ps_band_names_s3[0] if ps_band_names_s3 else "Ch1"
                        ch1_file_key = "Blue"

                    # ── λ³ batch re-fitting function ──────────────────────────
                    def batch_peak_fit_lam3(filepaths, peak_ev_list, sigma_init, mu_window, dead_px):
                        """Re-fit all files with λ³ correction for rate equation input."""
                        results_l3 = []
                        for fpath in filepaths:
                            fname = os.path.basename(fpath)
                            try:
                                wl, intensity = load_corrected_txt(fpath)
                                if dead_px:
                                    intensity = fix_dead_pixels(wl, intensity, dead_px)
                                ev   = wavelength_to_eV(wl)
                                sidx = np.argsort(ev)
                                ev   = ev[sidx]
                                wls  = wl[sidx]
                                il3  = intensity[sidx] * wls**3   # λ³ correction

                                p0, bounds_fit = auto_p0_bounds(ev, il3, peak_ev_list,
                                                                  sigma_init=sigma_init,
                                                                  mu_window=mu_window)
                                popt, _ = curve_fit(n_gaussian, ev, il3,
                                                    p0=p0, bounds=bounds_fit, maxfev=8000)
                                areas = [gaussian_area(popt[i*3], popt[i*3+2])
                                         for i in range(len(peak_ev_list))]
                                results_l3.append(dict(filename=fname,
                                                       label=extract_label(fname),
                                                       areas=areas, error=None))
                            except Exception as e:
                                results_l3.append(dict(filename=fname,
                                                       label=extract_label(fname),
                                                       areas=None, error=str(e)))
                        return results_l3

                    # ── Rate equation model functions ─────────────────────────
                    RE_BOUNDS = [
                        (1e6,  1e9),    # k_r1
                        (1e6,  1e9),    # k_nr1
                        (1e4,  2e5),    # k_capture
                        (1e3,  1e6),    # k_r2
                        (1e3,  1e6),    # k_nr2
                        (5e2,  1e4),    # N_defect
                        (1e7,  5e9),    # beta
                    ]
                    RE_PARAM_NAMES   = ["k_r1", "k_nr1", "k_capture", "k_r2", "k_nr2", "N_defect", "beta"]
                    # HTML labels for Plotly chart (supports sub/sup tags)
                    RE_PARAM_CHART   = ["k<sub>r1</sub>", "k<sub>nr1</sub>", "k<sub>capture</sub>",
                                        "k<sub>r2</sub>", "k<sub>nr2</sub>", "N<sub>defect</sub>", "β"]
                    # Unicode labels for st.dataframe (plain text only)
                    RE_PARAM_UNICODE = ["k_r1", "k_nr1", "k_capture", "k_r2", "k_nr2", "N_defect", "beta"]

                    TAU1_MIN, TAU1_MAX = 1e-8, 5e-7   # seconds (0.01–500 ns)
                    TAU2_MIN, TAU2_MAX = 1e-5, 5e-4   # seconds (10–500 µs)
                    ALPHA_PENALTY      = 0.05

                    def _rate_eq_solve(params, P_array):
                        """Solve steady-state rate equations for each power level."""
                        k_r1, k_nr1, k_capture, k_r2, k_nr2, N_defect, beta = params
                        I_ch1, I_ch2 = [], []
                        for P in P_array:
                            G = beta * P
                            def equations(x):
                                N1, N2 = x
                                eq1 = G - (k_r1 + k_nr1 + k_capture*(N_defect - N2))*N1
                                eq2 = k_capture*N1*(N_defect - N2) - (k_r2 + k_nr2)*N2
                                return [eq1, eq2]
                            try:
                                sol = root(equations, [1e-6, 1e-6], method='hybr')
                                if sol.success and all(np.isfinite(sol.x)) and all(sol.x >= 0):
                                    N1, N2 = sol.x
                                else:
                                    N1, N2 = 0.0, 0.0
                            except Exception:
                                N1, N2 = 0.0, 0.0
                            I_ch1.append(k_r1 * N1)
                            I_ch2.append(k_r2 * N2)
                        return np.array(I_ch1), np.array(I_ch2)

                    def _penalty_range(tau, lower, upper):
                        if tau < lower:
                            return ((lower - tau) / lower)**2
                        elif tau > upper:
                            return ((tau - upper) / upper)**2
                        return 0.0

                    def _make_loss_fn(P_abs, area_ch1, area_ch2):
                        max_ch1 = np.max(area_ch1) or 1.0
                        max_ch2 = np.max(area_ch2) or 1.0
                        def loss_fn(params):
                            I_ch1_pred, I_ch2_pred = _rate_eq_solve(params, P_abs)
                            res1 = (I_ch1_pred - area_ch1) / max_ch1
                            res2 = (I_ch2_pred - area_ch2) / max_ch2
                            loss_data = np.sum(res1**2) + np.sum(res2**2)
                            k_r1, k_nr1 = params[0], params[1]
                            k_r2, k_nr2 = params[3], params[4]
                            tau1 = 1.0 / (k_r1 + k_nr1)
                            tau2 = 1.0 / (k_r2 + k_nr2)
                            penalty = (_penalty_range(tau1, TAU1_MIN, TAU1_MAX) +
                                       _penalty_range(tau2, TAU2_MIN, TAU2_MAX))
                            return loss_data + ALPHA_PENALTY * penalty
                        return loss_fn

                    # ── Optimisation settings ─────────────────────────────────
                    with st.expander("⚙️ Advanced optimisation settings", expanded=False):
                        st.caption(
                            "These control the differential evolution global optimiser. "
                            "Default values follow the settings used in the thesis analysis. "
                            "Click **↺ Reset to defaults** to restore them."
                        )
                        re_def = dict(num_runs=5, strategy="best1bin",
                                      maxiter=1000, popsize=20, polish=True)
                        if st.button("↺ Reset to defaults", key="re_reset_defaults"):
                            for k, v in re_def.items():
                                st.session_state[f"re_opt_{k}"] = v

                        col_o1, col_o2, col_o3 = st.columns(3)
                        with col_o1:
                            num_runs = st.number_input(
                                "Number of runs",
                                min_value=1, max_value=50,
                                value=st.session_state.get("re_opt_num_runs", 5),
                                step=1, key="re_opt_num_runs",
                                help="How many independent optimisation runs to perform. "
                                     "More runs improve statistical reliability of mean/std "
                                     "estimates but increase computation time linearly.",
                            )
                            re_polish = st.checkbox(
                                "Polish (local refinement)",
                                value=st.session_state.get("re_opt_polish", True),
                                key="re_opt_polish",
                                help="After the evolutionary search, apply a gradient-based "
                                     "local optimiser (L-BFGS-B) to refine the best solution. "
                                     "Usually improves accuracy at a small extra cost.",
                            )
                        with col_o2:
                            re_strategy = st.selectbox(
                                "DE strategy",
                                options=["best1bin", "best2bin", "rand1bin",
                                         "rand2bin", "currenttobest1bin", "randtobest1bin"],
                                index=["best1bin", "best2bin", "rand1bin",
                                       "rand2bin", "currenttobest1bin", "randtobest1bin"].index(
                                    st.session_state.get("re_opt_strategy", "best1bin")),
                                key="re_opt_strategy",
                                help="Mutation/crossover strategy for differential evolution. "
                                     "'best1bin' mutates from the current best candidate — "
                                     "fast convergence, good for well-behaved landscapes. "
                                     "'rand1bin' mutates from a random candidate — "
                                     "more exploratory, less likely to get stuck.",
                            )
                            re_maxiter = st.number_input(
                                "Max iterations",
                                min_value=100, max_value=10000,
                                value=st.session_state.get("re_opt_maxiter", 1000),
                                step=100, key="re_opt_maxiter",
                                help="Maximum number of generations the evolutionary algorithm "
                                     "runs before stopping (even if not converged).",
                            )
                        with col_o3:
                            re_popsize = st.number_input(
                                "Population size (per parameter)",
                                min_value=5, max_value=100,
                                value=st.session_state.get("re_opt_popsize", 20),
                                step=5, key="re_opt_popsize",
                                help="Number of candidate solutions per parameter dimension. "
                                     "Total population = popsize × 7 (number of parameters). "
                                     "Larger values explore more broadly but are slower.",
                            )

                    NUM_RUNS = num_runs

                    # ── Run optimisation ──────────────────────────────────────
                    run_re_btn = st.button(
                        "▶️ Run Rate Equation Fitting",
                        type="primary",
                        width="stretch",
                        key="run_re_btn",
                    )

                    re_result_key = f"re_result__{ps_dir}"

                    if run_re_btn:
                        # 1) λ³ re-fit
                        filepaths_s3 = [os.path.join(ps_dir, r["filename"])
                                        for r in valid_results]
                        with st.spinner("Step 3a — Re-fitting spectra with λ³ correction…"):
                            l3_results = batch_peak_fit_lam3(
                                filepaths_s3, ps_peak_ev_s3,
                                ps_sigma, ps_mu_window, dead_pixels
                            )

                        # Build (power, area_ch1, area_ch2) arrays
                        P_list, a_ch1_list, a_ch2_list = [], [], []
                        skipped = []
                        for vr, l3r in zip(valid_results, l3_results):
                            if l3r["areas"] is None:
                                skipped.append(l3r["filename"])
                                continue
                            areas = l3r["areas"]
                            if is_triple:
                                a1 = areas[0] + areas[1]   # UVB + BB (always merged)
                                a2 = areas[2]              # YB
                            else:
                                a1 = areas[0]
                                a2 = areas[1]
                            P_list.append(vr["power"])
                            a_ch1_list.append(a1)
                            a_ch2_list.append(a2)

                        if skipped:
                            st.warning(f"λ³ fit failed for {len(skipped)} file(s): {', '.join(skipped)}")

                        if len(P_list) < 3:
                            st.error("Not enough valid data points (need ≥ 3). "
                                     "Check peak positions and λ³ fitting.")
                        else:
                            P_abs_s3   = np.array(P_list)
                            area_ch1   = np.array(a_ch1_list)
                            area_ch2   = np.array(a_ch2_list)

                            # Sort by power
                            sort_idx_s3 = np.argsort(P_abs_s3)
                            P_abs_s3    = P_abs_s3[sort_idx_s3]
                            area_ch1    = area_ch1[sort_idx_s3]
                            area_ch2    = area_ch2[sort_idx_s3]

                            loss_fn = _make_loss_fn(P_abs_s3, area_ch1, area_ch2)

                            # 2) Multi-run differential evolution
                            st.markdown("**Running global optimisation…**")
                            prog_re    = st.progress(0, text="Initialising…")
                            status_re  = st.empty()
                            all_de_results = []
                            best_loss  = np.inf

                            for run_i in range(NUM_RUNS):
                                prog_re.progress(run_i / NUM_RUNS,
                                                 text=f"Run {run_i+1}/{NUM_RUNS} — optimising…")
                                de_res = differential_evolution(
                                    loss_fn, RE_BOUNDS,
                                    strategy=re_strategy, maxiter=int(re_maxiter),
                                    popsize=int(re_popsize), polish=re_polish, seed=run_i,
                                )
                                all_de_results.append(de_res)
                                if de_res.fun < best_loss:
                                    best_loss = de_res.fun
                                status_re.info(
                                    f"Run {run_i+1}/{NUM_RUNS} complete — "
                                    f"this run loss = {de_res.fun:.5f} — "
                                    f"best so far = {best_loss:.5f}"
                                )

                            prog_re.progress(1.0, text="Optimisation complete.")

                            # 3) Collect results
                            all_params_s3 = np.array([r.x for r in all_de_results])
                            best_result   = min(all_de_results, key=lambda r: r.fun)
                            params_fit    = best_result.x

                            param_means = np.mean(all_params_s3, axis=0)
                            param_stds  = np.std(all_params_s3,  axis=0, ddof=1)

                            k_r1, k_nr1, k_capture, k_r2, k_nr2, N_defect, beta = params_fit
                            I_ch1_fit, I_ch2_fit = _rate_eq_solve(params_fit, P_abs_s3)

                            tau1     = 1.0 / (k_r1 + k_nr1)
                            tau2     = 1.0 / (k_r2 + k_nr2)
                            eta_ch1  = k_r1 / (k_r1 + k_nr1)
                            eta_ch2  = k_r2 / (k_r2 + k_nr2)

                            abs_res_ch1  = I_ch1_fit - area_ch1
                            abs_res_ch2  = I_ch2_fit - area_ch2
                            rel_res_ch1  = np.where(area_ch1 > 0, abs_res_ch1/area_ch1*100, 0)
                            rel_res_ch2  = np.where(area_ch2 > 0, abs_res_ch2/area_ch2*100, 0)

                            def _r2(y_true, y_pred):
                                ss_res = np.sum((y_true - y_pred)**2)
                                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                                return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
                            r2_ch1 = _r2(area_ch1, I_ch1_fit)
                            r2_ch2 = _r2(area_ch2, I_ch2_fit)

                            # Smooth fit curve (interpolated over finer power grid)
                            P_fit_fine = np.logspace(
                                np.log10(P_abs_s3.min()), np.log10(P_abs_s3.max()), 200
                            )
                            I_ch1_fine, I_ch2_fine = _rate_eq_solve(params_fit, P_fit_fine)

                            # Save to session state
                            st.session_state[re_result_key] = dict(
                                params_fit=params_fit,
                                param_means=param_means,
                                param_stds=param_stds,
                                P_abs=P_abs_s3,
                                area_ch1=area_ch1,
                                area_ch2=area_ch2,
                                I_ch1_fit=I_ch1_fit,
                                I_ch2_fit=I_ch2_fit,
                                I_ch1_fine=I_ch1_fine,
                                I_ch2_fine=I_ch2_fine,
                                P_fit_fine=P_fit_fine,
                                abs_res_ch1=abs_res_ch1,
                                abs_res_ch2=abs_res_ch2,
                                rel_res_ch1=rel_res_ch1,
                                rel_res_ch2=rel_res_ch2,
                                r2_ch1=r2_ch1, r2_ch2=r2_ch2,
                                tau1=tau1, tau2=tau2,
                                eta_ch1=eta_ch1, eta_ch2=eta_ch2,
                                ch1_label=ch1_label,
                                ch1_file_key=ch1_file_key,
                                best_loss=best_loss,
                                num_runs_used=NUM_RUNS,
                            )

                    # ── Display results ───────────────────────────────────────
                    if re_result_key in st.session_state:
                        rr = st.session_state[re_result_key]

                        params_fit   = rr["params_fit"]
                        param_means  = rr["param_means"]
                        param_stds   = rr["param_stds"]
                        P_abs_s3     = rr["P_abs"]
                        area_ch1     = rr["area_ch1"]
                        area_ch2     = rr["area_ch2"]
                        I_ch1_fit    = rr["I_ch1_fit"]
                        I_ch2_fit    = rr["I_ch2_fit"]
                        I_ch1_fine   = rr["I_ch1_fine"]
                        I_ch2_fine   = rr["I_ch2_fine"]
                        P_fit_fine   = rr["P_fit_fine"]
                        abs_res_ch1  = rr["abs_res_ch1"]
                        abs_res_ch2  = rr["abs_res_ch2"]
                        rel_res_ch1  = rr["rel_res_ch1"]
                        rel_res_ch2  = rr["rel_res_ch2"]
                        r2_ch1       = rr["r2_ch1"]
                        r2_ch2       = rr["r2_ch2"]
                        tau1         = rr["tau1"]
                        tau2         = rr["tau2"]
                        eta_ch1      = rr["eta_ch1"]
                        eta_ch2      = rr["eta_ch2"]
                        ch1_lbl      = rr["ch1_label"]
                        ch1_fk       = rr["ch1_file_key"]
                        best_loss    = rr["best_loss"]
                        num_runs_used = rr.get("num_runs_used", NUM_RUNS)

                        # ── Plot controls ─────────────────────────────────────
                        _re_col1, _re_col2, _re_col3 = st.columns([2, 2, 1])
                        with _re_col1:
                            re_legend_pos = st.radio(
                                "Legend position",
                                ["Outside (right)", "Inside top-right", "Inside top-left",
                                 "Inside bottom-right", "Inside bottom-left"],
                                horizontal=False, key="re_legend_pos",
                            )
                        with _re_col2:
                            re_ann_size = st.slider(
                                "Annotation font size", min_value=6, max_value=16,
                                value=9, step=1, key="re_ann_size",
                            )

                        # ── Color scheme ──────────────────────────────────────
                        # Panel (b): blue/cyan for ch1, orange/red for ch2
                        CH1_EXP   = "#1f77b4"   # blue   — ch1 experiment
                        CH1_MDL   = "#17becf"   # cyan   — ch1 model & fit
                        CH2_EXP   = "#ff7f0e"   # orange — ch2 experiment
                        CH2_MDL   = "#d62728"   # red    — ch2 model & fit
                        # Panel (c) residuals: green/purple (distinct from panel b)
                        CH1_RES   = "#2ca02c"   # green
                        CH2_RES   = "#9467bd"   # purple
                        # Panel (d) relative errors: teal/sienna (distinct from both b and c)
                        CH1_REL   = "#0e7c7b"   # teal
                        CH2_REL   = "#a05000"   # dark-orange/sienna

                        # Identify params with std > mean (unstable)
                        unstable = [RE_PARAM_NAMES[i]
                                    for i in range(len(RE_PARAM_NAMES))
                                    if param_means[i] > 0 and
                                    param_stds[i] > param_means[i]]

                        # Shared log x-axis tick settings (show only decade labels)
                        _log_x_kw = dict(
                            type="log",
                            dtick=1,
                            tickformat=".3~g",
                            minor=dict(showgrid=True, gridcolor="#f5f5f5"),
                            showgrid=True, gridcolor="#eeeeee",
                            tickfont=_tick_font,
                        )

                        fig_re = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=["(a) Optimised parameters",
                                            "(b) Experimental vs model PL intensity",
                                            "(c) Absolute residuals",
                                            "(d) Relative residuals (%)"],
                            horizontal_spacing=0.13,
                            vertical_spacing=0.20,
                        )

                        # ── Panel (a): parameter bar chart ────────────────────
                        bar_colors = [
                            "firebrick" if n in unstable else "steelblue"
                            for n in RE_PARAM_NAMES
                        ]
                        fig_re.add_trace(go.Bar(
                            x=RE_PARAM_CHART,
                            y=param_means,
                            error_y=dict(type='data', array=param_stds, visible=True,
                                         color='black', thickness=1.5, width=4),
                            marker_color=bar_colors,
                            text=[f"{v:.2e}" for v in param_means],
                            textposition="outside",
                            textfont=dict(size=re_ann_size),
                            name="Parameters",
                            showlegend=False,
                            hovertemplate="<b>%{x}</b><br>Mean: %{y:.3e}<br>Std: %{error_y.array:.3e}<extra></extra>",
                        ), row=1, col=1)
                        fig_re.update_yaxes(type="log", title_text="Parameter value (log scale)",
                                            dtick=1, showgrid=True, gridcolor="#eeeeee",
                                            tickfont=_tick_font, row=1, col=1)
                        fig_re.update_xaxes(title_text="Parameter",
                                            tickfont=_tick_font, row=1, col=1)

                        # ── Panel (b): exp vs model intensity ─────────────────
                        # CH1 experimental
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=area_ch1, mode="markers",
                            name=f"{ch1_lbl} Exp",
                            legendgroup="b_ch1",
                            legendgrouptitle_text="(b) PL intensity",
                            marker=dict(color=CH1_EXP, size=9, symbol="circle"),
                            hovertemplate=f"{ch1_lbl} Exp: %{{y:.3e}}<extra></extra>",
                        ), row=1, col=2)
                        # CH1 model points
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=I_ch1_fit, mode="markers",
                            name=f"{ch1_lbl} Model",
                            legendgroup="b_ch1",
                            marker=dict(color=CH1_MDL, size=9, symbol="x"),
                            hovertemplate=f"{ch1_lbl} Model: %{{y:.3e}}<extra></extra>",
                        ), row=1, col=2)
                        # CH1 fit curve
                        fig_re.add_trace(go.Scatter(
                            x=P_fit_fine, y=I_ch1_fine, mode="lines",
                            name=f"{ch1_lbl} Fit",
                            legendgroup="b_ch1",
                            line=dict(color=CH1_MDL, dash="dash", width=line_width),
                            hovertemplate=f"{ch1_lbl} Fit: %{{y:.3e}}<extra></extra>",
                        ), row=1, col=2)
                        # CH2 experimental
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=area_ch2, mode="markers",
                            name="YB Exp",
                            legendgroup="b_ch2",
                            marker=dict(color=CH2_EXP, size=9, symbol="circle"),
                            hovertemplate="YB Exp: %{y:.3e}<extra></extra>",
                        ), row=1, col=2)
                        # CH2 model points
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=I_ch2_fit, mode="markers",
                            name="YB Model",
                            legendgroup="b_ch2",
                            marker=dict(color=CH2_MDL, size=9, symbol="x"),
                            hovertemplate="YB Model: %{y:.3e}<extra></extra>",
                        ), row=1, col=2)
                        # CH2 fit curve
                        fig_re.add_trace(go.Scatter(
                            x=P_fit_fine, y=I_ch2_fine, mode="lines",
                            name="YB Fit",
                            legendgroup="b_ch2",
                            line=dict(color=CH2_MDL, dash="dash", width=line_width),
                            hovertemplate="YB Fit: %{y:.3e}<extra></extra>",
                        ), row=1, col=2)
                        # R² annotation inside panel (b)
                        fig_re.add_annotation(
                            xref="x2", yref="paper",
                            x=0, y=0.97,
                            xanchor="left", yanchor="top",
                            text=(f"R²({ch1_lbl}) = {r2_ch1:.4f}<br>"
                                  f"R²(YB) = {r2_ch2:.4f}"),
                            showarrow=False,
                            font=dict(size=max(font_size-4, 9), family=font_family),
                            bgcolor="rgba(255,255,255,0.75)",
                            bordercolor="#aaaaaa", borderwidth=1,
                            align="left",
                        )
                        fig_re.update_xaxes(title_text="Laser Power (%)",
                                            **_log_x_kw, row=1, col=2)
                        fig_re.update_yaxes(type="log",
                                            title_text="PL Intensity (a.u.)",
                                            dtick=1,
                                            exponentformat="power",
                                            showgrid=True, gridcolor="#eeeeee",
                                            tickfont=_tick_font, row=1, col=2)

                        # ── Panel (c): absolute residuals ─────────────────────
                        sigma_ch1 = np.std(abs_res_ch1)
                        sigma_ch2 = np.std(abs_res_ch2)
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=abs_res_ch1, mode="lines+markers",
                            name=f"Residual {ch1_lbl}",
                            legendgroup="c_ch1",
                            legendgrouptitle_text="(c) Residuals",
                            line=dict(color=CH1_RES, width=line_width),
                            marker=dict(color=CH1_RES, size=7),
                            hovertemplate=f"Residual {ch1_lbl}: %{{y:.3e}}<extra></extra>",
                        ), row=2, col=1)
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=abs_res_ch2, mode="lines+markers",
                            name="Residual YB",
                            legendgroup="c_ch2",
                            line=dict(color=CH2_RES, width=line_width),
                            marker=dict(color=CH2_RES, size=7),
                            hovertemplate="Residual YB: %{{y:.3e}}<extra></extra>",
                        ), row=2, col=1)
                        # ±3σ as single trace per channel using None gap (both lines toggled together)
                        x_span = [P_abs_s3.min(), P_abs_s3.max()]
                        fig_re.add_trace(go.Scatter(
                            x=x_span + [None] + x_span,
                            y=[3*sigma_ch1, 3*sigma_ch1, None, -3*sigma_ch1, -3*sigma_ch1],
                            mode="lines", name=f"±3σ {ch1_lbl}",
                            legendgroup="c_ch1",
                            line=dict(color=CH1_RES, dash="dot", width=max(line_width*0.6, 1.0)),
                            showlegend=True,
                            hovertemplate=f"±3σ {ch1_lbl}: ±{3*sigma_ch1:.3e}<extra></extra>",
                        ), row=2, col=1)
                        fig_re.add_trace(go.Scatter(
                            x=x_span + [None] + x_span,
                            y=[3*sigma_ch2, 3*sigma_ch2, None, -3*sigma_ch2, -3*sigma_ch2],
                            mode="lines", name="±3σ YB",
                            legendgroup="c_ch2",
                            line=dict(color=CH2_RES, dash="dot", width=max(line_width*0.6, 1.0)),
                            showlegend=True,
                            hovertemplate=f"±3σ YB: ±{3*sigma_ch2:.3e}<extra></extra>",
                        ), row=2, col=1)
                        fig_re.add_hline(y=0, line_dash="dash", line_color="gray",
                                         line_width=1, row=2, col=1)
                        fig_re.update_xaxes(title_text="Laser Power (%)",
                                            **_log_x_kw, row=2, col=1)
                        fig_re.update_yaxes(title_text="Residuals (Model − Exp)",
                                            showgrid=True, gridcolor="#eeeeee",
                                            tickfont=_tick_font, row=2, col=1)

                        # ── Panel (d): relative residuals ─────────────────────
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=rel_res_ch1, mode="lines+markers",
                            name=f"{ch1_lbl} Rel Error (%)",
                            legendgroup="d_ch1",
                            legendgrouptitle_text="(d) Rel. error",
                            line=dict(color=CH1_REL, width=line_width),
                            marker=dict(color=CH1_REL, size=7),
                            hovertemplate=f"{ch1_lbl} Rel Error: %{{y:.2f}}%<extra></extra>",
                        ), row=2, col=2)
                        fig_re.add_trace(go.Scatter(
                            x=P_abs_s3, y=rel_res_ch2, mode="lines+markers",
                            name="YB Rel Error (%)",
                            legendgroup="d_ch2",
                            line=dict(color=CH2_REL, width=line_width),
                            marker=dict(color=CH2_REL, size=7),
                            hovertemplate="YB Rel Error: %{y:.2f}%<extra></extra>",
                        ), row=2, col=2)
                        fig_re.add_hline(y=0, line_dash="dash", line_color="gray",
                                         line_width=1, row=2, col=2)
                        fig_re.update_xaxes(title_text="Laser Power (%)",
                                            **_log_x_kw, row=2, col=2)
                        fig_re.update_yaxes(title_text="Relative Residual (%)",
                                            showgrid=True, gridcolor="#eeeeee",
                                            tickfont=_tick_font, row=2, col=2)

                        # ── Global layout ─────────────────────────────────────
                        _re_legend_outside = re_legend_pos == "Outside (right)"
                        _re_legend_kw = {} if _re_legend_outside else {
                            "x":       0.99 if "right" in re_legend_pos else 0.01,
                            "y":       0.99 if "top"   in re_legend_pos else 0.01,
                            "xanchor": "right"  if "right" in re_legend_pos else "left",
                            "yanchor": "top"    if "top"   in re_legend_pos else "bottom",
                        }
                        fig_re.update_layout(
                            height=820,
                            margin=dict(l=70, r=220 if _re_legend_outside else 30, t=60, b=60),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            legend=dict(
                                font=dict(size=max(font_size-4, 9), family=font_family),
                                groupclick="toggleitem",
                                bgcolor="rgba(255,255,255,0.92)",
                                bordercolor="#cccccc", borderwidth=1,
                                tracegroupgap=8,
                                **_re_legend_kw,
                            ),
                            font=dict(family=font_family, size=max(font_size-2, 10)),
                        )

                        _style_axes(fig_re)
                        st.plotly_chart(fig_re, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

                        if unstable:
                            st.warning(
                                f"⚠️ The following parameters show std > mean across "
                                f"{num_runs_used} runs, suggesting the optimisation may not have "
                                f"fully converged: **{', '.join(unstable)}**. "
                                f"Consider re-running or increasing the number of runs."
                            )

                        # ── Summary table ─────────────────────────────────────
                        st.markdown("##### Fitting results summary")
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric(f"τ₁ ({ch1_lbl})", f"{tau1*1e9:.2f} ns")
                            st.metric(f"η ({ch1_lbl})", f"{eta_ch1*100:.2f}%")
                            st.metric(f"R² ({ch1_lbl})", f"{r2_ch1:.4f}")
                        with col_s2:
                            st.metric("τ₂ (YB)", f"{tau2*1e6:.2f} µs")
                            st.metric("η (YB)", f"{eta_ch2*100:.2f}%")
                            st.metric("R² (YB)", f"{r2_ch2:.4f}")

                        # Parameter table
                        param_rows = []
                        for i, name in enumerate(RE_PARAM_UNICODE):
                            param_rows.append({
                                "Parameter":   name,
                                "Optimal":     f"{params_fit[i]:.3e}",
                                f"Mean ({num_runs_used} runs)": f"{param_means[i]:.3e}",
                                "Std":         f"{param_stds[i]:.3e}",
                                "Stable":      "✅" if RE_PARAM_NAMES[i] not in unstable else "⚠️",
                            })
                        st.dataframe(pd.DataFrame(param_rows), width="stretch", hide_index=True)
                        st.caption(f"Best total loss = {best_loss:.5f}")

                        # ── Save result txt ───────────────────────────────────
                        st.markdown("---")
                        st.markdown("##### Save results")
                        st.caption(
                            "Results are saved in a format compatible with the "
                            "**Lifetime Compare** tab (Tab 5)."
                        )

                        default_save_dir = str(
                            Path(ps_dir).parent.parent / "lifetime_results"
                        )
                        save_dir_input = st.text_input(
                            "Output folder",
                            value=default_save_dir,
                            key="re_save_dir",
                        )
                        # Infer filename from the spectrum folder name
                        default_fname = Path(ps_dir).parent.name + ".txt"
                        save_fname_input = st.text_input(
                            "File name",
                            value=default_fname,
                            key="re_save_fname",
                        )

                        save_re_btn = st.button(
                            "💾 Save result txt",
                            type="primary",
                            key="save_re_btn",
                        )

                        if save_re_btn:
                            try:
                                os.makedirs(save_dir_input, exist_ok=True)
                                save_path = os.path.join(save_dir_input, save_fname_input)
                                with open(save_path, "w", encoding="utf-8") as f_out:
                                    f_out.write("===== Rate Equation Model — Fitting Results =====\n\n")
                                    f_out.write(f"--- Optimised parameters (best of {num_runs_used} runs) ---\n")
                                    for i, name in enumerate(RE_PARAM_NAMES):
                                        f_out.write(
                                            f"{name}: {params_fit[i]:.3e}  "
                                            f"(Mean: {param_means[i]:.3e}, "
                                            f"Std: {param_stds[i]:.3e})\n"
                                        )
                                    f_out.write("\n--- Derived quantities ---\n")
                                    # τ line — format must match lifetime compare regex
                                    tau1_tag = ch1_fk   # "UV+Blue" or "Blue"
                                    f_out.write(
                                        f"τ1 ({tau1_tag}): {tau1*1e9:.2f} ns, "
                                        f"τ2 (Yellow): {tau2*1e6:.2f} µs\n"
                                    )
                                    # Efficiency lines — format must match regex
                                    eff_tag = "UVB_BB" if ch1_fk == "UV+Blue" else "BB"
                                    f_out.write(
                                        f"Radiative efficiency {eff_tag}: {eta_ch1*100:.2f}%\n"
                                    )
                                    f_out.write(
                                        f"Radiative efficiency YB: {eta_ch2*100:.2f}%\n"
                                    )
                                    f_out.write(
                                        f"R² ({ch1_lbl}): {r2_ch1:.4f}\n"
                                    )
                                    f_out.write(
                                        f"R² (YB): {r2_ch2:.4f}\n"
                                    )
                                    f_out.write(f"\nBest total loss: {best_loss:.6f}\n")
                                st.success(f"✅ Saved to `{save_path}`")
                            except Exception as save_err:
                                st.error(f"Save failed: {save_err}")

# ══════════════════════════════════════════════
#  TAB 5 — Lifetime Compare
# ══════════════════════════════════════════════
with tab5:
    st.header("Lifetime Compare")
    st.markdown(
        "Load a folder of result `.txt` files produced by **Step 3** of the Power Series tab. "
        "Edit display labels and ordering in the table, then view the 2×2 comparison plot."
    )

    if "lc_dir" not in st.session_state:
        st.session_state.lc_dir = ""
    lc_dir = st.text_input(
        "Results folder path",
        value=st.session_state.lc_dir,
        placeholder=r"e.g.  C:\data\lifetime_results",
        key="lc_dir_input",
    )
    st.session_state.lc_dir = lc_dir

    # ── Regex patterns (identical to lifetime compare.py) ────────────────
    import re as _re
    _PAT_TAU  = _re.compile(
        r"τ1\s*\((?:Blue|UV\+Blue)\)\s*[:：]\s*([\d\.]+)\s*ns[,，]?\s*"
        r"τ2\s*\(Yellow\)\s*[:：]\s*([\d\.]+)\s*(?:us|µs|μs)",
        _re.IGNORECASE,
    )
    _PAT_BB   = _re.compile(
        r"Radiative efficiency\s+(?:BB|UVB_BB)\s*[:：]\s*([\d\.]+)\s*%",
        _re.IGNORECASE,
    )
    _PAT_YB   = _re.compile(
        r"Radiative efficiency\s+YB\s*[:：]\s*([\d\.]+)\s*%",
        _re.IGNORECASE,
    )

    def _parse_result_txt(path):
        """Return (tau1_ns, tau2_us, eff_bb_pct, eff_yb_pct) or Nones on failure."""
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            return None, None, None, None
        m_tau = _PAT_TAU.search(text)
        m_bb  = _PAT_BB.search(text)
        m_yb  = _PAT_YB.search(text)
        tau1  = float(m_tau.group(1)) if m_tau else None
        tau2  = float(m_tau.group(2)) if m_tau else None
        eff_bb = float(m_bb.group(1)) if m_bb else None
        eff_yb = float(m_yb.group(1)) if m_yb else None
        return tau1, tau2, eff_bb, eff_yb

    lc_txt_files = []
    if lc_dir and os.path.isdir(lc_dir):
        lc_txt_files = sorted([f for f in os.listdir(lc_dir) if f.lower().endswith(".txt")])

    if not lc_txt_files:
        if lc_dir:
            st.info("No `.txt` files found in the specified folder.")
    else:
        st.caption(f"Found **{len(lc_txt_files)}** result file(s).")

        # ── Editable table ───────────────────────────────────────────────
        lc_table_key = f"lc_table__{lc_dir}"
        lc_version_key = f"lc_version__{lc_dir}"

        # Build default DataFrame once
        def _lc_default_df(files):
            return pd.DataFrame([{
                "Include": True, "File": f,
                "Display label": Path(f).stem,
                "X-order": float(i + 1),
            } for i, f in enumerate(files)])

        if lc_table_key not in st.session_state:
            st.session_state[lc_table_key] = _lc_default_df(lc_txt_files)
            st.session_state[lc_version_key] = 0

        lc_df_raw = st.session_state[lc_table_key]
        # Sync any new files (append only — never overwrite existing rows)
        # Increment version when data source changes so widget rebuilds cleanly
        existing_files = set(lc_df_raw["File"].tolist())
        new_rows = [{"Include": True, "File": f, "Display label": Path(f).stem,
                     "X-order": float(lc_df_raw["X-order"].max() + 1)}
                    for f in lc_txt_files if f not in existing_files]
        if new_rows:
            lc_df_raw = pd.concat([lc_df_raw, pd.DataFrame(new_rows)], ignore_index=True)
            st.session_state[lc_table_key] = lc_df_raw
            st.session_state[lc_version_key] = st.session_state.get(lc_version_key, 0) + 1

        st.markdown("##### File table — edit labels and ordering")
        st.caption(
            "**Display label**: text shown on the x-axis.  "
            "**X-order**: number controlling left-to-right position — smaller = further left. "
            "Decimals are allowed (e.g. set 1.5 to insert between positions 1 and 2). "
            "Duplicate values are resolved by filename.  "
            "**Include**: uncheck to hide a file from the plot."
        )
        # Version-keyed widget: Streamlit manages the edit diff internally.
        # Version only increments when the data source changes (new files / folder change),
        # which forces a clean rebuild. User edits within a session are preserved by the widget.
        lc_editor_key = f"lc_editor__{lc_dir}__v{st.session_state.get(lc_version_key, 0)}"
        lc_edited = st.data_editor(
            lc_df_raw,
            width="stretch",
            hide_index=True,
            disabled=["File"],
            column_config={
                "Include":       st.column_config.CheckboxColumn("Include"),
                "X-order":       st.column_config.NumberColumn("X-order", min_value=0.0, step=0.5, format="%.1f"),
                "Display label": st.column_config.TextColumn("Display label"),
                "File":          st.column_config.TextColumn("File"),
            },
            key=lc_editor_key,
        )

        # ── Parse all files ──────────────────────────────────────────────
        lc_parsed = []
        for _, row in lc_edited.iterrows():
            if not row["Include"]:
                continue
            fpath = os.path.join(lc_dir, row["File"])
            tau1, tau2, eff_bb, eff_yb = _parse_result_txt(fpath)
            lc_parsed.append({
                "label":  row["Display label"],
                "order":  row["X-order"],
                "tau1":   tau1,
                "tau2":   tau2,
                "eff_bb": eff_bb,
                "eff_yb": eff_yb,
                "file":   row["File"],
            })

        if not lc_parsed:
            st.info("No included files. Check the Include column.")
            st.session_state["t5_ai_ready"] = False
        else:
            # Sort by X-order
            lc_parsed.sort(key=lambda r: r["order"])

            # ── Publish artifact for Tab 8 AI Copilot ───────────────────────
            st.session_state["t5_ai_ready"]    = True
            st.session_state["t5_ai_dir"]      = lc_dir
            st.session_state["t5_ai_included"] = [
                {"file": r["file"], "label": r["label"]} for r in lc_parsed
            ]

            # Warn about files that didn't parse
            failed = [r["file"] for r in lc_parsed
                      if r["tau1"] is None and r["eff_bb"] is None]
            if failed:
                st.warning(
                    f"⚠️ Could not parse {len(failed)} file(s) "
                    f"(no matching lifetime/efficiency lines found): "
                    + ", ".join(failed)
                )

            labels   = [r["label"]  for r in lc_parsed]
            tau1_arr = [r["tau1"]   for r in lc_parsed]
            tau2_arr = [r["tau2"]   for r in lc_parsed]
            ebb_arr  = [r["eff_bb"] for r in lc_parsed]
            eyb_arr  = [r["eff_yb"] for r in lc_parsed]
            x_pos    = list(range(len(labels)))

            # ── Plot controls ────────────────────────────────────────────
            _lc_ctl1, _lc_ctl2, _lc_ctl3 = st.columns([2, 2, 1])
            with _lc_ctl1:
                lc_legend_pos = st.radio(
                    "Legend position",
                    ["Outside (right)", "Inside top-right", "Inside top-left",
                     "Inside bottom-right", "Inside bottom-left"],
                    horizontal=False, key="lc_legend_pos",
                )
            with _lc_ctl2:
                lc_label_size = st.slider(
                    "Data label font size", min_value=6, max_value=16,
                    value=max(font_size - 4, 9), step=1, key="lc_label_size",
                )

            # ── Helper: add value annotations — fixed offset direction ───
            def _ann(fig, x, y, yref, fontsize, color, offset_px):
                """offset_px > 0 = above the point, < 0 = below."""
                for xi, yi in zip(x, y):
                    if yi is None:
                        continue
                    fig.add_annotation(
                        x=xi, y=yi, yref=yref,
                        text=f"{yi:.2f}", showarrow=False,
                        yshift=offset_px,
                        font=dict(size=fontsize, family=font_family, color=color),
                        xanchor="center",
                    )

            _BLUE = "#1f77b4"
            _GOLD = "#e6a800"
            ann_size = lc_label_size
            _lc_x_shared = dict(
                tickvals=x_pos, ticktext=labels, tickangle=30,
                tickfont=_tick_font, showgrid=True, gridcolor="#eeeeee",
                title_text="Laser focus & ion dose", title_font=_axis_font,
            )
            _lc_legend_outside = lc_legend_pos == "Outside (right)"
            _lc_legend_kw = {} if _lc_legend_outside else {
                "x":       0.99 if "right" in lc_legend_pos else 0.01,
                "y":       0.99 if "top"   in lc_legend_pos else 0.01,
                "xanchor": "right"  if "right" in lc_legend_pos else "left",
                "yanchor": "top"    if "top"   in lc_legend_pos else "bottom",
            }
            _lc_layout_common = dict(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=70, r=160 if _lc_legend_outside else 80, t=50, b=100),
                font=dict(family=font_family, size=max(font_size-2, 10)),
                legend=dict(
                    font=dict(size=max(font_size-3, 10), family=font_family),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#cccccc", borderwidth=1,
                    **({"x": 1.12, "y": 1, "xanchor": "left"} if _lc_legend_outside
                       else _lc_legend_kw),
                ),
            )

            # ── Figure 1: Carrier lifetimes (dual Y) ─────────────────────
            fig_tau = go.Figure()
            fig_tau.add_trace(go.Scatter(
                x=x_pos, y=tau1_arr, name="τ_1 (ns)",
                mode="lines+markers",
                line=dict(color=_BLUE, width=line_width),
                marker=dict(color=_BLUE, size=8),
                yaxis="y", connectgaps=False,
                hovertemplate="τ_1: %{y:.2f} ns<extra></extra>",
            ))
            fig_tau.add_trace(go.Scatter(
                x=x_pos, y=tau2_arr, name="τ_2 (µs)",
                mode="lines+markers",
                line=dict(color=_GOLD, width=line_width),
                marker=dict(color=_GOLD, size=8),
                yaxis="y2", connectgaps=False,
                hovertemplate="τ_2: %{y:.2f} µs<extra></extra>",
            ))
            # τ_1 labels above (+12), τ_2 labels below (-14) — never overlap
            _ann(fig_tau, x_pos, tau1_arr, "y",  ann_size, _BLUE, +12)
            _ann(fig_tau, x_pos, tau2_arr, "y2", ann_size, _GOLD, -14)
            fig_tau.update_layout(
                title=dict(text="(a) Carrier lifetimes",
                           font=dict(size=font_size, family=font_family), x=0.5),
                xaxis=_lc_x_shared,
                yaxis=dict(title_text="τ_1 (ns)",
                           title_font=dict(size=font_size, family=font_family, color=_BLUE),
                           tickfont=dict(size=max(font_size-2,10), family=font_family, color=_BLUE),
                           showgrid=True, gridcolor="#eeeeee"),
                yaxis2=dict(title_text="τ_2 (µs)", overlaying="y", side="right",
                            title_font=dict(size=font_size, family=font_family, color=_GOLD),
                            tickfont=dict(size=max(font_size-2,10), family=font_family, color=_GOLD),
                            showgrid=False),
                height=420,
                **_lc_layout_common,
            )
            _style_axes(fig_tau)
            st.plotly_chart(fig_tau, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

            # ── Figure 2: Radiative efficiencies (dual Y) ─────────────────
            fig_eta = go.Figure()
            fig_eta.add_trace(go.Scatter(
                x=x_pos, y=ebb_arr, name="η_BB (%)",
                mode="lines+markers",
                line=dict(color=_BLUE, width=line_width),
                marker=dict(color=_BLUE, size=8),
                yaxis="y", connectgaps=False,
                hovertemplate="η_BB: %{y:.2f}%<extra></extra>",
            ))
            fig_eta.add_trace(go.Scatter(
                x=x_pos, y=eyb_arr, name="η_YB (%)",
                mode="lines+markers",
                line=dict(color=_GOLD, width=line_width),
                marker=dict(color=_GOLD, size=8),
                yaxis="y2", connectgaps=False,
                hovertemplate="η_YB: %{y:.2f}%<extra></extra>",
            ))
            # η_BB labels above (+12), η_YB labels below (-14) — never overlap
            _ann(fig_eta, x_pos, ebb_arr,  "y",  ann_size, _BLUE, +12)
            _ann(fig_eta, x_pos, eyb_arr,  "y2", ann_size, _GOLD, -14)
            fig_eta.update_layout(
                title=dict(text="(b) Radiative efficiencies",
                           font=dict(size=font_size, family=font_family), x=0.5),
                xaxis=_lc_x_shared,
                yaxis=dict(title_text="η_BB (%)",
                           title_font=dict(size=font_size, family=font_family, color=_BLUE),
                           tickfont=dict(size=max(font_size-2,10), family=font_family, color=_BLUE),
                           showgrid=True, gridcolor="#eeeeee"),
                yaxis2=dict(title_text="η_YB (%)", overlaying="y", side="right",
                            title_font=dict(size=font_size, family=font_family, color=_GOLD),
                            tickfont=dict(size=max(font_size-2,10), family=font_family, color=_GOLD),
                            showgrid=False),
                height=420,
                **_lc_layout_common,
            )
            _style_axes(fig_eta)
            st.plotly_chart(fig_eta, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

            # ── Parsed values table ──────────────────────────────────────
            with st.expander("📋 Parsed values", expanded=False):
                tbl_rows = []
                for r in lc_parsed:
                    tbl_rows.append({
                        "Label":       r["label"],
                        "τ₁ (ns)":    f"{r['tau1']:.2f}"  if r["tau1"]   is not None else "—",
                        "τ₂ (µs)":    f"{r['tau2']:.2f}"  if r["tau2"]   is not None else "—",
                        "η_BB (%)":   f"{r['eff_bb']:.2f}" if r["eff_bb"] is not None else "—",
                        "η_YB (%)":   f"{r['eff_yb']:.2f}" if r["eff_yb"] is not None else "—",
                        "Source file": r["file"],
                    })
                st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)


# ══════════════════════════════════════════════
#  TAB 6 — CIE Diagram
# ══════════════════════════════════════════════
with tab6:
    st.header("CIE 1931 Chromaticity Diagram")
    st.markdown(
        "Load a folder of `_corrected.txt` spectra. The app computes CIE 1931 xy chromaticity "
        "coordinates for each spectrum and plots them on the standard horseshoe diagram."
    )

    if "cie_dir" not in st.session_state:
        st.session_state.cie_dir = ""
    cie_dir = st.text_input(
        "Spectrum folder path",
        value=st.session_state.cie_dir,
        placeholder=r"e.g.  C:\data\unimplanted\spectrum",
        key="cie_dir_input",
    )
    st.session_state.cie_dir = cie_dir

    # ── CIE 1931 2° colour matching functions (10 nm table → 1 nm interp) ──
    _CIE_CMF_10NM = {
        380:(0.001368,0.000039,0.006450), 390:(0.004243,0.000120,0.020050),
        400:(0.014310,0.000396,0.067850), 410:(0.043510,0.001210,0.207400),
        420:(0.134380,0.004000,0.645600), 430:(0.283900,0.011600,1.385600),
        440:(0.348280,0.023000,1.747060), 450:(0.336200,0.038000,1.772110),
        460:(0.290800,0.060000,1.669200), 470:(0.195360,0.090980,1.287640),
        480:(0.095640,0.139020,0.812950), 490:(0.032010,0.208020,0.465180),
        500:(0.004900,0.323000,0.272000), 510:(0.009300,0.503000,0.158200),
        520:(0.063270,0.710000,0.078250), 530:(0.165500,0.862000,0.042160),
        540:(0.290400,0.954000,0.020300), 550:(0.433450,0.995000,0.008750),
        560:(0.594500,0.995000,0.003900), 570:(0.762100,0.952000,0.002100),
        580:(0.916300,0.870000,0.001650), 590:(1.026300,0.757000,0.001100),
        600:(1.062200,0.631000,0.000800), 610:(1.002600,0.503000,0.000340),
        620:(0.854450,0.381000,0.000190), 630:(0.642400,0.265000,0.000050),
        640:(0.447900,0.175000,0.000020), 650:(0.283500,0.107000,0.000000),
        660:(0.164900,0.061000,0.000000), 670:(0.087400,0.032000,0.000000),
        680:(0.046770,0.017000,0.000000), 690:(0.022700,0.008210,0.000000),
        700:(0.011359,0.004102,0.000000), 710:(0.005790,0.002091,0.000000),
        720:(0.002899,0.001047,0.000000), 730:(0.001440,0.000520,0.000000),
        740:(0.000690,0.000249,0.000000), 750:(0.000332,0.000120,0.000000),
        760:(0.000166,0.000060,0.000000), 770:(0.000083,0.000030,0.000000),
        780:(0.000042,0.000015,0.000000),
    }
    _cie_wl  = np.array(list(_CIE_CMF_10NM.keys()), dtype=float)
    _cie_xb  = np.array([v[0] for v in _CIE_CMF_10NM.values()])
    _cie_yb  = np.array([v[1] for v in _CIE_CMF_10NM.values()])
    _cie_zb  = np.array([v[2] for v in _CIE_CMF_10NM.values()])
    _WL_FINE = np.arange(380, 781, 1, dtype=float)
    _XBAR = np.interp(_WL_FINE, _cie_wl, _cie_xb)
    _YBAR = np.interp(_WL_FINE, _cie_wl, _cie_yb)
    _ZBAR = np.interp(_WL_FINE, _cie_wl, _cie_zb)

    def _spectrum_to_xy(wavelengths, intensities):
        I = np.interp(_WL_FINE, wavelengths, intensities, left=0.0, right=0.0)
        X = np.trapz(I * _XBAR, _WL_FINE)
        Y = np.trapz(I * _YBAR, _WL_FINE)
        Z = np.trapz(I * _ZBAR, _WL_FINE)
        S = X + Y + Z
        return (X/S, Y/S) if S > 1e-30 else (0.0, 0.0)

    # Spectral locus (xy of monochromatic light)
    _spec_xy = np.array([
        (xb/(xb+yb+zb), yb/(xb+yb+zb)) if (xb+yb+zb) > 0 else (0.0, 0.0)
        for xb, yb, zb in zip(_XBAR, _YBAR, _ZBAR)
    ])

    # sRGB helpers for coloured background
    def _linear_to_srgb(c):
        c = np.clip(c, 0, None)
        return np.where(c <= 0.0031308, 12.92*c, 1.055*np.power(c, 1/2.4) - 0.055)

    def _xy_to_srgb(x, y):
        z = 1.0 - x - y
        if y <= 1e-12:
            return (1.0, 1.0, 1.0)
        Y=1.0; X=x/y; Z=z/y
        M = np.array([[ 3.2406,-1.5372,-0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [ 0.0557,-0.2040, 1.0570]])
        rgb = _linear_to_srgb(M @ np.array([X, Y, Z]))
        return tuple(np.clip(rgb, 0, 1))

    # ── Named colour palette ─────────────────────────────────────────────
    _NAMED_COLORS = {
        "Blue":       "#1f77b4",
        "Orange":     "#ff7f0e",
        "Green":      "#2ca02c",
        "Red":        "#d62728",
        "Purple":     "#9467bd",
        "Brown":      "#8c564b",
        "Pink":       "#e377c2",
        "Gray":       "#7f7f7f",
        "Olive":      "#bcbd22",
        "Cyan":       "#17becf",
        "Navy":       "#003f7f",
        "Teal":       "#0e7c7b",
        "Gold":       "#e6a800",
        "Magenta":    "#c800c8",
        "Black":      "#000000",
    }
    _COLOR_NAMES = list(_NAMED_COLORS.keys())
    _HEX_TO_NAME = {v: k for k, v in _NAMED_COLORS.items()}

    def _hex_to_name(h):
        """Return colour name for a hex string, or nearest fallback."""
        return _HEX_TO_NAME.get(h, _COLOR_NAMES[0])

    # ── Colour presets ───────────────────────────────────────────────────
    _PRESET_ND = {
        "ND100": "Blue",   "ND50":  "Red",    "ND25":  "Green",
        "ND10":  "Orange", "ND5":   "Purple", "ND2.5": "Cyan",
        "ND1":   "Pink",   "ND0.1": "Brown",  "ND0.01":"Black",
    }
    _PRESET_TEMP = {
        "unimplanted": "Blue",    "deposited":   "Gray",
        "anneal@500":  "Green",   "anneal@600":  "Orange",
        "anneal@700":  "Cyan",    "anneal@800":  "Red",
        "anneal@900":  "Pink",    "anneal@1000": "Black",
        "anneal@1100": "Brown",
    }
    _PALETTE_CYCLE = _COLOR_NAMES  # cycle through all named colours

    def _match_preset(stem, preset_dict):
        sl = stem.lower()
        for key, color_name in preset_dict.items():
            if key.lower() in sl:
                return color_name
        return None

    # ── Discover files ───────────────────────────────────────────────────
    cie_corr_files = []
    if cie_dir and os.path.isdir(cie_dir):
        cie_corr_files = sorted([
            f for f in os.listdir(cie_dir) if f.endswith("_corrected.txt")
        ])

    if not cie_corr_files:
        if cie_dir:
            st.info("No `_corrected.txt` files found in the specified folder.")
    else:
        st.caption(f"Found **{len(cie_corr_files)}** corrected spectrum file(s).")

        # ── Colour preset selector ────────────────────────────────────────
        col_pr1, col_pr2, col_pr3 = st.columns([2, 1, 2])
        with col_pr1:
            cie_preset = st.radio(
                "Colour preset",
                ["ND power series", "Temperature / anneal series", "Custom (edit table)"],
                horizontal=True,
                key="cie_preset_radio",
            )
        with col_pr2:
            cie_show_bg = st.checkbox("Show coloured background", value=True, key="cie_show_bg")
        with col_pr3:
            cie_legend_pos = st.radio(
                "Legend position",
                ["Inside top-right", "Inside top-left",
                 "Inside bottom-right", "Inside bottom-left", "Outside (right)"],
                horizontal=False, key="cie_legend_pos",
            )

        # ── Editable table ────────────────────────────────────────────────
        cie_table_key = f"cie_table__{cie_dir}"

        def _build_default_cie_rows(files, preset):
            rows = []
            for i, fname in enumerate(files):
                stem = clean_label(fname)
                if preset == "ND power series":
                    color_name = _match_preset(stem, _PRESET_ND) or _PALETTE_CYCLE[i % len(_PALETTE_CYCLE)]
                elif preset == "Temperature / anneal series":
                    color_name = _match_preset(stem, _PRESET_TEMP) or _PALETTE_CYCLE[i % len(_PALETTE_CYCLE)]
                else:
                    color_name = _PALETTE_CYCLE[i % len(_PALETTE_CYCLE)]
                rows.append({
                    "Include":       True,
                    "File":          fname,
                    "Display label": stem,
                    "X-order":       float(i + 1),
                    "Color":         color_name,
                })
            return rows

        # Rebuild table when preset changes
        prev_preset_key = f"cie_prev_preset__{cie_dir}"
        cie_version_key = f"cie_version__{cie_dir}"
        if (cie_table_key not in st.session_state
                or st.session_state.get(prev_preset_key) != cie_preset):
            st.session_state[cie_table_key] = pd.DataFrame(
                _build_default_cie_rows(cie_corr_files, cie_preset))
            st.session_state[prev_preset_key] = cie_preset
            st.session_state[cie_version_key] = st.session_state.get(cie_version_key, 0) + 1

        cie_df_raw = st.session_state[cie_table_key]
        # Sync new files (append only) — increment version so widget rebuilds
        existing_cie = set(cie_df_raw["File"].tolist())
        new_cie_rows = [{
            "Include": True, "File": fname,
            "Display label": clean_label(fname),
            "X-order": float(cie_df_raw["X-order"].max() + 1) if len(cie_df_raw) else 1.0,
            "Color": _PALETTE_CYCLE[len(cie_df_raw) % len(_PALETTE_CYCLE)],
        } for fname in cie_corr_files if fname not in existing_cie]
        if new_cie_rows:
            cie_df_raw = pd.concat([cie_df_raw, pd.DataFrame(new_cie_rows)], ignore_index=True)
            st.session_state[cie_table_key] = cie_df_raw
            st.session_state[cie_version_key] = st.session_state.get(cie_version_key, 0) + 1

        st.markdown("##### File table — edit labels, ordering and colours")
        st.caption(
            "**X-order**: number controlling left-to-right position — smaller = further left. "
            "Decimals are allowed (e.g. 1.5 inserts between 1 and 2). Duplicates resolved by filename.  "
            "**Color**: select from the named palette. "
            "Colours are pre-filled by the preset above; you can override any row at any time."
        )
        # Version-keyed widget: same approach as Tab 5
        cie_editor_key = f"cie_editor__{cie_dir}__v{st.session_state.get(cie_version_key, 0)}"
        cie_edited = st.data_editor(
            cie_df_raw,
            width="stretch",
            hide_index=True,
            disabled=["File"],
            column_config={
                "Include":       st.column_config.CheckboxColumn("Include"),
                "X-order":       st.column_config.NumberColumn("X-order", min_value=0.0, step=0.5, format="%.1f"),
                "Display label": st.column_config.TextColumn("Display label"),
                "File":          st.column_config.TextColumn("File"),
                "Color":         st.column_config.SelectboxColumn(
                    "Color",
                    options=_COLOR_NAMES,
                    help="Choose a colour name. The preset above resets all colours.",
                ),
            },
            key=cie_editor_key,
        )

        # ── Compute CIE coordinates ───────────────────────────────────────
        # Cast X-order to float first — avoids int/float mixed-type sort instability
        cie_included = (cie_edited[cie_edited["Include"]]
                        .copy()
                        .assign(**{"X-order": lambda df: pd.to_numeric(df["X-order"], errors="coerce").fillna(0.0)})
                        .sort_values("X-order")
                        .reset_index(drop=True))

        cie_points = []
        for _, row in cie_included.iterrows():
            fpath = os.path.join(cie_dir, row["File"])
            try:
                wl, intensity = load_corrected_txt(fpath)
                if dead_pixels:
                    intensity = fix_dead_pixels(wl, intensity, dead_pixels)
                cx, cy = _spectrum_to_xy(wl, intensity)
                cie_points.append({
                    "label": row["Display label"],
                    "color": row["Color"],
                    "x": cx, "y": cy,
                })
            except Exception as e:
                st.warning(f"Could not process {row['File']}: {e}")

        if not cie_points:
            st.session_state["t6_ai_ready"] = False
            st.info("No data to plot. Check that the folder contains valid corrected spectra.")
        else:
            # ── Publish artifact for Tab 8 AI Copilot ────────────────────────
            st.session_state["t6_ai_ready"]  = True
            st.session_state["t6_ai_points"] = cie_points   # [{label, color, x, y}, ...]

            # ── First-render stabilisation ────────────────────────────────────
            # scaleanchor + constrain="domain" + width="stretch" can produce
            # unstable paper-coordinate legend/title positions on the first frame
            # (container width not yet settled). One controlled rerun fixes this.
            _cie_init_key = f"cie_layout_ready__{cie_dir}"
            if not st.session_state.get(_cie_init_key, False):
                st.session_state[_cie_init_key] = True
                st.rerun()

            # ── Build Plotly figure ───────────────────────────────────────
            fig_cie = go.Figure()

            # Coloured background (rasterised sRGB grid drawn as image)
            if cie_show_bg:
                _gx = np.linspace(0, 0.8, 300)
                _gy = np.linspace(0, 0.9, 300)
                _Xg, _Yg = np.meshgrid(_gx, _gy)

                # Build polygon path for inside test
                import matplotlib.path as _mpath
                _purple_line = np.array([_spec_xy[-1], _spec_xy[0]])
                _poly = np.vstack([_spec_xy, _purple_line])
                _poly_path = _mpath.Path(_poly)

                _pts = np.stack([_Xg.ravel(), _Yg.ravel()], axis=1)
                _inside = _poly_path.contains_points(_pts)
                _rgb_flat = np.ones((_pts.shape[0], 3))
                for idx in np.where(_inside)[0]:
                    _rgb_flat[idx] = _xy_to_srgb(_pts[idx, 0], _pts[idx, 1])
                _rgb_img = (_rgb_flat.reshape(300, 300, 3) * 255).astype(np.uint8)
                _rgb_img = np.flipud(_rgb_img)   # row 0 = top of image = y=0.9 in data space

                import base64, io
                from PIL import Image as _PILImage
                _pil = _PILImage.fromarray(_rgb_img, "RGB")
                _buf = io.BytesIO()
                _pil.save(_buf, format="PNG")
                _b64 = base64.b64encode(_buf.getvalue()).decode()
                fig_cie.add_layout_image(dict(
                    source=f"data:image/png;base64,{_b64}",
                    xref="x", yref="y",
                    x=0, y=0.9,
                    sizex=0.8, sizey=0.9,
                    sizing="stretch",
                    layer="below",
                    opacity=1.0,
                ))

            # Spectral locus (horseshoe boundary)
            fig_cie.add_trace(go.Scatter(
                x=np.append(_spec_xy[:, 0], _spec_xy[0, 0]),
                y=np.append(_spec_xy[:, 1], _spec_xy[0, 1]),
                mode="lines",
                line=dict(color="black", width=1.8),
                showlegend=False,
                hoverinfo="skip",
            ))
            # Purple line (closing the locus)
            fig_cie.add_trace(go.Scatter(
                x=[_spec_xy[-1, 0], _spec_xy[0, 0]],
                y=[_spec_xy[-1, 1], _spec_xy[0, 1]],
                mode="lines",
                line=dict(color="black", width=1.8),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Wavelength tick marks and labels
            _WP = np.array([0.333, 0.333])   # white point direction reference
            for _nm in [460, 480, 500, 520, 540, 560, 580, 600]:
                _idx = _nm - 380
                if 0 <= _idx < len(_spec_xy):
                    _px, _py = _spec_xy[_idx]
                    _v = np.array([_px, _py]) - _WP
                    _vn = _v / np.linalg.norm(_v) if np.linalg.norm(_v) > 1e-6 else np.array([1e-3, 0])
                    _p2x, _p2y = _px + 0.018*_vn[0], _py + 0.018*_vn[1]
                    fig_cie.add_shape(type="line",
                        x0=_px, y0=_py, x1=_p2x, y1=_p2y,
                        line=dict(color="black", width=1.2))
                    fig_cie.add_annotation(
                        x=_p2x, y=_p2y,
                        text=str(_nm), showarrow=False,
                        font=dict(size=max(font_size-5, 9), family=font_family),
                        xanchor="center", yanchor="middle",
                    )

            # Data points
            for pt in cie_points:
                # Resolve colour name → hex (fallback to black)
                _col = _NAMED_COLORS.get(pt["color"], pt["color"])
                if not str(_col).startswith("#"):
                    _col = "#333333"
                fig_cie.add_trace(go.Scatter(
                    x=[pt["x"]], y=[pt["y"]],
                    mode="markers",
                    name=pt["label"],
                    marker=dict(color=_col, size=10,
                                line=dict(color="black", width=1)),
                    hovertemplate=(
                        f"<b>{pt['label']}</b><br>"
                        f"x = {pt['x']:.4f}<br>y = {pt['y']:.4f}<extra></extra>"
                    ),
                ))

            fig_cie.update_layout(
                xaxis=dict(
                    title=dict(text="x", font=_axis_font),
                    tickfont=_tick_font,
                    range=[0, 0.8],
                    constrain="domain",
                    showgrid=False, zeroline=False,
                    showline=True, linewidth=2, linecolor="black", mirror=True,
                ),
                yaxis=dict(
                    title=dict(text="y", font=_axis_font),
                    tickfont=_tick_font,
                    range=[0, 0.9],
                    scaleanchor="x", scaleratio=1,
                    constrain="domain",
                    showgrid=False, zeroline=False,
                    showline=True, linewidth=2, linecolor="black", mirror=True,
                ),
                title=dict(
                    text="CIE 1931 Chromaticity Diagram",
                    font=dict(size=font_size, family=font_family),
                    x=0.405,
                    xref="paper",
                ),
                legend=dict(
                    font=dict(size=max(font_size-3, 10), family=font_family),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#cccccc", borderwidth=1,
                    xref="paper", yref="paper",
                    **({"x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"}
                       if cie_legend_pos == "Outside (right)"
                       else {
                           "x":       0.55 if "right" in cie_legend_pos else 0.02,
                           "y":       0.98 if "top"   in cie_legend_pos else 0.02,
                           "xanchor": "left",
                           "yanchor": "top"   if "top" in cie_legend_pos else "bottom",
                       }),
                ),
                height=620,
                margin=dict(l=60, r=160 if cie_legend_pos == "Outside (right)" else 20,
                            t=60, b=60),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            _style_axes(fig_cie)
            st.plotly_chart(fig_cie, width="stretch", config={"doubleClick": False, "modeBarButtonsToRemove": ["autoScale2d"]})

            # ── CIE coordinates table ─────────────────────────────────────
            with st.expander("📋 CIE coordinates", expanded=False):
                cie_tbl = [{"Label": p["label"],
                             "x": f"{p['x']:.4f}",
                             "y": f"{p['y']:.4f}"}
                           for p in cie_points]
                st.dataframe(pd.DataFrame(cie_tbl), width="stretch", hide_index=True)




# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
#  TAB 7 — Mapping Heatmap
# ══════════════════════════════════════════════
with tab7:
    import pickle as _pickle

    # ── helpers ───────────────────────────────────────────────────────────────
    # _load_intensity_txt is defined at module level (shared with Tab 8)

    def _load_spec_dict(path: str):
        try:
            with open(path, "rb") as fh:
                return _pickle.load(fh)
        except Exception:
            return None

    def _nearest_key(spec_dict, x_v, y_v):
        keys  = np.array(list(spec_dict.keys()), dtype=float)
        dists = np.hypot(keys[:, 0] - x_v, keys[:, 1] - y_v)
        return tuple(int(v) for v in keys[int(np.argmin(dists))])

    def _nearest_val(arr: np.ndarray, val: float) -> float:
        return float(arr[int(np.argmin(np.abs(arr - val)))])

    def _make_tickvals(coords: np.ndarray, multiplier: int) -> list:
        data_step = float(np.median(np.diff(coords)))
        tick_step = data_step * multiplier
        vals = list(np.arange(float(coords[0]),
                               float(coords[-1]) + data_step * 0.1,
                               tick_step))
        if abs(vals[-1] - float(coords[-1])) > data_step * 0.1:
            vals.append(float(coords[-1]))
        return vals

    # ── session state defaults ────────────────────────────────────────────────
    _m7_defaults = {
        "mapping_dir":      "",
        "mapping_dd_x":     None,
        "mapping_dd_y":     None,
        "m7_last_sel_sig":  None,
        "m7_do_detect":     False,
        "m7_det_pct":       25,
        "m7_det_abs":       None,
        "m7_circ_mult":     0.15,
    }
    for _k, _v in _m7_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.header("🗺️ Mapping Heatmap")

    # ── path input ────────────────────────────────────────────────────────────
    mapping_dir_input = st.text_input(
        "Mapping data folder",
        value=st.session_state["mapping_dir"],
        placeholder="Paste the folder path containing intensity_with_axes.txt …",
        key="mapping_dir_input",
    ).strip()
    if mapping_dir_input != st.session_state["mapping_dir"]:
        st.session_state["mapping_dir"]     = mapping_dir_input
        st.session_state["mapping_dd_x"]    = None
        st.session_state["mapping_dd_y"]    = None
        st.session_state["m7_last_sel_sig"] = None
        st.session_state["m7_det_abs"]      = None

    mapping_dir = st.session_state["mapping_dir"]
    txt_path = os.path.join(mapping_dir, "intensity_with_axes.txt") if mapping_dir else ""
    pkl_path = os.path.join(mapping_dir, "spec_dict.pkl")           if mapping_dir else ""
    has_txt  = bool(mapping_dir and os.path.isfile(txt_path))
    has_pkl  = bool(mapping_dir and os.path.isfile(pkl_path))

    # ── load data ─────────────────────────────────────────────────────────────
    x_coords_m = y_coords_m = data_m = spec_dict_m = None

    if has_txt:
        x_coords_m, y_coords_m, data_m = _load_intensity_txt(txt_path)
        if has_pkl:
            spec_dict_m = _load_spec_dict(pkl_path)
            if spec_dict_m is None:
                st.warning("⚠️ Could not load `spec_dict.pkl` — click-spectrum disabled.")
    elif has_pkl:
        _sd = _load_spec_dict(pkl_path)
        if _sd is not None:
            spec_dict_m = _sd
            _keys = list(_sd.keys())
            _xs = sorted(set(k[0] for k in _keys))
            _ys = sorted(set(k[1] for k in _keys))
            x_coords_m = np.array(_xs, dtype=float)
            y_coords_m = np.array(_ys, dtype=float)
            data_m = np.full((len(_ys), len(_xs)), np.nan)
            for k, v in _sd.items():
                data_m[_ys.index(k[1]), _xs.index(k[0])] = float(v.get("intensity", np.nan))
            st.info("ℹ️ `intensity_with_axes.txt` not found — heatmap rebuilt from `spec_dict.pkl`.")

    # ── main content ──────────────────────────────────────────────────────────
    if data_m is None:
        if mapping_dir:
            st.warning(
                "⚠️ No recognised data files found. "
                "The folder must contain `intensity_with_axes.txt` "
                "(and optionally `spec_dict.pkl`)."
            )
        else:
            st.info("Enter the folder path above to load mapping data.")
    else:
        all_vals = data_m[np.isfinite(data_m)]
        x_lo, x_hi = float(x_coords_m[0]), float(x_coords_m[-1])
        y_lo, y_hi = float(y_coords_m[0]), float(y_coords_m[-1])
        _x_step = float(np.median(np.diff(x_coords_m)))
        _y_step = float(np.median(np.diff(y_coords_m)))

        # Initialise data-dependent defaults once per dataset
        if st.session_state["m7_det_abs"] is None:
            st.session_state["m7_det_abs"] = float(np.percentile(all_vals, 25))
        _xs_avail = _ys_avail = None
        if spec_dict_m is not None:
            _xs_avail = sorted(set(int(k[0]) for k in spec_dict_m.keys()))
            _ys_avail = sorted(set(int(k[1]) for k in spec_dict_m.keys()))
            if st.session_state["mapping_dd_x"] is None:
                st.session_state["mapping_dd_x"] = _xs_avail[0]
            if st.session_state["mapping_dd_y"] is None:
                st.session_state["mapping_dd_y"] = _ys_avail[0]

        # Detection mask from PREVIOUS run's session_state
        _do_detect_pre = st.session_state["m7_do_detect"]
        _det_pct_pre   = st.session_state["m7_det_pct"]
        _det_abs_pre   = st.session_state["m7_det_abs"]
        _circ_mult_pre = st.session_state["m7_circ_mult"]
        if _do_detect_pre:
            mask_detect = (
                (data_m < float(np.percentile(all_vals, _det_pct_pre))) |
                (data_m < _det_abs_pre)
            )
        else:
            mask_detect = np.zeros_like(data_m, dtype=bool)

        col_ctrl, col_hmap = st.columns([1, 2.6])

        # ── Publish artifact for Tab 8 AI Copilot ────────────────────────────
        st.session_state["t7_ai_ready"]      = True
        st.session_state["t7_ai_x"]          = x_coords_m
        st.session_state["t7_ai_y"]          = y_coords_m
        st.session_state["t7_ai_data"]       = data_m
        st.session_state["t7_ai_do_detect"]  = _do_detect_pre
        st.session_state["t7_ai_det_pct"]    = float(_det_pct_pre)
        st.session_state["t7_ai_det_abs"]    = _det_abs_pre

        # ── left column: display settings only ───────────────────────────────
        with col_ctrl:
            st.markdown("#### Display settings")
            cscale = st.radio("Colorscale",
                              ["hot", "viridis", "plasma", "RdYlBu_r"],
                              index=0, horizontal=True)
            pct_range = st.slider("Intensity range (percentile)",
                                  0, 100, (0, 100), step=1)
            vmin_m = float(np.percentile(all_vals, pct_range[0]))
            vmax_m = float(np.percentile(all_vals, pct_range[1]))

            rev_x = st.checkbox("Reverse X axis  (high → low)", value=True,
                                help="Match motor stage / camera orientation.")
            rev_y = st.checkbox("Reverse Y axis  (high → low)", value=True,
                                help="Match motor stage / camera orientation.")
            tick_mult_x = st.number_input(
                "X tick interval multiplier  (× data step)",
                min_value=1, max_value=20, value=2, step=1,
                key="mapping_tick_mult_x",
            )
            tick_mult_y = st.number_input(
                "Y tick interval multiplier  (× data step)",
                min_value=1, max_value=20, value=2, step=1,
                key="mapping_tick_mult_y",
            )
            zsmooth_val = False if st.radio(
                "Smoothing", ["None", "best"], index=1, horizontal=True
            ) == "None" else "best"

        # ── right column: heatmap ─────────────────────────────────────────────
        with col_hmap:
            x_range_disp = [x_hi, x_lo] if rev_x else [x_lo, x_hi]
            y_range_disp = [y_hi, y_lo] if rev_y else [y_lo, y_hi]
            x_tickvals   = _make_tickvals(x_coords_m, tick_mult_x)
            y_tickvals   = _make_tickvals(y_coords_m, tick_mult_y)
            _circ_rx     = _x_step * _circ_mult_pre
            _circ_ry     = _y_step * _circ_mult_pre

            fig_hmap = go.Figure()

            # ① Heatmap trace — hover disabled so it cannot interfere with
            #    the _click_target selection layer.  X/Y/Intensity info is
            #    shown in the Click-to-spectrum panel below instead.
            fig_hmap.add_trace(go.Heatmap(
                z=data_m, x=x_coords_m, y=y_coords_m,
                colorscale=cscale, zmin=vmin_m, zmax=vmax_m,
                zsmooth=zsmooth_val,
                colorbar=dict(
                    title=dict(text="Intensity", side="right", font=_axis_font),
                    tickfont=_tick_font,
                ),
                hoverinfo="none",  # no tooltip; panel below shows X/Y/Intensity
            ))

            # ② Click-target scatter — internal click-capture layer.
            #    Dense uniform grid covers the full mapping extent so ANY click
            #    within the data area lands on a nearby marker.  The click handler
            #    below snaps the raw coordinate to the nearest real data point.
            #    hoverinfo="none" hides the tooltip without suppressing selection
            #    events (unlike "skip" which blocks events entirely).
            if spec_dict_m is not None:
                # 8× denser than data grid → fine enough to cover gaps visually
                _n_dense_x = max(len(x_coords_m) * 8, 120)
                _n_dense_y = max(len(y_coords_m) * 8, 120)
                _xd = np.linspace(float(x_coords_m[0]), float(x_coords_m[-1]), _n_dense_x)
                _yd = np.linspace(float(y_coords_m[0]), float(y_coords_m[-1]), _n_dense_y)
                _xxd, _yyd = np.meshgrid(_xd, _yd)
                fig_hmap.add_trace(go.Scatter(
                    x=_xxd.flatten(), y=_yyd.flatten(),
                    mode="markers",
                    marker=dict(size=6, opacity=0.01,
                                color="rgba(0,0,0,0.01)"),
                    showlegend=False,
                    hoverinfo="none",
                    name="_click_target",
                ))

            # ③ Detection overlay — layout.shapes (non-selectable circles)
            if _do_detect_pre and mask_detect.any():
                _det_yi, _det_xi = np.where(mask_detect)
                _shapes = [
                    dict(
                        type="circle", xref="x", yref="y",
                        x0=float(x_coords_m[xi]) - _circ_rx,
                        x1=float(x_coords_m[xi]) + _circ_rx,
                        y0=float(y_coords_m[yi]) - _circ_ry,
                        y1=float(y_coords_m[yi]) + _circ_ry,
                        line=dict(color="cyan", width=2),
                        fillcolor="rgba(0,0,0,0)",
                    )
                    for yi, xi in zip(_det_yi, _det_xi)
                ]
            else:
                _shapes = []

            fig_hmap.update_layout(
                shapes=_shapes,
                showlegend=False,
                xaxis=dict(
                    title=dict(text="X Position (µm)", font=_axis_font, standoff=12),
                    tickfont=_tick_font,
                    range=x_range_disp, autorange=False,
                    tickmode="array", tickvals=x_tickvals,
                    showgrid=False,
                ),
                yaxis=dict(
                    title=dict(text="Y Position (µm)", font=_axis_font, standoff=12),
                    tickfont=_tick_font,
                    range=y_range_disp, autorange=False,
                    tickmode="array", tickvals=y_tickvals,
                    showgrid=False,
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=60, r=20, t=40, b=60),
                height=520,
            )

            st.caption(
                "💡 Use **Reset axes** in the top-right modebar to restore the initial view."
            )
            _style_axes(fig_hmap)
            _hmap_evt = st.plotly_chart(
                fig_hmap, width="stretch", key="heatmap_main",
                on_select="rerun", selection_mode="points",
                config={
                    "doubleClick": False,
                    "modeBarButtonsToRemove": ["autoScale2d"],
                },
            )

            # ── Click handling: accept only _click_target (curve_number=1) ──
            # Heatmap (curve 0) does not reliably populate selection.points;
            # the dedicated scatter overlay is the proven reliable path.
            try:
                _raw_pts = list(_hmap_evt.selection.points)
            except Exception:
                _raw_pts = []

            _valid_pts = [p for p in _raw_pts if p.get("curve_number", -1) == 1]

            if _valid_pts:
                _p   = _valid_pts[0]
                _sig = (
                    _p.get("curve_number"),
                    _p.get("point_index"),
                    round(float(_p.get("x", 0)), 2),
                    round(float(_p.get("y", 0)), 2),
                )
                if _sig != st.session_state["m7_last_sel_sig"]:
                    st.session_state["m7_last_sel_sig"] = _sig
                    if spec_dict_m is not None:
                        st.session_state["mapping_dd_x"] = int(_nearest_val(
                            np.array(_xs_avail, dtype=float), float(_p["x"])))
                        st.session_state["mapping_dd_y"] = int(_nearest_val(
                            np.array(_ys_avail, dtype=float), float(_p["y"])))
                    # No explicit st.rerun() — on_select="rerun" already triggers one.
                    # A second rerun would cause calibration/dead_pixel state to be
                    # read before sidebar widgets are fully settled in this run.

        # ═══════════════════════════════════════════════════════════════════════
        # Below columns: Click-to-spectrum → Ion Implantation Detection → Export
        # ═══════════════════════════════════════════════════════════════════════

        # ── Click-to-spectrum ─────────────────────────────────────────────────
        if spec_dict_m is not None:
            st.markdown("---")
            st.markdown("#### 📉 Click-to-spectrum")
            st.caption(
                "Single-click a point on the heatmap to load its spectrum, "
                "or use the dropdowns to select manually."
            )

            dd_col1, dd_col2, dd_col3, dd_col4 = st.columns([2, 2, 2, 2])
            with dd_col1:
                dd_x = st.selectbox("X position (µm)", _xs_avail,
                                    key="mapping_dd_x")
            with dd_col2:
                dd_y = st.selectbox("Y position (µm)", _ys_avail,
                                    key="mapping_dd_y")
            with dd_col3:
                # Intensity at the nearest grid point from the heatmap data
                _best_key_tmp = _nearest_key(spec_dict_m, dd_x, dd_y)
                _intensity_at_pt = float(_nearest_val(
                    x_coords_m, float(_best_key_tmp[0])))  # placeholder init
                # Look up from data_m using nearest indices
                _ix = int(np.argmin(np.abs(x_coords_m - float(_best_key_tmp[0]))))
                _iy = int(np.argmin(np.abs(y_coords_m - float(_best_key_tmp[1]))))
                _intensity_at_pt = float(data_m[_iy, _ix]) if np.isfinite(data_m[_iy, _ix]) else float("nan")
                st.metric("Intensity (a.u.)", f"{_intensity_at_pt:.3f}")
            with dd_col4:
                apply_calib_spec = st.checkbox(
                    "Apply sidebar calibration", value=False,
                    key="mapping_apply_calib",
                    help="Apply the calibration files configured in the sidebar.",
                )

            best_key   = _nearest_key(spec_dict_m, dd_x, dd_y)
            spec_entry = spec_dict_m[best_key]
            wl_spec    = np.array(spec_entry["wavelengths_single"], dtype=float)
            int_spec   = np.array(spec_entry["data_single"],        dtype=float)

            int_spec = fix_dead_pixels(wl_spec, int_spec, dead_pixels)

            if apply_calib_spec:
                try:
                    wl_spec, int_spec, _ = apply_correction(
                        wl_spec, int_spec, calib_folder, calib_file_roles)
                    y_label_spec      = "Corrected Intensity (a.u.)"
                    spec_title_suffix = "  [calibrated]"
                except Exception as _e:
                    st.warning(f"⚠️ Calibration failed: {_e}  — showing raw spectrum.")
                    y_label_spec      = "Intensity (a.u.)"
                    spec_title_suffix = ""
            else:
                y_label_spec      = "Intensity (a.u.)"
                spec_title_suffix = ""

            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=wl_spec, y=int_spec, mode="lines",
                line=dict(color="#0072B2", width=line_width),
                name=f"({best_key[0]}, {best_key[1]}) µm",
            ))
            fig_spec.update_layout(
                xaxis=dict(title=dict(text="Wavelength (nm)", font=_axis_font),
                           tickfont=_tick_font, range=[WL_MIN, WL_MAX]),
                yaxis=dict(title=dict(text=y_label_spec, font=_axis_font),
                           tickfont=_tick_font),
                title=dict(
                    text=(f"Spectrum at X = {best_key[0]} µm,  Y = {best_key[1]} µm"
                          + spec_title_suffix),
                    font=dict(size=font_size, family=font_family),
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=60, r=20, t=50, b=60), height=350,
            )
            _style_axes(fig_spec)
            st.plotly_chart(fig_spec, width="stretch",
                            config={"doubleClick": False,
                                    "modeBarButtonsToRemove": ["autoScale2d"]})

        # ── Ion Implantation Site Detection ───────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 Ion Implantation Site Detection")

        _do_detect_new = st.checkbox(
            "Enable implantation site overlay",
            value=st.session_state["m7_do_detect"],
            key="m7_do_detect_widget",
        )
        st.session_state["m7_do_detect"] = _do_detect_new

        _det_pct_new = st.slider(
            "Percentile threshold  (flag if intensity < Nth percentile)",
            0, 100, int(st.session_state["m7_det_pct"]), step=1,
            disabled=not _do_detect_new, key="m7_det_pct_widget",
        )
        st.session_state["m7_det_pct"] = _det_pct_new

        _det_abs_new = st.number_input(
            "Absolute threshold  (flag if intensity < value)",
            min_value=0.0, value=float(st.session_state["m7_det_abs"]),
            step=0.1, format="%.2f",
            disabled=not _do_detect_new, key="m7_det_abs_widget",
        )
        st.session_state["m7_det_abs"] = _det_abs_new

        # Circle size — lives here, next to the other detection controls
        _circ_mult_new = st.slider(
            "Detection circle size multiplier  (× data step)",
            min_value=0.05, max_value=0.50,
            value=float(st.session_state["m7_circ_mult"]),
            step=0.05,
            disabled=not _do_detect_new,
            key="m7_circ_mult_widget",
            help="Sets the radius of detection circles as a fraction of the "
                 "data point spacing.  0.15 = small, 0.45 = original large size.",
        )
        st.session_state["m7_circ_mult"] = _circ_mult_new

        # Recompute mask using CURRENT widget values for the results table.
        # (The heatmap overlay above still uses _do_detect_pre / mask_detect
        #  because it rendered before these widgets; the table must not lag.)
        if _do_detect_new:
            _mask_now = (
                (data_m < float(np.percentile(all_vals, _det_pct_new))) |
                (data_m < _det_abs_new)
            )
        else:
            _mask_now = np.zeros_like(data_m, dtype=bool)

        if _do_detect_new and _mask_now.any():
            _det_yi2, _det_xi2 = np.where(_mask_now)
            detected_coords = [
                {"X (µm)": float(x_coords_m[xi]),
                 "Y (µm)": float(y_coords_m[yi]),
                 "Intensity": float(data_m[yi, xi])}
                for yi, xi in zip(_det_yi2, _det_xi2)
            ]
            st.markdown(
                f"**{len(detected_coords)} implantation sites detected** "
                f"(percentile < {_det_pct_new}th  OR  "
                f"absolute < {_det_abs_new:.2f})"
            )
            st.dataframe(
                pd.DataFrame(detected_coords).sort_values("Intensity"),
                width="stretch", hide_index=True,
            )
        else:
            detected_coords = []

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("💾 Export", expanded=False):
            exp_col1, exp_col2, exp_col3 = st.columns(3)

            with exp_col1:
                st.markdown("**Heatmap PNG**")
                png_dpi = st.number_input(
                    "Resolution (DPI)", min_value=72, max_value=600,
                    value=300, step=50, key="mapping_png_dpi",
                )
                if st.button("Generate PNG", key="mapping_png_btn"):
                    try:
                        import kaleido as _kaleido  # noqa: F401
                        _exp_fs      = max(font_size * 2, 28)
                        _exp_tick_fs = max(_exp_fs - 4, 20)
                        _exp_cb_fs   = max(_exp_fs - 6, 18)
                        _exp_font    = dict(size=_exp_fs,      family=font_family, color="black")
                        _exp_tick    = dict(size=_exp_tick_fs, family=font_family, color="black")
                        _exp_cb_tick = dict(size=_exp_cb_fs,   family=font_family, color="black")
                        _ml = max(100, _exp_tick_fs * 5)
                        _mb = max(100, _exp_tick_fs * 5)
                        _mr = max(160, _exp_cb_fs * 9)
                        _fig_exp = go.Figure(fig_hmap)
                        # Remove click-target scatter from export
                        _fig_exp.data = tuple(
                            t for t in _fig_exp.data
                            if not (hasattr(t, "name") and t.name == "_click_target")
                        )
                        _fig_exp.update_layout(
                            showlegend=False,
                            xaxis=dict(
                                title=dict(text="X Position (µm)", font=_exp_font,
                                           standoff=max(20, _exp_tick_fs)),
                                tickfont=_exp_tick,
                                range=x_range_disp, autorange=False,
                                tickmode="array", tickvals=x_tickvals,
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title=dict(text="Y Position (µm)", font=_exp_font,
                                           standoff=max(20, _exp_tick_fs)),
                                tickfont=_exp_tick,
                                range=y_range_disp, autorange=False,
                                tickmode="array", tickvals=y_tickvals,
                                showgrid=False,
                            ),
                            margin=dict(l=_ml, r=_mr, t=80, b=_mb),
                        )
                        _fig_exp.update_traces(
                            selector=dict(type="heatmap"),
                            colorbar=dict(
                                title=dict(text="Intensity", side="right", font=_exp_font),
                                tickfont=_exp_cb_tick,
                                len=0.85, x=1.02,
                            ),
                        )
                        _png_bytes = _fig_exp.to_image(
                            format="png", width=1400, height=1000,
                            scale=png_dpi / 96.0,
                        )
                        st.download_button(
                            "⬇️ Download heatmap.png",
                            data=_png_bytes, file_name="heatmap.png",
                            mime="image/png", key="mapping_png_dl",
                        )
                    except ImportError:
                        st.error(
                            "❌ `kaleido` not installed.  "
                            "Run `pip install kaleido` and restart the app."
                        )

            with exp_col2:
                st.markdown("**Detected sites → xlsx**")
                if detected_coords:
                    import io as _io
                    _buf_xl = _io.BytesIO()
                    with pd.ExcelWriter(_buf_xl, engine="openpyxl") as _wr:
                        pd.DataFrame(detected_coords).sort_values("Intensity") \
                          .to_excel(_wr, index=False, sheet_name="Detected sites")
                    st.download_button(
                        "⬇️ Download detected_sites.xlsx",
                        data=_buf_xl.getvalue(), file_name="detected_sites.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="mapping_xl_dl",
                    )
                else:
                    st.caption("Enable implantation site overlay first.")

            with exp_col3:
                st.markdown("**Detected sites → txt**")
                if detected_coords:
                    _txt_lines = ["X (µm)\tY (µm)\tIntensity"]
                    for _row in sorted(detected_coords, key=lambda r: r["Intensity"]):
                        _txt_lines.append(
                            f"{_row['X (µm)']:.1f}\t{_row['Y (µm)']:.1f}\t"
                            f"{_row['Intensity']:.4f}"
                        )
                    st.download_button(
                        "⬇️ Download detected_sites.txt",
                        data="\n".join(_txt_lines).encode("utf-8"),
                        file_name="detected_sites.txt",
                        mime="text/plain", key="mapping_txt_dl",
                    )
                else:
                    st.caption("Enable implantation site overlay first.")


# ══════════════════════════════════════════════
#  TAB 8 — AI Physics Copilot
# ══════════════════════════════════════════════

# ── Provider availability checks ──────────────────────────────────────────────
def _t8_get_secret(key: str) -> str:
    """Read API key from st.secrets, fall back to empty string."""
    try:
        return st.secrets[key]
    except Exception:
        return ""

_T8_GROQ_KEY = _t8_get_secret("GROQ_API_KEY")

# Check Groq package availability
try:
    from langchain_openai import ChatOpenAI as _T8ChatOpenAI
    _T8_GROQ_PKG = True
except ImportError:
    _T8_GROQ_PKG = False

# Available Groq models for user selection
_T8_GROQ_MODELS = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]


# ── Helper: call Groq ──────────────────────────────────────────────────────────
def _t8_call_groq(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """
    Call Groq via ChatOpenAI-compatible layer using the user-selected model.
    Returns result string or [ERROR] message.
    """
    if not _T8_GROQ_PKG:
        return "[ERROR] langchain_openai package not installed. Run: pip install langchain-openai"
    if not _T8_GROQ_KEY:
        return "[ERROR] GROQ_API_KEY not found in .streamlit/secrets.toml"
    try:
        import os as _os
        import re as _re
        from langchain_core.messages import HumanMessage, SystemMessage
        _os.environ["OPENAI_API_KEY"]  = _T8_GROQ_KEY
        _os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
        _os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        _llm = _T8ChatOpenAI(
            model=model_id,
            base_url="https://api.groq.com/openai/v1",
            api_key=_T8_GROQ_KEY,
            temperature=0.3,
            max_tokens=4000 if "120b" in model_id else 2000,
        )
        _msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        _result = _llm.invoke(_msgs)
        _text = (_result.content or "").strip()
        # Strip <think>…</think> blocks (chain-of-thought reasoning models)
        _text = _re.sub(r"<think>.*?</think>", "", _text, flags=_re.DOTALL).strip()
        return _text
    except Exception as _e:
        return f"[ERROR] Groq call failed ({model_id}): {_e}"


# ── Helper: build structured payload ──────────────────────────────────────────
def _t8_build_payload(rr: dict, ll_summary: list, sample_label: str, sample_note: str) -> str:
    """Convert session state rate-equation result into a structured text payload for the LLM."""
    import numpy as _np

    params_fit  = rr["params_fit"]
    param_means = rr["param_means"]
    param_stds  = rr["param_stds"]
    ch1_lbl     = rr["ch1_label"]
    tau1        = rr["tau1"]
    tau2        = rr["tau2"]
    eta_ch1     = rr["eta_ch1"]
    eta_ch2     = rr["eta_ch2"]
    r2_ch1      = rr["r2_ch1"]
    r2_ch2      = rr["r2_ch2"]
    best_loss   = rr["best_loss"]
    num_runs    = rr.get("num_runs_used", "?")

    # Detect unstable parameters (std > mean)
    RE_PARAM_NAMES = ["k_r1", "k_nr1", "k_capture", "k_r2", "k_nr2", "N_defect", "beta"]
    unstable = [
        RE_PARAM_NAMES[i]
        for i in range(len(RE_PARAM_NAMES))
        if param_stds[i] > param_means[i]
    ]

    is_triple = (ch1_lbl == "UVB+BB")
    mode_str  = "Triple peak (UVB+BB+YB)" if is_triple else "Twin peak (BB+YB)"

    kr1, knr1, kcap, kr2, knr2, N_def, beta = params_fit

    # Format tau
    tau1_ns = tau1 * 1e9
    tau2_us = tau2 * 1e6

    lines = []
    lines.append(f"Sample label:       {sample_label if sample_label else '(not specified)'}")
    lines.append(f"Researcher notes:   {sample_note if sample_note else '(none)'}")
    lines.append("")
    lines.append(f"Analysis mode:      {mode_str}")
    lines.append(f"Channel 1 label:    {ch1_lbl}")
    lines.append("")
    lines.append("--- Fit quality ---")
    lines.append(f"Optimisation runs:  {num_runs}")
    lines.append(f"Best total loss:     {best_loss:.5f}")
    lines.append(f"R² ({ch1_lbl}):          {r2_ch1:.4f}")
    lines.append(f"R² (YB):            {r2_ch2:.4f}")
    if unstable:
        lines.append(f"Unstable params:    {', '.join(unstable)}  ← std > mean across runs")
    else:
        lines.append("Unstable params:    none")
    lines.append("")
    lines.append("--- Derived physical quantities ---")
    lines.append(f"τ₁ ({ch1_lbl} lifetime):  {tau1_ns:.2f} ns")
    lines.append(f"τ₂ (YB lifetime):    {tau2_us:.3f} µs")
    lines.append(f"η ({ch1_lbl}):            {eta_ch1 * 100:.2f}%  (radiative efficiency)")
    lines.append(f"η (YB):              {eta_ch2 * 100:.2f}%  (radiative efficiency)")
    lines.append("")
    lines.append("--- Rate equation parameters (best fit) ---")
    lines.append(f"k_r1   = {kr1:.3e}  s⁻¹  (radiative rate, {ch1_lbl} channel)")
    lines.append(f"k_nr1  = {knr1:.3e}  s⁻¹  (non-radiative rate, {ch1_lbl} channel)")
    lines.append(f"k_cap  = {kcap:.3e}  s⁻¹  (carrier capture rate, {ch1_lbl}→YB)")
    lines.append(f"k_r2   = {kr2:.3e}  s⁻¹  (radiative rate, YB channel)")
    lines.append(f"k_nr2  = {knr2:.3e}  s⁻¹  (non-radiative rate, YB channel)")
    lines.append(f"N_def  = {N_def:.3e}       (defect density, a.u.)")
    lines.append(f"β      = {beta:.3e}  (generation scaling factor)")
    lines.append("")

    if ll_summary:
        lines.append("--- Power-law (log-log) slopes ---")
        for row in ll_summary:
            lines.append(
                f"  {row.get('Band','?'):10s}  slope = {row.get('Slope','?')}   "
                f"R² = {row.get('R² (fit)', row.get('R²','?'))}"
            )
    else:
        lines.append("--- Power-law slopes ---")
        lines.append("  (Step 2 log-log analysis not available for this session)")

    return "\n".join(lines)


# ── Helper: parse fixed tags from model output ────────────────────────────────
_T8_TAGS = ["CORE_FINDINGS", "PHYSICAL_INTERPRETATION", "CONFIDENCE_AND_LIMITS", "NEXT_EXPERIMENTS"]

def _t8_parse_tags(text: str) -> dict:
    """Extract [TAG]…[/TAG] sections. Missing tags get a placeholder."""
    import re as _re
    result = {}
    for tag in _T8_TAGS:
        pat = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_T8_TAGS)})\]|$)"
        m = _re.search(pat, text, flags=_re.DOTALL | _re.IGNORECASE)
        result[tag] = m.group(1).strip() if m else "(This section was not returned by the model.)"
    return result


# ── Helper: parse a saved Tab-4 Step-3 .txt file ─────────────────────────────
def _t8_parse_saved_txt(content: str) -> dict | None:
    """
    Parse a rate-equation result .txt file into a dict compatible with
    _t8_build_payload(). Supports both current Tab 4 Step 3 export format
    and older local fitting script formats.

    Tolerant matching:
    - Accepts both ':' and '=' as separators
    - Accepts Greek τ / ASCII tau
    - Accepts R² / R2 with various labels
    - Accepts k_capture / k_cap
    - Non-critical fields may be absent; only the 5 core rate constants are required.
    """
    import re as _re
    import numpy as _np
    import math as _math

    # Normalise: replace Greek colon / non-breaking spaces for simpler regex
    _c = content.replace("\u00a0", " ").replace("：", ":")

    # Separator pattern: optional whitespace, then ':' or '=', then optional whitespace
    _SEP = r"[\s]*[=:][\s]*"

    def _grab(pattern, text, cast=float):
        """Return first match, or None."""
        m = _re.search(pattern, text, flags=_re.IGNORECASE | _re.MULTILINE)
        if m:
            try:
                return cast(m.group(1))
            except (ValueError, TypeError):
                return None
        return None

    # ── Core rate constants (required) ────────────────────────────────────────
    # Each line looks like: "k_r1: 8.803e+06  (Mean: ..., Std: ...)"
    # or: "k_r1 = 8.803e+06"
    # Regex grabs only the FIRST numeric token after the separator.
    _NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"

    kr1  = _grab(rf"k[_\s]?r1{_SEP}{_NUM}",              _c)
    knr1 = _grab(rf"k[_\s]?nr1{_SEP}{_NUM}",             _c)
    kcap = _grab(rf"k[_\s]?cap(?:ture)?{_SEP}{_NUM}",    _c)
    kr2  = _grab(rf"k[_\s]?r2{_SEP}{_NUM}",              _c)
    knr2 = _grab(rf"k[_\s]?nr2{_SEP}{_NUM}",             _c)
    N_def = _grab(rf"N[_\s]?def(?:ect)?{_SEP}{_NUM}",    _c)
    beta  = _grab(rf"beta{_SEP}{_NUM}",                   _c)

    # ── Validate core fields ──────────────────────────────────────────────────
    missing = [nm for nm, val in [
        ("k_r1", kr1), ("k_nr1", knr1), ("k_capture", kcap),
        ("k_r2", kr2), ("k_nr2", knr2),
    ] if val is None]

    if missing:
        return {"_parse_error": True, "_missing": missing}

    # ── Lifetimes ─────────────────────────────────────────────────────────────
    # Formats: "τ1 (UV+Blue): 9.85 ns"  / "tau1 = 9.85e-9 s"
    # Detect unit: if value > 1e-3 it's in ns/µs, convert to seconds.
    def _grab_tau(label_pat: str) -> float | None:
        """Try ns-unit form first, then raw seconds form."""
        # ns form: small number followed by ns / µs
        m_ns = _re.search(
            rf"(?:τ|tau)[_\s]?{label_pat}[^\n]*?{_SEP}{_NUM}\s*(n?s|[nµu]s)",
            _c, flags=_re.IGNORECASE,
        )
        if m_ns:
            val = float(m_ns.group(1))
            unit = m_ns.group(2).lower()
            if unit in ("ns", "n"):
                return val * 1e-9
            if unit in ("µs", "us", "μs"):
                return val * 1e-6
            return val  # assume seconds

        # Inline form: "τ1 (Blue): 9.89 ns, τ2 (Yellow): ..."
        m_inline = _re.search(
            rf"(?:τ|tau)[_\s]?{label_pat}[^,\n]*?{_SEP}{_NUM}\s*(ns|µs|us|μs|s\b)",
            _c, flags=_re.IGNORECASE,
        )
        if m_inline:
            val = float(m_inline.group(1))
            unit = m_inline.group(2).lower()
            if unit == "ns":
                return val * 1e-9
            if unit in ("µs", "us", "μs"):
                return val * 1e-6
            return val

        # Raw seconds form: tau1 = 9.85e-9
        m_s = _re.search(
            rf"(?:τ|tau)[_\s]?{label_pat}{_SEP}{_NUM}",
            _c, flags=_re.IGNORECASE,
        )
        if m_s:
            val = float(m_s.group(1))
            # Heuristic: if > 1e-4 it's nanoseconds, not seconds
            return val * 1e-9 if val > 1e-4 else val
        return None

    tau1 = _grab_tau("1")
    tau2 = _grab_tau("2")

    # Fallback: derive from rate constants
    if tau1 is None:
        tau1 = 1.0 / (kr1 + knr1 + kcap)
    if tau2 is None:
        tau2 = 1.0 / (kr2 + knr2)

    # ── Radiative efficiencies ────────────────────────────────────────────────
    # Formats: "Radiative efficiency UVB_BB: 8.67%"
    #          "Radiative efficiency BB: 3.27%"
    eta1 = _grab(
        rf"[Rr]adiative efficiency\s+(?:UVB[_+]?BB|UVB|BB|ch1|UV\+?Blue)[^\n]*?{_SEP}{_NUM}",
        _c,
    )
    eta2 = _grab(
        rf"[Rr]adiative efficiency\s+(?:YB|Yellow|ch2)[^\n]*?{_SEP}{_NUM}",
        _c,
    )

    if eta1 is not None and eta1 > 1.0:   # stored as percent
        eta1 /= 100.0
    if eta1 is None:
        eta1 = kr1 / (kr1 + knr1) if (kr1 and knr1) else 0.0

    if eta2 is not None and eta2 > 1.0:
        eta2 /= 100.0
    if eta2 is None:
        eta2 = kr2 / (kr2 + knr2) if (kr2 and knr2) else 0.0

    # ── R² ───────────────────────────────────────────────────────────────────
    # Formats: "R² (BB): 0.9942"  "R² (UVB+BB): 0.9942"  (new export format)
    #          "R² BB: 0.9999, R² YB: 0.9869"             (unfocused.txt style)
    #          "R2_ch1 = 0.9812"
    r2c1 = _grab(
        rf"R[²2][^:\n]{{0,20}}(?:ch1|UVB\+?BB|UVB_BB|UV\+?Blue|(?<!\w)BB(?!\w))[^\n]*?{_SEP}{_NUM}",
        _c,
    )
    r2c2 = _grab(
        rf"R[²2][^:\n]{{0,20}}(?:ch2|YB|Yellow)[^\n]*?{_SEP}{_NUM}",
        _c,
    )

    # ── Fit quality ───────────────────────────────────────────────────────────
    loss = _grab(r"[Bb]est total loss[:\s]*" + _NUM, _c)
    runs = _grab(r"(?:best of|最优)\s+(\d+)\s+(?:runs|轮)", _c, cast=int)
    if runs is None:
        runs = _grab(r"(\d+)\s+(?:轮|runs).*?(?:优化|optimis)", _c, cast=int)

    # ── Channel label ─────────────────────────────────────────────────────────
    ch1_lbl = "UVB+BB" if _re.search(r"\bUVB\b", _c) else "BB"

    # ── Build arrays ──────────────────────────────────────────────────────────
    params_fit  = _np.array([kr1, knr1, kcap, kr2, knr2,
                              N_def if N_def else 1e3,
                              beta  if beta  else 1e9])
    param_means = params_fit.copy()
    param_stds  = _np.zeros_like(params_fit)   # no multi-run data from file

    return dict(
        params_fit    = params_fit,
        param_means   = param_means,
        param_stds    = param_stds,
        ch1_label     = ch1_lbl,
        tau1          = tau1,
        tau2          = tau2,
        eta_ch1       = eta1,
        eta_ch2       = eta2,
        r2_ch1        = r2c1  if r2c1  is not None else float("nan"),
        r2_ch2        = r2c2  if r2c2  is not None else float("nan"),
        best_loss     = loss  if loss  is not None else float("nan"),
        num_runs_used = runs  if runs  is not None else "?",
    )


# ── System prompt ──────────────────────────────────────────────────────────────
_T8_SYSTEM_PROMPT = """You are an expert AI physics assistant specialising in GaN photoluminescence (PL) spectroscopy and carrier dynamics.

Your role is to interpret the results of dual-channel rate equation fitting performed on power-dependent PL spectra of GaN samples (which may have been processed by ion implantation, annealing, or other treatments).

Physical background:
- BB (~460 nm) and YB (~560 nm) emissions arise from different recombination channels in GaN.
- UVB (~362 nm) is near-band-edge emission, quenched by ion implantation.
- The model solves two coupled steady-state rate equations with parameters: k_r1, k_nr1, k_capture (BB channel), k_r2, k_nr2 (YB channel), N_defect, and β.
- τ₁ = 1/(k_r1 + k_nr1): effective BB lifetime; τ₂ = 1/(k_r2 + k_nr2): effective YB lifetime.
- η = k_r / (k_r + k_nr): radiative efficiency (internal quantum efficiency of each channel).
- Power-law slopes from log-log plots indicate recombination regime (slope ~1 = linear/monomolecular, slope ~2 = bimolecular/excitonic, slope ~0.5–1 may indicate defect-mediated saturation).
- Unstable parameters across optimisation runs suggest the model is underdetermined for those quantities.

Output format: You MUST use exactly these four section tags, in this order, with no extra text outside them:

[CORE_FINDINGS]
Concise bullet-point summary of the most important numerical results and what they mean physically (3–5 bullets).
[PHYSICAL_INTERPRETATION]
A coherent paragraph (4–8 sentences) interpreting the physical mechanisms suggested by the parameter set. Discuss carrier dynamics, recombination pathways, and any notable features (e.g. high k_capture, low η, long τ₂).
[CONFIDENCE_AND_LIMITS]
Honest assessment of fit reliability. Address: R² values, unstable parameters, number of optimisation runs, whether the model may be underdetermined. Frame as limitations the researcher should verify.
[NEXT_EXPERIMENTS]
2–4 concrete suggested follow-up experiments or analyses that would validate or extend these findings.

Important: This is a draft interpretation to assist the researcher. Do not overstate certainty. Do not invent mechanisms not supported by the data."""


# ══════════════════════════════════════════════
#  TAB 8 — Phase 2 helpers (multi-sample)
# ══════════════════════════════════════════════

# Phase 2 output tags
_T8_P2_TAGS = ["TREND_SUMMARY", "ANOMALIES", "SUGGESTED_MECHANISMS", "NEXT_EXPERIMENTS"]


def _t8_build_comparison_payload(records: list) -> str:
    """
    Build a structured multi-sample comparison payload for the LLM.
    records: list of dicts, each with keys from _t8_parse_saved_txt() plus 'label'.
    """
    import math as _math

    def _fmt(val, fmt_spec, nan_str="n/a"):
        if isinstance(val, float) and _math.isnan(val):
            return nan_str
        return format(val, fmt_spec)

    n = len(records)
    lines = []
    lines.append(f"Multi-sample GaN PL rate-equation comparison — {n} sample{'s' if n != 1 else ''}")
    lines.append("=" * 70)
    lines.append("")

    # ── Summary table ──────────────────────────────────────────────────────
    _W = 20
    hdr = (f"{'Label':{_W}} {'Ch1':<8} {'τ₁(ns)':>8} {'τ₂(µs)':>8} "
           f"{'η Ch1%':>8} {'η YB%':>8} {'R²(Ch1)':>8} {'R²(YB)':>8} "
           f"{'Loss':>10} {'Runs':>5}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in records:
        row = (
            f"{r['label']:{_W}} {r['ch1_label']:<8} "
            f"{r['tau1']*1e9:>8.2f} {r['tau2']*1e6:>8.3f} "
            f"{r['eta_ch1']*100:>8.2f} {r['eta_ch2']*100:>8.2f} "
            f"{_fmt(r['r2_ch1'], '.4f'):>8} {_fmt(r['r2_ch2'], '.4f'):>8} "
            f"{_fmt(r['best_loss'], '.5f'):>10} {str(r['num_runs_used']):>5}"
        )
        lines.append(row)
    lines.append("")

    # ── Per-sample rate constants ──────────────────────────────────────────
    lines.append("--- Detailed rate constants per sample ---")
    for r in records:
        kr1, knr1, kcap, kr2, knr2, N_def, beta = r["params_fit"]
        lines.append(f"\n{r['label']} ({r['ch1_label']} + YB):")
        lines.append(f"  k_r1      = {kr1:.3e} s⁻¹  (radiative, Ch1)")
        lines.append(f"  k_nr1     = {knr1:.3e} s⁻¹  (non-radiative, Ch1)")
        lines.append(f"  k_capture = {kcap:.3e} s⁻¹  (Ch1→YB)")
        lines.append(f"  k_r2      = {kr2:.3e} s⁻¹  (radiative, YB)")
        lines.append(f"  k_nr2     = {knr2:.3e} s⁻¹  (non-radiative, YB)")
        lines.append(f"  N_defect  = {N_def:.3e}")
        lines.append(f"  β         = {beta:.3e}")
    return "\n".join(lines)


def _t8_parse_p2_tags(text: str) -> dict:
    """Extract Phase 2 fixed-tag sections from LLM output."""
    import re as _re
    result = {}
    for tag in _T8_P2_TAGS:
        pat = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_T8_P2_TAGS)})\]|$)"
        m = _re.search(pat, text, flags=_re.DOTALL | _re.IGNORECASE)
        result[tag] = m.group(1).strip() if m else "(This section was not returned by the model.)"
    return result


# Phase 2 system prompt
_T8_P2_SYSTEM_PROMPT = """You are an expert AI physics assistant specialising in GaN photoluminescence (PL) spectroscopy and carrier dynamics.

You are comparing multiple GaN samples that have undergone rate-equation fitting. Each row in the comparison table is one sample, described by its fitted parameters.

Physical background:
- BB (~460 nm) and YB (~560 nm) emissions arise from different recombination channels in GaN.
- UVB (~362 nm) is near-band-edge emission, quenched by ion implantation; in triple-peak mode it is merged with BB into channel 1 (UVB+BB).
- k_r1, k_nr1: radiative / non-radiative rates for channel 1 (BB or UVB+BB).
- k_r2, k_nr2: rates for YB channel.
- τ₁ = 1/(k_r1 + k_nr1): channel-1 effective lifetime; τ₂ = 1/(k_r2 + k_nr2): YB lifetime.
- η = k_r/(k_r + k_nr): radiative efficiency (internal quantum efficiency).
- k_capture: carrier capture rate from channel 1 to YB.
- N_defect: effective defect density (model units).
- Systematic changes in η or τ across samples typically reflect dose, anneal temperature, or damage density effects.
- A drop in η(Ch1) with stable η(YB) often indicates increasing non-radiative loss in the BB channel, e.g. from residual implantation damage.

Output format: You MUST use exactly these four section tags, in this order, with no extra text outside them:

[TREND_SUMMARY]
Identify the main trends across the sample set (3–5 bullets). Focus on how key parameters (τ, η, k_r, k_nr, N_defect) vary across samples. Reference specific sample labels and numerical values.
[ANOMALIES]
Identify any samples or parameters that stand out as unexpected, inconsistent, or potentially erroneous (2–4 bullets). If no clear anomalies exist, state that explicitly.
[SUGGESTED_MECHANISMS]
Propose 2–4 physical mechanisms consistent with the observed trends. Discuss carrier dynamics, defect evolution, annealing effects, or implantation damage as appropriate.
[NEXT_EXPERIMENTS]
Suggest 2–4 concrete follow-up experiments or analyses that would validate the proposed mechanisms or resolve ambiguities.

Important: This is a draft interpretation to assist the researcher. Do not overstate certainty. Do not invent mechanisms not supported by the data."""


# ══════════════════════════════════════════════
#  TAB 8 — CIE helpers
# ══════════════════════════════════════════════

_T8_CIE_TAGS = [
    "CHROMATICITY_OVERVIEW",
    "TRENDS_AND_SHIFTS",
    "ANOMALIES_AND_OUTLIERS",
    "NEXT_EXPERIMENTS",
]


def _t8_parse_cie_tags(text: str) -> dict:
    """Extract CIE-mode fixed-tag sections from LLM output."""
    import re as _re
    result = {}
    for tag in _T8_CIE_TAGS:
        pat = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_T8_CIE_TAGS)})\]|$)"
        m = _re.search(pat, text, flags=_re.DOTALL | _re.IGNORECASE)
        result[tag] = m.group(1).strip() if m else "(This section was not returned by the model.)"
    return result


def _t8_build_cie_payload(points: list) -> str:
    """
    Build a structured CIE descriptor payload for the LLM.
    points: list of {label, x, y, color} dicts from Tab 6.
    All computation is done in Python; only the summary text is sent.
    """
    import math as _math

    n = len(points)
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    # White point D65
    WX, WY = 0.3127, 0.3290

    # Per-sample descriptors
    def _region(x, y):
        """Rough CIE 1931 region label."""
        if x < 0.22:
            return "deep-blue"
        elif x < 0.30 and y > 0.25:
            return "blue-white"
        elif x < 0.38 and y > 0.35:
            return "green-white"
        elif x < 0.45 and y > 0.40:
            return "greenish-yellow"
        elif x > 0.45:
            return "yellow-orange"
        else:
            return "near-white"

    lines = []
    lines.append(f"=== GaN PL CIE Chromaticity Summary ===")
    lines.append(f"Number of samples: {n}")
    lines.append("")
    lines.append("--- Per-sample CIE 1931 coordinates ---")
    lines.append(f"{'Label':<28} {'x':>7} {'y':>7} {'Dist-to-D65':>13} Region")
    lines.append("-" * 70)
    for p in points:
        dist = _math.hypot(p["x"] - WX, p["y"] - WY)
        reg  = _region(p["x"], p["y"])
        lines.append(
            f"{p['label']:<28} {p['x']:>7.4f} {p['y']:>7.4f} {dist:>13.4f} {reg}"
        )
    lines.append("")

    # Ensemble statistics
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    sx, sy = float(np.std(xs)), float(np.std(ys))
    spread = float(np.max(np.hypot(xs - cx, ys - cy)))
    max_pair_dist = 0.0
    max_pair_labels = ("", "")
    for i in range(n):
        for j in range(i + 1, n):
            d = _math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d > max_pair_dist:
                max_pair_dist = d
                max_pair_labels = (points[i]["label"], points[j]["label"])

    lines.append("--- Ensemble statistics ---")
    lines.append(f"Centroid: x = {cx:.4f}, y = {cy:.4f}")
    lines.append(f"Std dev:  σx = {sx:.4f}, σy = {sy:.4f}")
    lines.append(f"Max radius from centroid: {spread:.4f}")
    lines.append(
        f"Max pairwise distance: {max_pair_dist:.4f}  "
        f"(between '{max_pair_labels[0]}' and '{max_pair_labels[1]}')"
        if n >= 2 else "Max pairwise distance: n/a (single sample)"
    )
    lines.append(f"D65 white point reference: x = {WX}, y = {WY}")
    lines.append(
        f"Centroid distance to D65: {_math.hypot(cx - WX, cy - WY):.4f}"
    )

    # X and Y range
    lines.append("")
    lines.append("--- Coordinate ranges ---")
    lines.append(
        f"x: {float(np.min(xs)):.4f} – {float(np.max(xs)):.4f}  "
        f"(range {float(np.max(xs) - np.min(xs)):.4f})"
    )
    lines.append(
        f"y: {float(np.min(ys)):.4f} – {float(np.max(ys)):.4f}  "
        f"(range {float(np.max(ys) - np.min(ys)):.4f})"
    )

    return "\n".join(lines)


_T8_CIE_SYSTEM_PROMPT = """You are an expert AI physics assistant specialising in GaN photoluminescence (PL) spectroscopy and CIE 1931 chromaticity analysis.

You are interpreting CIE 1931 xy chromaticity coordinates computed from GaN PL spectra. The coordinates have been calculated by Python from calibrated spectra; you receive only structured summary statistics.

Physical background:
- GaN PL typically contains three emission bands: UVB (~362 nm, near-band-edge), BB (~460 nm, blue defect), YB (~560 nm, yellow defect).
- The CIE xy coordinate reflects the spectral balance: UVB-dominant spectra appear blue (low x, low y); strong YB shifts points toward yellow-green (higher x, higher y).
- Ion implantation quenches UVB, shifting chromaticity away from blue toward green-yellow.
- Systematic chromaticity shifts across samples often correlate with changes in carrier dynamics (τ, η) or defect density.
- The D65 white point (x=0.3127, y=0.3290) is provided as a reference for perceptual context only.
- CIE coordinates alone cannot uniquely identify defect mechanisms — always treat this as a complementary tool alongside spectral fitting results.

Output format: You MUST use exactly these four section tags, in this order, with no extra text before or after them:

[CHROMATICITY_OVERVIEW]
Describe where the sample set sits in the CIE 1931 diagram (3–4 points). Cover: approximate colour region, proximity to white point, overall spread and compactness. Reference specific labels and x/y values.

[TRENDS_AND_SHIFTS]
Identify any systematic trends in chromaticity across the sample set (2–4 points). Describe direction of movement (toward blue/green/yellow) and whether it is consistent. If only one sample is present, describe its position relative to typical GaN emission regions.

[ANOMALIES_AND_OUTLIERS]
Identify any samples whose chromaticity deviates strongly from the group (2–3 points). If no clear outliers exist, state that explicitly. Flag any points that may warrant re-measurement.

[NEXT_EXPERIMENTS]
Suggest 2–3 follow-up measurements or cross-analyses that would help contextualise the chromaticity observations (e.g. correlate with lifetime data, power-dependent maps, or imaging).

Important: This is a draft interpretation to assist the researcher. Do not over-interpret chromaticity coordinates as definitive indicators of defect type or density. Suggest cross-validation with spectral fitting and lifetime data where appropriate."""


# ══════════════════════════════════════════════
#  TAB 8 — Mapping helpers
# ══════════════════════════════════════════════

_T8_MAP_TAGS = [
    "SPATIAL_SUMMARY",
    "REGION_COMPARISON",
    "ANOMALIES_AND_LIMITS",
    "NEXT_EXPERIMENTS",
]


def _t8_parse_mapping_tags(text: str) -> dict:
    """Extract Mapping-mode fixed-tag sections from LLM output."""
    import re as _re
    result = {}
    for tag in _T8_MAP_TAGS:
        pat = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_T8_MAP_TAGS)})\]|$)"
        m = _re.search(pat, text, flags=_re.DOTALL | _re.IGNORECASE)
        result[tag] = m.group(1).strip() if m else "(This section was not returned by the model.)"
    return result


def _t8_build_mapping_payload(
    x_coords: "np.ndarray",
    y_coords: "np.ndarray",
    data_m: "np.ndarray",
    do_detect: bool,
    det_pct: float,
    det_abs: "float | None",
) -> str:
    """
    Build a structured mapping descriptor payload for the LLM.
    All computation is done in Python; only the summary text is sent.
    """
    import math as _math

    all_vals = data_m[np.isfinite(data_m)]
    nx, ny = len(x_coords), len(y_coords)
    n_total = nx * ny

    x_step = float(np.median(np.diff(x_coords))) if nx > 1 else 0.0
    y_step = float(np.median(np.diff(y_coords))) if ny > 1 else 0.0

    g_mean   = float(np.mean(all_vals))
    g_std    = float(np.std(all_vals))
    g_min    = float(np.min(all_vals))
    g_max    = float(np.max(all_vals))
    g_median = float(np.median(all_vals))
    g_p10    = float(np.percentile(all_vals, 10))
    g_p90    = float(np.percentile(all_vals, 90))
    dyn_range = g_max / g_min if g_min > 0 else float("nan")
    p90p10    = g_p90 / g_p10 if g_p10 > 0 else float("nan")

    def _trend_label(profile: np.ndarray) -> str:
        if len(profile) < 2:
            return "n/a (insufficient points)"
        slope = float(np.polyfit(np.arange(len(profile)), profile, 1)[0])
        rel = abs(slope) * len(profile) / (float(np.mean(profile)) + 1e-12)
        if rel < 0.05:
            return "relatively flat"
        return "increases" if slope > 0 else "decreases"

    x_profile = np.nanmean(data_m, axis=0)
    y_profile = np.nanmean(data_m, axis=1)
    x_trend   = _trend_label(x_profile)
    y_trend   = _trend_label(y_profile)

    # Center vs edge
    if nx >= 3 and ny >= 3:
        cx0, cx1 = nx // 4, 3 * nx // 4
        cy0, cy1 = ny // 4, 3 * ny // 4
        center_vals  = data_m[cy0:cy1, cx0:cx1]
        edge_mask    = np.ones_like(data_m, dtype=bool)
        edge_mask[cy0:cy1, cx0:cx1] = False
        edge_vals    = data_m[edge_mask]
        center_mean  = float(np.nanmean(center_vals))
        edge_mean    = float(np.nanmean(edge_vals))
        ce_ratio     = center_mean / edge_mean if edge_mean > 0 else float("nan")
        ce_line = (
            f"Center mean: {center_mean:.1f} | Edge mean: {edge_mean:.1f} | "
            f"Ratio (center/edge): {ce_ratio:.2f}x"
        )
    else:
        ce_line = "Grid too small for center/edge comparison."

    # Detection / region comparison
    if do_detect:
        thresh_pct_val = float(np.percentile(all_vals, det_pct))
        if det_abs is not None:
            mask_imp   = (data_m < thresh_pct_val) | (data_m < det_abs)
            thresh_desc = (
                f"<{det_pct:.0f}th percentile ({thresh_pct_val:.1f}) "
                f"OR < absolute {det_abs:.1f}"
            )
        else:
            mask_imp    = data_m < thresh_pct_val
            thresh_desc = f"<{det_pct:.0f}th percentile ({thresh_pct_val:.1f})"

        n_impl       = int(np.sum(mask_imp))
        frac         = n_impl / n_total * 100.0
        impl_mean    = float(np.nanmean(data_m[mask_imp]))    if n_impl > 0 else float("nan")
        nonimpl_mean = float(np.nanmean(data_m[~mask_imp]))   if (n_total - n_impl) > 0 else float("nan")
        ratio        = nonimpl_mean / impl_mean if (impl_mean > 0 and not _math.isnan(impl_mean)) else float("nan")

        detect_block = (
            f"Detection enabled. Threshold: {thresh_desc}\n"
            f"Detected (low-intensity) sites: {n_impl} / {n_total} points ({frac:.1f}%)\n"
            f"Mean intensity — detected sites: {impl_mean:.1f} | non-detected: {nonimpl_mean:.1f}\n"
            f"Intensity ratio (non-detected / detected): {ratio:.2f}x"
        )
    else:
        detect_block = "Detection not enabled. No region comparison available."

    lines = [
        "=== GaN PL Mapping Summary ===",
        "",
        "--- Grid info ---",
        f"X: {float(x_coords[0]):.1f} – {float(x_coords[-1]):.1f}  (step {x_step:.1f}, {nx} points)",
        f"Y: {float(y_coords[0]):.1f} – {float(y_coords[-1]):.1f}  (step {y_step:.1f}, {ny} points)",
        f"Total measurement points: {n_total}",
        "",
        "--- Global intensity statistics ---",
        f"Mean:   {g_mean:.2f}",
        f"Std:    {g_std:.2f}  (CV = {g_std/g_mean*100:.1f}%)" if g_mean > 0 else f"Std: {g_std:.2f}",
        f"Median: {g_median:.2f}",
        f"Min:    {g_min:.2f}  |  Max: {g_max:.2f}",
        f"P10:    {g_p10:.2f}  |  P90: {g_p90:.2f}",
        f"Dynamic range (max/min): {dyn_range:.1f}x  |  P90/P10: {p90p10:.1f}x",
        "",
        "--- Spatial trends ---",
        f"X-direction (low-X → high-X): {x_trend}",
        f"  X-axis mean profile: " + "  ".join(f"{v:.1f}" for v in x_profile),
        f"Y-direction (low-Y → high-Y): {y_trend}",
        f"  Y-axis mean profile: " + "  ".join(f"{v:.1f}" for v in y_profile),
        "",
        "--- Center vs edge ---",
        ce_line,
        "",
        "--- Region comparison (detection / implantation) ---",
        detect_block,
    ]
    return "\n".join(lines)


_T8_MAP_SYSTEM_PROMPT = """You are an expert AI physics assistant specialising in GaN photoluminescence (PL) spectroscopy and spatially resolved mapping measurements.

You are interpreting a 2D PL intensity map of a GaN sample acquired with a Thorlabs CCS200 system. Python has pre-processed the data; you receive only structured statistical descriptors, not raw spectra.

Physical background:
- GaN PL maps show spatial variation in emission intensity due to material inhomogeneity, surface damage, or deliberate patterning (e.g. ion implantation).
- Ion implantation creates localised regions of reduced PL intensity (quenched UVB/BB emission), appearing as low-intensity sites.
- Spatial gradients may arise from beam non-uniformity, stage drift, sample tilt, or genuine material variation.
- High dynamic range (max/min >> 1) indicates strong spatial inhomogeneity.
- If the intensity ratio (non-detected / detected) is large (e.g. > 3×), this is strong evidence of deliberate patterning or localised damage.

Output format: You MUST use exactly these four section tags, in this order, with no extra text before or after them:

[SPATIAL_SUMMARY]
Describe the overall spatial distribution of PL intensity (3–5 points). Cover: map dimensions, global statistics, dominant spatial trend, and degree of uniformity.

[REGION_COMPARISON]
Compare detected/low-intensity regions versus the rest (or centre vs edge if detection is disabled). Quantify differences and suggest physical origin if supported by the data.

[ANOMALIES_AND_LIMITS]
Identify unexpected features, potential artefacts, or analysis limitations (2–4 points). Include spatial asymmetry inconsistent with the expected pattern, or limitations from the chosen detection threshold.

[NEXT_EXPERIMENTS]
Suggest 2–4 concrete follow-up measurements or analyses that would extend understanding of the spatial PL variation.

Important: Base commentary only on the numerical descriptors provided. Do not invent mechanisms not supported by the data."""


# ══════════════════════════════════════════════
#  TAB 8 — Phase 3: Captions & Results helpers
# ══════════════════════════════════════════════

_T8_CAP_TAGS = ["FIGURE_CAPTION", "RESULTS_PARAGRAPH"]

_T8_CAP_FIG_TYPES = [
    "Rate Equation Fit",
    "Lifetime Comparison",
    "CIE Diagram",
    "Mapping Heatmap",
]


def _t8_parse_cap_tags(text: str) -> dict:
    """Extract Caption-mode fixed-tag sections from LLM output."""
    import re as _re
    result = {}
    for tag in _T8_CAP_TAGS:
        pat = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_T8_CAP_TAGS)})\]|$)"
        m = _re.search(pat, text, flags=_re.DOTALL | _re.IGNORECASE)
        result[tag] = m.group(1).strip() if m else "(This section was not returned by the model.)"
    return result


def _t8_cap_payload_rateq(rr: dict) -> str:
    """Build caption payload from a parsed rate-equation result dict."""
    import math as _math

    def _fmt(val, spec, nan_str="n/a"):
        if val is None or (isinstance(val, float) and _math.isnan(val)):
            return nan_str
        return format(val, spec)

    kr1, knr1, kcap, kr2, knr2, N_def, beta = rr.get("params_fit",
        [rr.get("kr1"), rr.get("knr1"), rr.get("kcap"),
         rr.get("kr2"), rr.get("knr2"), rr.get("N_def"), rr.get("beta")])

    lines = [
        "=== Rate Equation Fit Summary ===",
        "",
        f"Sample label:   {rr.get('label', 'unnamed')}",
        f"Fit mode:       {rr.get('ch1_label', 'BB')} + YB (dual-channel rate equation)",
        "",
        "--- Derived quantities ---",
        f"τ₁ (Ch1 lifetime):  {_fmt(rr.get('tau1'), '.3e')} s  "
            f"({_fmt(rr.get('tau1', 0) * 1e9 if rr.get('tau1') else None, '.2f')} ns)",
        f"τ₂ (YB lifetime):   {_fmt(rr.get('tau2'), '.3e')} s  "
            f"({_fmt(rr.get('tau2', 0) * 1e6 if rr.get('tau2') else None, '.3f')} µs)",
        f"η Ch1 (radiative efficiency):  {_fmt(rr.get('eta_ch1', 0) * 100 if rr.get('eta_ch1') is not None else None, '.2f')} %",
        f"η YB  (radiative efficiency):  {_fmt(rr.get('eta_ch2', 0) * 100 if rr.get('eta_ch2') is not None else None, '.2f')} %",
        f"R² (Ch1):  {_fmt(rr.get('r2_ch1'), '.4f')}",
        f"R² (YB):   {_fmt(rr.get('r2_ch2'), '.4f')}",
        f"Best loss: {_fmt(rr.get('best_loss'), '.5f')}",
        "",
        "--- Fitted rate constants ---",
        f"k_r1      = {_fmt(kr1,  '.3e')} s⁻¹  (radiative, Ch1)",
        f"k_nr1     = {_fmt(knr1, '.3e')} s⁻¹  (non-radiative, Ch1)",
        f"k_capture = {_fmt(kcap, '.3e')} s⁻¹  (Ch1 → YB)",
        f"k_r2      = {_fmt(kr2,  '.3e')} s⁻¹  (radiative, YB)",
        f"k_nr2     = {_fmt(knr2, '.3e')} s⁻¹  (non-radiative, YB)",
        f"N_defect  = {_fmt(N_def, '.3e')}",
        f"β         = {_fmt(beta,  '.3e')}",
    ]
    return "\n".join(lines)


def _t8_cap_payload_lifetime(included: list, lc_dir: str) -> str:
    """Build caption payload from Tab 5 artifact records."""
    import math as _math

    def _fmt(val, spec, nan_str="n/a"):
        if val is None or (isinstance(val, float) and _math.isnan(val)):
            return nan_str
        return format(val, spec)

    n = len(included)
    lines = [
        f"=== Lifetime Comparison Summary ({n} sample{'s' if n != 1 else ''}) ===",
        "",
        f"Source folder: {lc_dir}",
        "",
        "--- Samples (in X-order) ---",
        f"{'Label':<28} {'τ₁ (ns)':>10} {'τ₂ (µs)':>10} {'η BB (%)':>10} {'η YB (%)':>10}",
        "-" * 70,
    ]
    tau1_vals, tau2_vals, ebb_vals, eyb_vals = [], [], [], []
    for r in included:
        tau1_ns = r.get("tau1")
        tau2_us = r.get("tau2")
        ebb     = r.get("eff_bb")
        eyb     = r.get("eff_yb")
        lines.append(
            f"{r['label']:<28} "
            f"{_fmt(tau1_ns, '.2f'):>10} "
            f"{_fmt(tau2_us, '.2f'):>10} "
            f"{_fmt(ebb, '.2f'):>10} "
            f"{_fmt(eyb, '.2f'):>10}"
        )
        if tau1_ns is not None: tau1_vals.append(tau1_ns)
        if tau2_us is not None: tau2_vals.append(tau2_us)
        if ebb     is not None: ebb_vals.append(ebb)
        if eyb     is not None: eyb_vals.append(eyb)

    lines.append("")
    lines.append("--- Ensemble range ---")
    if tau1_vals:
        lines.append(f"τ₁ range: {min(tau1_vals):.2f} – {max(tau1_vals):.2f} ns")
    if tau2_vals:
        lines.append(f"τ₂ range: {min(tau2_vals):.2f} – {max(tau2_vals):.2f} µs")
    if ebb_vals:
        lines.append(f"η BB range: {min(ebb_vals):.2f} – {max(ebb_vals):.2f} %")
    if eyb_vals:
        lines.append(f"η YB range: {min(eyb_vals):.2f} – {max(eyb_vals):.2f} %")
    return "\n".join(lines)


def _t8_cap_payload_cie(points: list) -> str:
    """Build caption payload from Tab 6 artifact (list of {label, x, y} dicts)."""
    import math as _math
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    WX, WY = 0.3127, 0.3290

    lines = [
        f"=== CIE Diagram Summary ({len(points)} sample{'s' if len(points) != 1 else ''}) ===",
        "",
        f"{'Label':<28} {'x':>8} {'y':>8} {'Dist-D65':>10}",
        "-" * 58,
    ]
    for p in points:
        d = _math.hypot(p["x"] - WX, p["y"] - WY)
        lines.append(f"{p['label']:<28} {p['x']:>8.4f} {p['y']:>8.4f} {d:>10.4f}")

    lines += [
        "",
        "--- Ensemble ---",
        f"Centroid:  x = {cx:.4f},  y = {cy:.4f}",
        f"x range:   {min(xs):.4f} – {max(xs):.4f}  (spread {max(xs)-min(xs):.4f})",
        f"y range:   {min(ys):.4f} – {max(ys):.4f}  (spread {max(ys)-min(ys):.4f})",
        f"Centroid distance to D65 (0.3127, 0.3290): {_math.hypot(cx-WX, cy-WY):.4f}",
    ]
    return "\n".join(lines)


def _t8_cap_payload_mapping(x_coords, y_coords, data_m,
                             do_detect, det_pct, det_abs) -> str:
    """Build caption payload from Tab 7 artifact (compact version for caption use)."""
    import math as _math
    all_vals = data_m[np.isfinite(data_m)]
    nx, ny   = len(x_coords), len(y_coords)
    n_total  = nx * ny
    x_step   = float(np.median(np.diff(x_coords))) if nx > 1 else 0.0
    y_step   = float(np.median(np.diff(y_coords))) if ny > 1 else 0.0

    lines = [
        "=== Mapping Heatmap Summary ===",
        "",
        f"Grid:   {nx} × {ny} points",
        f"X range: {float(x_coords[0]):.1f} – {float(x_coords[-1]):.1f}  (step {x_step:.1f})",
        f"Y range: {float(y_coords[0]):.1f} – {float(y_coords[-1]):.1f}  (step {y_step:.1f})",
        "",
        "--- Intensity statistics ---",
        f"Mean:   {float(np.mean(all_vals)):.2f}",
        f"Std:    {float(np.std(all_vals)):.2f}",
        f"Min:    {float(np.min(all_vals)):.2f}  |  Max: {float(np.max(all_vals)):.2f}",
        f"Median: {float(np.median(all_vals)):.2f}",
    ]

    if do_detect:
        thresh = float(np.percentile(all_vals, det_pct))
        if det_abs is not None:
            mask = (data_m < thresh) | (data_m < det_abs)
        else:
            mask = data_m < thresh
        n_impl = int(np.sum(mask))
        impl_mean    = float(np.nanmean(data_m[mask]))    if n_impl > 0 else float("nan")
        nonimpl_mean = float(np.nanmean(data_m[~mask]))   if (n_total - n_impl) > 0 else float("nan")
        ratio = nonimpl_mean / impl_mean if (impl_mean > 0 and not _math.isnan(impl_mean)) else float("nan")
        lines += [
            "",
            "--- Ion implantation detection ---",
            f"Detected sites: {n_impl} / {n_total}  ({n_impl/n_total*100:.1f}%)",
            f"Mean intensity — implanted: {impl_mean:.2f}  |  non-implanted: {nonimpl_mean:.2f}",
            f"Intensity ratio (non-impl / impl): {ratio:.2f}×",
        ]
    else:
        lines += ["", "Detection: not enabled."]

    return "\n".join(lines)


_T8_CAP_SYSTEM_PROMPT = """You are an expert scientific writer assisting a GaN photoluminescence (PL) researcher.

You will receive a structured numerical summary of one figure produced by the researcher's data analysis platform. Your task is to generate two pieces of scientific writing based only on the data provided.

Physical background (for reference only — use what is relevant):
- GaN PL spectra typically contain UVB (~362 nm, near-band-edge), BB (~460 nm, blue defect), and YB (~560 nm, yellow defect) emission bands.
- Ion implantation quenches UVB/BB emission at implanted sites, reducing PL intensity locally.
- Rate equation fitting extracts carrier lifetimes (τ) and radiative efficiencies (η) for each emission channel.
- CIE chromaticity coordinates reflect the spectral colour balance; shifts indicate changes in relative band intensities.
- Mapping heatmaps show spatial variation of integrated PL intensity across the sample surface.

Output format: You MUST use exactly these two section tags, in this order, with no extra text before or after them:

[FIGURE_CAPTION]
Write 2–4 sentences in the style of a journal figure caption (legend). Requirements:
- Concise, precise, and factual
- Include key numerical values (e.g. τ, η, R², grid size, intensity ratio)
- Use passive voice where natural ("Fitted parameters are shown…", "The map was acquired…")
- Do not speculate beyond what the data shows
- Do not use first person

[RESULTS_PARAGRAPH]
Write 3–5 sentences in the style of a Results & Discussion section paragraph. Requirements:
- Begin with the main observation or finding
- Include specific numerical evidence
- Offer a brief physical interpretation where clearly supported by the data
- Do not over-state conclusions or introduce mechanisms not evidenced in the data
- Write in past tense, third person

Important: Base your writing strictly on the numerical summary provided. Do not invent values or claim results not present in the data."""


# ══════════════════════════════════════════════
#  TAB 8 UI
# ══════════════════════════════════════════════
with tab8:
    st.markdown("### 🤖 AI Physics Copilot")

    # ── Top-level mode selector ───────────────────────────────────────────────
    _t8_top_mode = st.radio(
        "Analysis mode",
        options=[
            "📄 Single sample",
            "🗂️ Multi-sample comparison",
            "🎨 CIE analysis",
            "🗺️ Mapping analysis",
            "📝 Captions & Results",
        ],
        horizontal=True,
        key="t8_top_mode",
    )
    st.divider()

    # ════════════════════════════════════════
    #  PHASE 1 — Single sample
    # ════════════════════════════════════════
    if _t8_top_mode == "📄 Single sample":
        st.caption(
            "Generates an editable physics interpretation draft from your Tab 4 Step 3 "
            "rate-equation fitting results. All numbers are computed by Python; the AI only "
            "produces the narrative interpretation."
        )

        # ── Initialise session state ──────────────────────────────────────────
        for _k, _v in [
            ("t8_input_mode",   "session"),
            ("t8_sample_label", ""),
            ("t8_sample_note",  ""),
            ("t8_groq_model",   _T8_GROQ_MODELS[0]),
            ("t8_raw_output",   ""),
            ("t8_sections",     {}),
            ("t8_edit_version", 0),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v
        for _tag in _T8_TAGS:
            if f"t8_edit_{_tag}" not in st.session_state:
                st.session_state[f"t8_edit_{_tag}"] = ""

        # ── Section 1: Input source ───────────────────────────────────────────
        st.markdown("#### 1 · Input source")

        _t8_mode = st.radio(
            "Where should the AI read fitting results from?",
            options=["Use current session (Tab 4 Step 3)", "Upload a saved result .txt file"],
            horizontal=True,
            key="t8_input_mode_radio",
        )
        _t8_using_session = _t8_mode.startswith("Use current")

        _t8_rr = None
        _t8_ll = []
        _t8_source_ok = False

        if _t8_using_session:
            _re_keys = [k for k in st.session_state if k.startswith("re_result__")]
            if not _re_keys:
                st.info(
                    "No rate-equation results found in this session. "
                    "Please run Tab 4 → Step 3 first, then return here."
                )
            else:
                _re_display = {k.replace("re_result__", ""): k for k in _re_keys}
                _chosen_dir = st.selectbox(
                    "Select result set",
                    options=list(_re_display.keys()),
                    help="Corresponds to the Power Series directory used in Tab 4.",
                )
                _t8_rr = st.session_state[_re_display[_chosen_dir]]
                _t8_ll = st.session_state.get(f"ps_ll_summary__{_chosen_dir}", [])
                _t8_source_ok = True

                with st.expander("📋 Result preview", expanded=False):
                    import math as _math
                    _ch1 = _t8_rr["ch1_label"]
                    _r1  = _t8_rr["r2_ch1"]
                    _r2  = _t8_rr["r2_ch2"]
                    _prev = {
                        f"Channel 1 ({_ch1})": "—",
                        f"Channel 2 (YB)": "—",
                        f"τ₁ ({_ch1})":    f"{_t8_rr['tau1']*1e9:.2f} ns",
                        "τ₂ (YB)":        f"{_t8_rr['tau2']*1e6:.3f} µs",
                        f"η ({_ch1})":    f"{_t8_rr['eta_ch1']*100:.2f}%",
                        "η (YB)":         f"{_t8_rr['eta_ch2']*100:.2f}%",
                        f"R² ({_ch1})":   f"{_r1:.4f}" if not _math.isnan(_r1) else "n/a",
                        "R² (YB)":        f"{_r2:.4f}" if not _math.isnan(_r2) else "n/a",
                        "Best loss":      f"{_t8_rr['best_loss']:.5f}",
                        "Runs":           _t8_rr.get("num_runs_used", "?"),
                    }
                    for _pk, _pv in _prev.items():
                        if _pv == "—":
                            st.write(f"**{_pk}**")
                        else:
                            st.write(f"**{_pk}:** {_pv}")

        else:
            _uploaded_txt = st.file_uploader(
                "Upload a rate-equation result .txt file",
                type=["txt"],
                help="Use the file saved by the 'Save results' button in Tab 4 Step 3.",
            )
            if _uploaded_txt is not None:
                _raw_content = _uploaded_txt.read().decode("utf-8", errors="replace")
                _t8_rr = _t8_parse_saved_txt(_raw_content)
                if _t8_rr is None or _t8_rr.get("_parse_error"):
                    _missing = (_t8_rr or {}).get("_missing", ["k_r1", "k_nr1", "k_capture", "k_r2", "k_nr2"])
                    st.error(
                        f"Could not parse required rate-equation parameters from this txt file.  \n"
                        f"Missing fields: **{', '.join(_missing)}**  \n"
                        f"The file must contain the five core rate constants "
                        f"(k_r1, k_nr1, k_capture, k_r2, k_nr2) with ':' or '=' separators."
                    )
                    _t8_rr = None
                else:
                    _t8_source_ok = True
                    import math as _math
                    _uch1 = _t8_rr['ch1_label']
                    _ur1  = _t8_rr['r2_ch1']
                    _ur2  = _t8_rr['r2_ch2']
                    st.success(
                        f"✅ Parsed successfully.  \n"
                        f"**Channel 1:** {_uch1} · "
                        f"τ₁ = {_t8_rr['tau1']*1e9:.2f} ns · "
                        f"η = {_t8_rr['eta_ch1']*100:.2f}% · "
                        f"R² = {'n/a' if _math.isnan(_ur1) else f'{_ur1:.4f}'}  \n"
                        f"**Channel 2:** YB · "
                        f"τ₂ = {_t8_rr['tau2']*1e6:.3f} µs · "
                        f"η = {_t8_rr['eta_ch2']*100:.2f}% · "
                        f"R² = {'n/a' if _math.isnan(_ur2) else f'{_ur2:.4f}'}"
                    )

        # ── Section 2: Sample context ─────────────────────────────────────────
        st.markdown("#### 2 · Sample context")
        _col_lbl, _col_note = st.columns([1, 2])
        with _col_lbl:
            _t8_sample_label = st.text_input(
                "Sample label",
                value=st.session_state["t8_sample_label"],
                placeholder="e.g. GaN-Si-1MeV-1e14",
                key="t8_sample_label_input",
            )
            st.session_state["t8_sample_label"] = _t8_sample_label
        with _col_note:
            _t8_sample_note = st.text_area(
                "Researcher notes",
                value=st.session_state["t8_sample_note"],
                placeholder=(
                    "Describe implant conditions, anneal temperature, known sample history, "
                    "or any hypothesis you want the AI to address…"
                ),
                height=90,
                key="t8_sample_note_input",
            )
            st.session_state["t8_sample_note"] = _t8_sample_note

        # ── Section 3: Groq model ─────────────────────────────────────────────
        st.markdown("#### 3 · Groq model")
        _groq_key_status = (
            "✅ API key configured" if _T8_GROQ_KEY else "❌ GROQ_API_KEY not found in secrets.toml"
        )
        _groq_pkg_status = "✅ Package ready" if _T8_GROQ_PKG else "❌ langchain-openai not installed"
        st.caption(f"{_groq_key_status}   ·   {_groq_pkg_status}")

        _t8_groq_model = st.selectbox(
            "Select Groq model",
            options=_T8_GROQ_MODELS,
            index=_T8_GROQ_MODELS.index(
                st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                if st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0]) in _T8_GROQ_MODELS
                else _T8_GROQ_MODELS[0]
            ),
            key="t8_groq_model_select",
            help=(
                "qwen/qwen3-32b: rich, nuanced style; chain-of-thought stripped automatically.\n\n"
                "llama-3.3-70b-versatile: concise, structured, production-stable.\n\n"
                "openai/gpt-oss-120b: highest-capability reasoning; largest model."
            ),
        )
        st.session_state["t8_groq_model"] = _t8_groq_model

        # ── Section 4: Generate ───────────────────────────────────────────────
        st.markdown("#### 4 · Generate interpretation")

        if st.button(
            "▶ Generate AI Interpretation",
            type="primary",
            disabled=not _t8_source_ok,
            key="t8_generate_btn",
        ):
            _payload = _t8_build_payload(
                _t8_rr, _t8_ll,
                st.session_state["t8_sample_label"],
                st.session_state["t8_sample_note"],
            )
            _user_prompt = (
                "Please analyse the following GaN PL rate-equation fitting results and "
                "provide your interpretation using the required section tags.\n\n"
                + _payload
            )
            _chosen_model = st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
            with st.spinner(f"Calling Groq ({_chosen_model}) — this usually takes 5–20 seconds…"):
                _raw = _t8_call_groq(_T8_SYSTEM_PROMPT, _user_prompt, _chosen_model)

            st.session_state["t8_raw_output"] = _raw
            if _raw.startswith("[ERROR]"):
                st.error(_raw)
                st.session_state["t8_sections"] = {}
            else:
                _sections = _t8_parse_tags(_raw)
                st.session_state["t8_sections"] = _sections
                for _tag in _T8_TAGS:
                    st.session_state[f"t8_edit_{_tag}"] = _sections[_tag]
                st.session_state["t8_edit_version"] = st.session_state.get("t8_edit_version", 0) + 1

        # ── Section 5: Editable output ────────────────────────────────────────
        if st.session_state.get("t8_sections"):
            st.markdown("#### 5 · Interpretation draft  *(editable)*")
            st.caption(
                "All sections are editable. Revise the AI draft before exporting. "
                "Re-running the generator will overwrite your edits."
            )
            _TAG_LABELS = {
                "CORE_FINDINGS":           "🔑 Core Findings",
                "PHYSICAL_INTERPRETATION": "⚛️ Physical Interpretation",
                "CONFIDENCE_AND_LIMITS":   "⚠️ Confidence & Limits",
                "NEXT_EXPERIMENTS":        "🧪 Next Experiments",
            }
            for _tag in _T8_TAGS:
                with st.expander(_TAG_LABELS[_tag], expanded=True):
                    _t8_ver = st.session_state.get("t8_edit_version", 0)
                    _edited = st.text_area(
                        label=_TAG_LABELS[_tag],
                        value=st.session_state.get(f"t8_edit_{_tag}", ""),
                        height=180,
                        key=f"t8_textarea_{_tag}_v{_t8_ver}",
                        label_visibility="collapsed",
                    )
                    st.session_state[f"t8_edit_{_tag}"] = _edited

            # ── Export ────────────────────────────────────────────────────────
            st.markdown("#### 6 · Export")
            _export_label = st.session_state.get("t8_sample_label", "") or "sample"
            _export_lines = [
                f"GaN PL AI Interpretation — {_export_label}",
                "=" * 60,
                f"Sample: {st.session_state.get('t8_sample_label', '')}",
                f"Notes:  {st.session_state.get('t8_sample_note', '')}",
                f"Model:  {st.session_state.get('t8_groq_model', '')}",
                "",
            ]
            for _tag in _T8_TAGS:
                _export_lines.append(f"[{_tag}]")
                _export_lines.append(st.session_state.get(f"t8_edit_{_tag}", ""))
                _export_lines.append("")
            _export_text = "\n".join(_export_lines)

            _dl_col1, _dl_col2 = st.columns(2)
            with _dl_col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=_export_text.encode("utf-8"),
                    file_name="single_sample_analysis.txt",
                    mime="text/plain",
                    key="t8_dl_txt",
                )
            with _dl_col2:
                _md_lines = [
                    f"# GaN PL AI Interpretation — {_export_label}",
                    "",
                    f"**Sample:** {st.session_state.get('t8_sample_label', '')}  ",
                    f"**Notes:** {st.session_state.get('t8_sample_note', '')}  ",
                    f"**Model:** {st.session_state.get('t8_groq_model', '')}",
                    "",
                ]
                _MD_HEADERS_P1 = {
                    "CORE_FINDINGS":           "## 🔑 Core Findings",
                    "PHYSICAL_INTERPRETATION": "## ⚛️ Physical Interpretation",
                    "CONFIDENCE_AND_LIMITS":   "## ⚠️ Confidence & Limits",
                    "NEXT_EXPERIMENTS":        "## 🧪 Next Experiments",
                }
                for _tag in _T8_TAGS:
                    _md_lines.append(_MD_HEADERS_P1[_tag])
                    _md_lines.append("")
                    _md_lines.append(st.session_state.get(f"t8_edit_{_tag}", ""))
                    _md_lines.append("")
                _md_text = "\n".join(_md_lines)
                st.download_button(
                    "⬇️ Download as .md",
                    data=_md_text.encode("utf-8"),
                    file_name="single_sample_analysis.md",
                    mime="text/markdown",
                    key="t8_dl_md",
                )

            with st.expander("🔧 Raw model output (debug)", expanded=False):
                st.text(st.session_state.get("t8_raw_output", ""))

    # ════════════════════════════════════════
    #  PHASE 2 — Multi-sample comparison (inherits Tab 5)
    # ════════════════════════════════════════
    elif _t8_top_mode == "🗂️ Multi-sample comparison":
        st.caption(
            "Inherits the current Tab 5 (Lifetime Compare) result set and generates "
            "a multi-sample AI interpretation. Go to Tab 5 first to load and configure your files."
        )

        # ── P2 Session state init ─────────────────────────────────────────────
        for _k, _v in [
            ("t8_p2_raw_output",   ""),
            ("t8_p2_sections",     {}),
            ("t8_p2_edit_version", 0),
            ("t8_p2_context",      ""),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v
        for _tag in _T8_P2_TAGS:
            if f"t8_p2_edit_{_tag}" not in st.session_state:
                st.session_state[f"t8_p2_edit_{_tag}"] = ""

        # ── Section P2-1: Tab 5 artifact check ───────────────────────────────
        _p2_ready = st.session_state.get("t5_ai_ready", False)
        _p2_dir   = st.session_state.get("t5_ai_dir", "")

        if not _p2_ready:
            st.info(
                "No Lifetime Compare results found in this session.  \n"
                "Go to **Tab 5 → Lifetime Compare**, load a results folder and "
                "include at least one file, then return here."
            )
        else:
            st.success(
                f"Loaded from Tab 5 — folder: `{_p2_dir}`  \n"
                f"{len(st.session_state.get('t5_ai_included', []))} file(s) included."
            )

            # ── Section P2-2: Scan & parse (from t5_ai_dir) ──────────────────
            _p2_included = st.session_state.get("t5_ai_included", [])
            # full parse using _t8_parse_saved_txt for rate constants
            _p2_records = []
            _p2_parse_errors = []
            for _item in _p2_included:
                _fpath = os.path.join(_p2_dir, _item["file"])
                try:
                    _content = open(_fpath, encoding="utf-8", errors="replace").read()
                    _p = _t8_parse_saved_txt(_content)
                    if _p is None or _p.get("_parse_error"):
                        _p2_parse_errors.append(
                            f"{_item['file']} (missing: {', '.join(_p.get('_missing', []) if _p else ['unknown'])})"
                        )
                    else:
                        _p["label"] = _item["label"]
                        _p2_records.append(_p)
                except Exception as _e:
                    _p2_parse_errors.append(f"{_item['file']} ({_e})")

            if _p2_parse_errors:
                st.warning(
                    f"⚠️ {len(_p2_parse_errors)} file(s) could not be fully parsed "
                    f"(rate constants missing): " + ", ".join(_p2_parse_errors)
                )

            if not _p2_records:
                st.info("None of the included files could be parsed successfully.")
            else:
                # ── Section P2-3: Comparison table ───────────────────────────
                st.markdown("#### 1 · Parsed comparison table")
                _p2_tbl = []
                for _r in _p2_records:
                    _p2_tbl.append({
                        "Label":      _r["label"],
                        "τ₁ (ns)":   f"{_r['tau1']*1e9:.2f}"    if _r.get("tau1")    else "—",
                        "τ₂ (µs)":   f"{_r['tau2']*1e6:.3f}"    if _r.get("tau2")    else "—",
                        "η Ch1 (%)": f"{_r['eta_ch1']*100:.2f}" if _r.get("eta_ch1") else "—",
                        "η YB (%)":  f"{_r['eta_ch2']*100:.2f}" if _r.get("eta_ch2") else "—",
                        "R² Ch1":    f"{_r['r2_ch1']:.4f}"       if not (isinstance(_r.get("r2_ch1"), float) and __import__("math").isnan(_r.get("r2_ch1", float("nan")))) else "—",
                        "R² YB":     f"{_r['r2_ch2']:.4f}"       if not (isinstance(_r.get("r2_ch2"), float) and __import__("math").isnan(_r.get("r2_ch2", float("nan")))) else "—",
                    })
                st.dataframe(pd.DataFrame(_p2_tbl), width="stretch", hide_index=True)

                # ── Section P2-4: Researcher notes ───────────────────────────
                st.markdown("#### 2 · Researcher notes")
                _p2_ctx = st.text_area(
                    "Optional context (experimental conditions, sample differences, etc.)",
                    value=st.session_state.get("t8_p2_context", ""),
                    height=90,
                    key="t8_p2_context_area",
                )
                st.session_state["t8_p2_context"] = _p2_ctx

                # ── Section P2-5: Model selection ─────────────────────────────
                st.markdown("#### 3 · Model")
                _p2_key_s = "✅ API key configured" if _T8_GROQ_KEY else "❌ GROQ_API_KEY not found"
                _p2_pkg_s = "✅ Package ready"      if _T8_GROQ_PKG else "❌ langchain-openai not installed"
                st.caption(f"{_p2_key_s}   {_p2_pkg_s}")
                _p2_model = st.selectbox(
                    "Groq model",
                    options=_T8_GROQ_MODELS,
                    index=(
                        _T8_GROQ_MODELS.index(
                            st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                        )
                        if st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                        in _T8_GROQ_MODELS else 0
                    ),
                    key="t8_p2_groq_model_select",
                )
                st.session_state["t8_groq_model"] = _p2_model

                # ── Section P2-6: Generate ────────────────────────────────────
                st.markdown("#### 4 · Generate comparison interpretation")
                if st.button(
                    "▶ Generate Multi-sample Interpretation",
                    type="primary",
                    key="t8_p2_generate_btn",
                    disabled=(not _T8_GROQ_KEY or not _T8_GROQ_PKG),
                ):
                    _p2_payload     = _t8_build_comparison_payload(_p2_records)
                    _p2_ctx_val     = st.session_state["t8_p2_context"].strip()
                    _p2_user_prompt = (
                        "Please analyse the following multi-sample GaN PL comparison and "
                        "provide your interpretation using the required section tags.\n\n"
                    )
                    if _p2_ctx_val:
                        _p2_user_prompt += f"Researcher context: {_p2_ctx_val}\n\n"
                    _p2_user_prompt += _p2_payload

                    _p2_chosen = st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    with st.spinner(f"Calling Groq ({_p2_chosen}) — this usually takes 5–30 seconds…"):
                        _p2_raw = _t8_call_groq(_T8_P2_SYSTEM_PROMPT, _p2_user_prompt, _p2_chosen)

                    st.session_state["t8_p2_raw_output"] = _p2_raw
                    if _p2_raw.startswith("[ERROR]"):
                        st.error(_p2_raw)
                        st.session_state["t8_p2_sections"] = {}
                    else:
                        _p2_secs    = _t8_parse_p2_tags(_p2_raw)
                        _p2_missing = [t for t in _T8_P2_TAGS
                                       if _p2_secs[t] == "(This section was not returned by the model.)"]
                        if _p2_missing:
                            _rescue_prompt = (
                                "Your previous response was missing these required sections: "
                                + ", ".join(f"[{t}]" for t in _p2_missing)
                                + ".\n\nPlease provide ONLY those missing sections now, "
                                "using the exact tag format. Do not repeat sections already provided. "
                                "Base your answer on the same data as before.\n\n"
                                + "\n".join(f"[{t}]\n(your content here)" for t in _p2_missing)
                            )
                            with st.spinner(f"Some sections were missing — requesting completion ({', '.join(_p2_missing)})…"):
                                _rescue_raw = _t8_call_groq(_T8_P2_SYSTEM_PROMPT, _rescue_prompt, _p2_chosen)
                            if not _rescue_raw.startswith("[ERROR]"):
                                _rescue_secs = _t8_parse_p2_tags(_rescue_raw)
                                for _t in _p2_missing:
                                    if _rescue_secs[_t] != "(This section was not returned by the model.)":
                                        _p2_secs[_t] = _rescue_secs[_t]
                                st.session_state["t8_p2_raw_output"] = (
                                    _p2_raw + "\n\n--- rescue call ---\n" + _rescue_raw
                                )
                            _still_missing = [t for t in _T8_P2_TAGS
                                              if _p2_secs[t] == "(This section was not returned by the model.)"]
                            if _still_missing:
                                st.warning(
                                    f"⚠️ Section(s) still missing after retry: **{', '.join(_still_missing)}**. "
                                    f"Try switching to `qwen/qwen3-32b` or `llama-3.3-70b-versatile`."
                                )

                        st.session_state["t8_p2_sections"] = _p2_secs
                        for _tag in _T8_P2_TAGS:
                            st.session_state[f"t8_p2_edit_{_tag}"] = _p2_secs[_tag]
                        st.session_state["t8_p2_edit_version"] = (
                            st.session_state.get("t8_p2_edit_version", 0) + 1
                        )

                # ── Section P2-7: Editable output ─────────────────────────────
                if st.session_state.get("t8_p2_sections"):
                    st.markdown("#### 5 · Comparison draft  *(editable)*")
                    st.caption("All sections are editable. Re-running the generator will overwrite your edits.")
                    _P2_TAG_LABELS = {
                        "TREND_SUMMARY":        "📡 Trend Summary",
                        "ANOMALIES":            "⚠️ Anomalies",
                        "SUGGESTED_MECHANISMS": "🧬 Suggested Mechanisms",
                        "NEXT_EXPERIMENTS":     "🧪 Next Experiments",
                    }
                    for _tag in _T8_P2_TAGS:
                        with st.expander(_P2_TAG_LABELS[_tag], expanded=True):
                            _p2_ev = st.session_state.get("t8_p2_edit_version", 0)
                            _p2_edited_text = st.text_area(
                                label=_P2_TAG_LABELS[_tag],
                                value=st.session_state.get(f"t8_p2_edit_{_tag}", ""),
                                height=180,
                                key=f"t8_p2_textarea_{_tag}_v{_p2_ev}",
                                label_visibility="collapsed",
                            )
                            st.session_state[f"t8_p2_edit_{_tag}"] = _p2_edited_text

                    # ── Export ─────────────────────────────────────────────────
                    st.markdown("#### 6 · Export")
                    _p2_sample_list = ", ".join(r["label"] for r in _p2_records)
                    _p2_export_lines = [
                        "GaN PL AI Multi-sample Comparison",
                        "=" * 60,
                        f"Samples: {_p2_sample_list}",
                        f"Model:   {st.session_state.get('t8_groq_model', '')}",
                        f"Context: {st.session_state.get('t8_p2_context', '')}",
                        "",
                    ]
                    for _tag in _T8_P2_TAGS:
                        _p2_export_lines.append(f"[{_tag}]")
                        _p2_export_lines.append(st.session_state.get(f"t8_p2_edit_{_tag}", ""))
                        _p2_export_lines.append("")
                    _p2_export_text = "\n".join(_p2_export_lines)

                    _p2_dl1, _p2_dl2 = st.columns(2)
                    with _p2_dl1:
                        st.download_button(
                            "⬇️ Download as .txt",
                            data=_p2_export_text.encode("utf-8"),
                            file_name="multi_sample_analysis.txt",
                            mime="text/plain",
                            key="t8_p2_dl_txt",
                        )
                    with _p2_dl2:
                        _p2_md_lines = [
                            "# GaN PL AI Multi-sample Comparison",
                            "",
                            f"**Samples:** {_p2_sample_list}  ",
                            f"**Model:** {st.session_state.get('t8_groq_model', '')}  ",
                            f"**Context:** {st.session_state.get('t8_p2_context', '')}",
                            "",
                        ]
                        _P2_MD_HEADERS = {
                            "TREND_SUMMARY":        "## 📡 Trend Summary",
                            "ANOMALIES":            "## ⚠️ Anomalies",
                            "SUGGESTED_MECHANISMS": "## 🧬 Suggested Mechanisms",
                            "NEXT_EXPERIMENTS":     "## 🧪 Next Experiments",
                        }
                        for _tag in _T8_P2_TAGS:
                            _p2_md_lines.append(_P2_MD_HEADERS[_tag])
                            _p2_md_lines.append("")
                            _p2_md_lines.append(st.session_state.get(f"t8_p2_edit_{_tag}", ""))
                            _p2_md_lines.append("")
                        _p2_md_text = "\n".join(_p2_md_lines)
                        st.download_button(
                            "⬇️ Download as .md",
                            data=_p2_md_text.encode("utf-8"),
                            file_name="multi_sample_analysis.md",
                            mime="text/markdown",
                            key="t8_p2_dl_md",
                        )

                    with st.expander("🔧 Raw model output (debug)", expanded=False):
                        st.text(st.session_state.get("t8_p2_raw_output", ""))

    # ════════════════════════════════════════
    #  CIE ANALYSIS MODE (inherits Tab 6)
    # ════════════════════════════════════════
    elif _t8_top_mode == "🎨 CIE analysis":
        st.caption(
            "Inherits the current Tab 6 (CIE Diagram) result set and generates "
            "a chromaticity AI interpretation. Go to Tab 6 first to load your spectra."
        )

        # ── Session state defaults ────────────────────────────────────────────
        for _ck, _cv in [
            ("t8_cie_notes",        ""),
            ("t8_cie_raw_output",   ""),
            ("t8_cie_sections",     {}),
            ("t8_cie_edit_version", 0),
        ]:
            if _ck not in st.session_state:
                st.session_state[_ck] = _cv
        for _ctag in _T8_CIE_TAGS:
            if f"t8_cie_edit_{_ctag}" not in st.session_state:
                st.session_state[f"t8_cie_edit_{_ctag}"] = ""

        _cie_ready  = st.session_state.get("t6_ai_ready", False)
        _cie_points = st.session_state.get("t6_ai_points", [])

        if not _cie_ready or not _cie_points:
            st.info(
                "No CIE data found in this session.  \n"
                "Go to **Tab 6 → CIE Diagram**, load a folder of corrected spectra, "
                "then return here."
            )
        else:
            st.success(f"Loaded from Tab 6 — {len(_cie_points)} sample(s).")

            # ── Computed summary ──────────────────────────────────────────────
            with st.expander("📊 Computed CIE descriptors (Python)", expanded=True):
                _cie_payload = _t8_build_cie_payload(_cie_points)
                st.text(_cie_payload)

            # ── Researcher notes ──────────────────────────────────────────────
            st.markdown("#### 1 · Researcher notes")
            _cie_notes = st.text_area(
                "Optional context (sample IDs, implantation conditions, etc.)",
                value=st.session_state.get("t8_cie_notes", ""),
                height=90,
                key="t8_cie_notes_area",
            )
            st.session_state["t8_cie_notes"] = _cie_notes

            # ── Model selection ───────────────────────────────────────────────
            st.markdown("#### 2 · Model")
            _cie_key_s = "✅ API key configured" if _T8_GROQ_KEY else "❌ GROQ_API_KEY not found"
            _cie_pkg_s = "✅ Package ready"      if _T8_GROQ_PKG else "❌ langchain-openai not installed"
            st.caption(f"{_cie_key_s}   {_cie_pkg_s}")
            _cie_model = st.selectbox(
                "Groq model",
                options=_T8_GROQ_MODELS,
                index=(
                    _T8_GROQ_MODELS.index(
                        st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    )
                    if st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    in _T8_GROQ_MODELS else 0
                ),
                key="t8_cie_groq_model_select",
            )
            st.session_state["t8_groq_model"] = _cie_model

            # ── Generate ──────────────────────────────────────────────────────
            st.markdown("#### 3 · Generate interpretation")
            if st.button(
                "▶ Generate CIE Interpretation",
                type="primary",
                key="t8_cie_generate_btn",
                disabled=(not _T8_GROQ_KEY or not _T8_GROQ_PKG),
            ):
                _cie_user_prompt = (
                    "Please analyse the following GaN PL CIE chromaticity summary and "
                    "provide your interpretation using the required section tags.\n\n"
                )
                if _cie_notes.strip():
                    _cie_user_prompt += f"Researcher context: {_cie_notes.strip()}\n\n"
                _cie_user_prompt += _cie_payload

                _cie_chosen = st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                with st.spinner(f"Calling Groq ({_cie_chosen}) — this usually takes 5–30 seconds…"):
                    _cie_raw = _t8_call_groq(_T8_CIE_SYSTEM_PROMPT, _cie_user_prompt, _cie_chosen)

                st.session_state["t8_cie_raw_output"] = _cie_raw
                if _cie_raw.startswith("[ERROR]"):
                    st.error(_cie_raw)
                    st.session_state["t8_cie_sections"] = {}
                else:
                    _cie_secs    = _t8_parse_cie_tags(_cie_raw)
                    _cie_missing = [t for t in _T8_CIE_TAGS
                                    if _cie_secs[t] == "(This section was not returned by the model.)"]
                    if _cie_missing:
                        _cie_rescue_prompt = (
                            "Your previous response was missing these required sections: "
                            + ", ".join(f"[{t}]" for t in _cie_missing)
                            + ".\n\nPlease provide ONLY those missing sections now, "
                            "using the exact tag format. Do not repeat sections already provided. "
                            "Base your answer on the same data as before.\n\n"
                            + "\n".join(f"[{t}]\n(your content here)" for t in _cie_missing)
                        )
                        with st.spinner(f"Some sections were missing — requesting completion ({', '.join(_cie_missing)})…"):
                            _cie_rescue_raw = _t8_call_groq(
                                _T8_CIE_SYSTEM_PROMPT, _cie_rescue_prompt, _cie_chosen
                            )
                        if not _cie_rescue_raw.startswith("[ERROR]"):
                            _cie_rescue_secs = _t8_parse_cie_tags(_cie_rescue_raw)
                            for _ct in _cie_missing:
                                if _cie_rescue_secs[_ct] != "(This section was not returned by the model.)":
                                    _cie_secs[_ct] = _cie_rescue_secs[_ct]
                            st.session_state["t8_cie_raw_output"] = (
                                _cie_raw + "\n\n--- rescue call ---\n" + _cie_rescue_raw
                            )
                        _cie_still_missing = [t for t in _T8_CIE_TAGS
                                              if _cie_secs[t] == "(This section was not returned by the model.)"]
                        if _cie_still_missing:
                            st.warning(
                                f"⚠️ Section(s) still missing after retry: **{', '.join(_cie_still_missing)}**. "
                                f"Try switching to `qwen/qwen3-32b` or `llama-3.3-70b-versatile`."
                            )

                    st.session_state["t8_cie_sections"] = _cie_secs
                    for _ctag in _T8_CIE_TAGS:
                        st.session_state[f"t8_cie_edit_{_ctag}"] = _cie_secs[_ctag]
                    st.session_state["t8_cie_edit_version"] = (
                        st.session_state.get("t8_cie_edit_version", 0) + 1
                    )

            # ── Editable output ───────────────────────────────────────────────
            if st.session_state.get("t8_cie_sections"):
                st.markdown("#### 4 · CIE interpretation draft  *(editable)*")
                st.caption("All sections are editable. Re-running the generator will overwrite your edits.")
                _CIE_TAG_LABELS = {
                    "CHROMATICITY_OVERVIEW":  "🎨 Chromaticity Overview",
                    "TRENDS_AND_SHIFTS":      "📈 Trends & Shifts",
                    "ANOMALIES_AND_OUTLIERS": "⚠️ Anomalies & Outliers",
                    "NEXT_EXPERIMENTS":       "🧪 Next Experiments",
                }
                for _ctag in _T8_CIE_TAGS:
                    with st.expander(_CIE_TAG_LABELS[_ctag], expanded=True):
                        _cie_ev = st.session_state.get("t8_cie_edit_version", 0)
                        _cie_edited = st.text_area(
                            label=_CIE_TAG_LABELS[_ctag],
                            value=st.session_state.get(f"t8_cie_edit_{_ctag}", ""),
                            height=180,
                            key=f"t8_cie_textarea_{_ctag}_v{_cie_ev}",
                            label_visibility="collapsed",
                        )
                        st.session_state[f"t8_cie_edit_{_ctag}"] = _cie_edited

                # ── Export ────────────────────────────────────────────────────
                st.markdown("#### 5 · Export")
                _cie_label_list = ", ".join(p["label"] for p in _cie_points)
                _cie_export_lines = [
                    "GaN PL AI CIE Chromaticity Analysis",
                    "=" * 60,
                    f"Samples: {_cie_label_list}",
                    f"Model:   {st.session_state.get('t8_groq_model', '')}",
                    f"Context: {st.session_state.get('t8_cie_notes', '')}",
                    "",
                ]
                for _ctag in _T8_CIE_TAGS:
                    _cie_export_lines.append(f"[{_ctag}]")
                    _cie_export_lines.append(st.session_state.get(f"t8_cie_edit_{_ctag}", ""))
                    _cie_export_lines.append("")
                _cie_export_text = "\n".join(_cie_export_lines)

                _cie_dl1, _cie_dl2 = st.columns(2)
                with _cie_dl1:
                    st.download_button(
                        "⬇️ Download as .txt",
                        data=_cie_export_text.encode("utf-8"),
                        file_name="cie_analysis.txt",
                        mime="text/plain",
                        key="t8_cie_dl_txt",
                    )
                with _cie_dl2:
                    _cie_md_lines = [
                        "# GaN PL AI CIE Chromaticity Analysis",
                        "",
                        f"**Samples:** {_cie_label_list}  ",
                        f"**Model:** {st.session_state.get('t8_groq_model', '')}  ",
                        f"**Context:** {st.session_state.get('t8_cie_notes', '')}",
                        "",
                    ]
                    _CIE_MD_HEADERS = {
                        "CHROMATICITY_OVERVIEW":  "## 🎨 Chromaticity Overview",
                        "TRENDS_AND_SHIFTS":      "## 📈 Trends & Shifts",
                        "ANOMALIES_AND_OUTLIERS": "## ⚠️ Anomalies & Outliers",
                        "NEXT_EXPERIMENTS":       "## 🧪 Next Experiments",
                    }
                    for _ctag in _T8_CIE_TAGS:
                        _cie_md_lines.append(_CIE_MD_HEADERS[_ctag])
                        _cie_md_lines.append("")
                        _cie_md_lines.append(st.session_state.get(f"t8_cie_edit_{_ctag}", ""))
                        _cie_md_lines.append("")
                    _cie_md_text = "\n".join(_cie_md_lines)
                    st.download_button(
                        "⬇️ Download as .md",
                        data=_cie_md_text.encode("utf-8"),
                        file_name="cie_analysis.md",
                        mime="text/markdown",
                        key="t8_cie_dl_md",
                    )

                with st.expander("🔧 Raw model output (debug)", expanded=False):
                    st.text(st.session_state.get("t8_cie_raw_output", ""))

    # ════════════════════════════════════════
    #  MAPPING ANALYSIS MODE (inherits Tab 7)
    # ════════════════════════════════════════
    elif _t8_top_mode == "🗺️ Mapping analysis":
        st.caption(
            "Inherits the current Tab 7 (Mapping Heatmap) data and generates "
            "a spatial AI interpretation. Go to Tab 7 first to load your mapping data."
        )

        # ── Session state defaults ────────────────────────────────────────────
        for _mk, _mv in [
            ("t8_map_notes",        ""),
            ("t8_map_raw_output",   ""),
            ("t8_map_sections",     {}),
            ("t8_map_edit_version", 0),
        ]:
            if _mk not in st.session_state:
                st.session_state[_mk] = _mv
        for _mtag in _T8_MAP_TAGS:
            if f"t8_map_edit_{_mtag}" not in st.session_state:
                st.session_state[f"t8_map_edit_{_mtag}"] = ""

        _map_ready  = st.session_state.get("t7_ai_ready", False)
        _map_x      = st.session_state.get("t7_ai_x",        None)
        _map_y      = st.session_state.get("t7_ai_y",        None)
        _map_data   = st.session_state.get("t7_ai_data",     None)
        _m_do_detect = st.session_state.get("t7_ai_do_detect", False)
        _m_det_pct   = float(st.session_state.get("t7_ai_det_pct", 25))
        _m_det_abs   = st.session_state.get("t7_ai_det_abs",  None)

        if not _map_ready or _map_data is None:
            st.info(
                "No mapping data found in this session.  \n"
                "Go to **Tab 7 → Mapping Heatmap**, load a mapping folder, "
                "then return here."
            )
        else:
            _map_dir = st.session_state.get("mapping_dir", "")
            st.success(
                f"Loaded from Tab 7 — folder: `{_map_dir}`  \n"
                f"Grid: {len(_map_x)} × {len(_map_y)} points  |  "
                f"Detection: {'enabled' if _m_do_detect else 'disabled'}"
            )

            # ── Computed summary ──────────────────────────────────────────────
            with st.expander("📊 Computed mapping descriptors (Python)", expanded=True):
                _map_payload = _t8_build_mapping_payload(
                    _map_x, _map_y, _map_data,
                    do_detect=_m_do_detect,
                    det_pct=_m_det_pct,
                    det_abs=_m_det_abs,
                )
                st.text(_map_payload)

            # ── Researcher notes ──────────────────────────────────────────────
            st.markdown("#### 1 · Researcher notes")
            _map_notes = st.text_area(
                "Optional context (sample ID, implantation conditions, imaging setup, etc.)",
                value=st.session_state.get("t8_map_notes", ""),
                height=90,
                key="t8_map_notes_area",
            )
            st.session_state["t8_map_notes"] = _map_notes

            # ── Model selection ───────────────────────────────────────────────
            st.markdown("#### 2 · Model")
            _map_key_s = "✅ API key configured" if _T8_GROQ_KEY else "❌ GROQ_API_KEY not found"
            _map_pkg_s = "✅ Package ready"      if _T8_GROQ_PKG else "❌ langchain-openai not installed"
            st.caption(f"{_map_key_s}   {_map_pkg_s}")
            _map_model = st.selectbox(
                "Groq model",
                options=_T8_GROQ_MODELS,
                index=(
                    _T8_GROQ_MODELS.index(
                        st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    )
                    if st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    in _T8_GROQ_MODELS else 0
                ),
                key="t8_map_groq_model_select",
            )
            st.session_state["t8_groq_model"] = _map_model

            # ── Generate ──────────────────────────────────────────────────────
            st.markdown("#### 3 · Generate interpretation")
            if st.button(
                "▶ Generate Mapping Interpretation",
                type="primary",
                key="t8_map_generate_btn",
                disabled=(not _T8_GROQ_KEY or not _T8_GROQ_PKG),
            ):
                _map_user_prompt = (
                    "Please analyse the following GaN PL mapping data summary and "
                    "provide your interpretation using the required section tags.\n\n"
                )
                if _map_notes.strip():
                    _map_user_prompt += f"Researcher context: {_map_notes.strip()}\n\n"
                _map_user_prompt += _map_payload

                _map_chosen = st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                with st.spinner(f"Calling Groq ({_map_chosen}) — this usually takes 5–30 seconds…"):
                    _map_raw = _t8_call_groq(_T8_MAP_SYSTEM_PROMPT, _map_user_prompt, _map_chosen)

                st.session_state["t8_map_raw_output"] = _map_raw
                if _map_raw.startswith("[ERROR]"):
                    st.error(_map_raw)
                    st.session_state["t8_map_sections"] = {}
                else:
                    _map_secs    = _t8_parse_mapping_tags(_map_raw)
                    _map_missing = [t for t in _T8_MAP_TAGS
                                    if _map_secs[t] == "(This section was not returned by the model.)"]
                    if _map_missing:
                        _map_rescue_prompt = (
                            "Your previous response was missing these required sections: "
                            + ", ".join(f"[{t}]" for t in _map_missing)
                            + ".\n\nPlease provide ONLY those missing sections now, "
                            "using the exact tag format. Do not repeat sections already provided. "
                            "Base your answer on the same data as before.\n\n"
                            + "\n".join(f"[{t}]\n(your content here)" for t in _map_missing)
                        )
                        with st.spinner(f"Some sections were missing — requesting completion ({', '.join(_map_missing)})…"):
                            _map_rescue_raw = _t8_call_groq(
                                _T8_MAP_SYSTEM_PROMPT, _map_rescue_prompt, _map_chosen
                            )
                        if not _map_rescue_raw.startswith("[ERROR]"):
                            _map_rescue_secs = _t8_parse_mapping_tags(_map_rescue_raw)
                            for _mt in _map_missing:
                                if _map_rescue_secs[_mt] != "(This section was not returned by the model.)":
                                    _map_secs[_mt] = _map_rescue_secs[_mt]
                            st.session_state["t8_map_raw_output"] = (
                                _map_raw + "\n\n--- rescue call ---\n" + _map_rescue_raw
                            )
                        _map_still_missing = [t for t in _T8_MAP_TAGS
                                              if _map_secs[t] == "(This section was not returned by the model.)"]
                        if _map_still_missing:
                            st.warning(
                                f"⚠️ Section(s) still missing after retry: **{', '.join(_map_still_missing)}**. "
                                f"Try switching to `qwen/qwen3-32b` or `llama-3.3-70b-versatile`."
                            )

                    st.session_state["t8_map_sections"] = _map_secs
                    for _mtag in _T8_MAP_TAGS:
                        st.session_state[f"t8_map_edit_{_mtag}"] = _map_secs[_mtag]
                    st.session_state["t8_map_edit_version"] = (
                        st.session_state.get("t8_map_edit_version", 0) + 1
                    )

            # ── Editable output ───────────────────────────────────────────────
            if st.session_state.get("t8_map_sections"):
                st.markdown("#### 4 · Mapping interpretation draft  *(editable)*")
                st.caption("All sections are editable. Re-running the generator will overwrite your edits.")
                _MAP_TAG_LABELS = {
                    "SPATIAL_SUMMARY":      "🗺️ Spatial Summary",
                    "REGION_COMPARISON":    "🔍 Region Comparison",
                    "ANOMALIES_AND_LIMITS": "⚠️ Anomalies & Limits",
                    "NEXT_EXPERIMENTS":     "🧪 Next Experiments",
                }
                for _mtag in _T8_MAP_TAGS:
                    with st.expander(_MAP_TAG_LABELS[_mtag], expanded=True):
                        _map_ev = st.session_state.get("t8_map_edit_version", 0)
                        _map_edited = st.text_area(
                            label=_MAP_TAG_LABELS[_mtag],
                            value=st.session_state.get(f"t8_map_edit_{_mtag}", ""),
                            height=180,
                            key=f"t8_map_textarea_{_mtag}_v{_map_ev}",
                            label_visibility="collapsed",
                        )
                        st.session_state[f"t8_map_edit_{_mtag}"] = _map_edited

                # ── Export ────────────────────────────────────────────────────
                st.markdown("#### 5 · Export")
                _map_export_lines = [
                    "GaN PL AI Mapping Analysis",
                    "=" * 60,
                    f"Source:  {st.session_state.get('mapping_dir', '')}",
                    f"Model:   {st.session_state.get('t8_groq_model', '')}",
                    f"Context: {st.session_state.get('t8_map_notes', '')}",
                    "",
                ]
                for _mtag in _T8_MAP_TAGS:
                    _map_export_lines.append(f"[{_mtag}]")
                    _map_export_lines.append(st.session_state.get(f"t8_map_edit_{_mtag}", ""))
                    _map_export_lines.append("")
                _map_export_text = "\n".join(_map_export_lines)

                _map_dl1, _map_dl2 = st.columns(2)
                with _map_dl1:
                    st.download_button(
                        "⬇️ Download as .txt",
                        data=_map_export_text.encode("utf-8"),
                        file_name="mapping_analysis.txt",
                        mime="text/plain",
                        key="t8_map_dl_txt",
                    )
                with _map_dl2:
                    _map_md_lines = [
                        "# GaN PL AI Mapping Analysis",
                        "",
                        f"**Source:** {st.session_state.get('mapping_dir', '')}  ",
                        f"**Model:** {st.session_state.get('t8_groq_model', '')}  ",
                        f"**Context:** {st.session_state.get('t8_map_notes', '')}",
                        "",
                    ]
                    _MAP_MD_HEADERS = {
                        "SPATIAL_SUMMARY":      "## 🗺️ Spatial Summary",
                        "REGION_COMPARISON":    "## 🔍 Region Comparison",
                        "ANOMALIES_AND_LIMITS": "## ⚠️ Anomalies & Limits",
                        "NEXT_EXPERIMENTS":     "## 🧪 Next Experiments",
                    }
                    for _mtag in _T8_MAP_TAGS:
                        _map_md_lines.append(_MAP_MD_HEADERS[_mtag])
                        _map_md_lines.append("")
                        _map_md_lines.append(st.session_state.get(f"t8_map_edit_{_mtag}", ""))
                        _map_md_lines.append("")
                    _map_md_text = "\n".join(_map_md_lines)
                    st.download_button(
                        "⬇️ Download as .md",
                        data=_map_md_text.encode("utf-8"),
                        file_name="mapping_analysis.md",
                        mime="text/markdown",
                        key="t8_map_dl_md",
                    )

                with st.expander("🔧 Raw model output (debug)", expanded=False):
                    st.text(st.session_state.get("t8_map_raw_output", ""))
    # ════════════════════════════════════════
    #  CAPTIONS & RESULTS MODE
    # ════════════════════════════════════════
    elif _t8_top_mode == "📝 Captions & Results":
        st.caption(
            "Generate a figure caption and Results & Discussion paragraph for any of your "
            "analysis figures. Python assembles the numerical summary; the AI writes the text."
        )

        # ── Session state defaults ────────────────────────────────────────────
        for _capk, _capv in [
            ("t8_cap_fig_type",        _T8_CAP_FIG_TYPES[0]),
            ("t8_cap_rateq_mode",      "session"),
            ("t8_cap_notes",           ""),
            ("t8_cap_raw_output",      ""),
            ("t8_cap_sections",        {}),
            ("t8_cap_edit_version",    0),
            ("t8_cap_edit_FIGURE_CAPTION",    ""),
            ("t8_cap_edit_RESULTS_PARAGRAPH", ""),
        ]:
            if _capk not in st.session_state:
                st.session_state[_capk] = _capv

        # ── 1 · Figure type ───────────────────────────────────────────────────
        st.markdown("#### 1 · Figure type")
        _cap_fig_type = st.selectbox(
            "Select the figure to caption",
            options=_T8_CAP_FIG_TYPES,
            index=_T8_CAP_FIG_TYPES.index(
                st.session_state.get("t8_cap_fig_type", _T8_CAP_FIG_TYPES[0])
                if st.session_state.get("t8_cap_fig_type", _T8_CAP_FIG_TYPES[0])
                in _T8_CAP_FIG_TYPES else _T8_CAP_FIG_TYPES[0]
            ),
            key="t8_cap_fig_type_select",
        )
        st.session_state["t8_cap_fig_type"] = _cap_fig_type

        # ── 2 · Data source (conditional on figure type) ──────────────────────
        st.markdown("#### 2 · Data source")
        _cap_payload    = None
        _cap_source_ok  = False
        _cap_source_label = ""

        if _cap_fig_type == "Rate Equation Fit":
            _cap_rateq_src = st.radio(
                "Where should the fitting results come from?",
                options=["Use current session (Tab 4 Step 3)", "Upload a saved result .txt file"],
                horizontal=True,
                key="t8_cap_rateq_src_radio",
            )
            st.session_state["t8_cap_rateq_mode"] = _cap_rateq_src

            _cap_rr = None
            if _cap_rateq_src.startswith("Use current"):
                _cap_re_keys = [k for k in st.session_state if k.startswith("re_result__")]
                if not _cap_re_keys:
                    st.info(
                        "No rate-equation results found in this session. "
                        "Run Tab 4 → Step 3 first, then return here."
                    )
                else:
                    _cap_re_display = {k.replace("re_result__", ""): k for k in _cap_re_keys}
                    _cap_chosen_dir = st.selectbox(
                        "Select result set",
                        options=list(_cap_re_display.keys()),
                        key="t8_cap_rateq_dir_select",
                    )
                    _cap_rr = st.session_state[_cap_re_display[_cap_chosen_dir]]
                    _cap_source_label = _cap_chosen_dir
                    _cap_source_ok = True
            else:
                _cap_uploaded = st.file_uploader(
                    "Upload a rate-equation result .txt file",
                    type=["txt"],
                    key="t8_cap_rateq_uploader",
                )
                if _cap_uploaded is not None:
                    _raw_content = _cap_uploaded.read().decode("utf-8", errors="replace")
                    _cap_rr = _t8_parse_saved_txt(_raw_content)
                    if _cap_rr is None or _cap_rr.get("_parse_error"):
                        _missing = _cap_rr.get("_missing", []) if _cap_rr else ["unknown"]
                        st.error(
                            f"Could not parse file. Missing fields: {', '.join(_missing)}"
                        )
                    else:
                        _cap_source_label = _cap_uploaded.name
                        _cap_source_ok = True
                else:
                    st.info("Upload a rate-equation result .txt file to continue.")

            if _cap_source_ok and _cap_rr is not None:
                _cap_rr["label"] = _cap_source_label
                _cap_payload = _t8_cap_payload_rateq(_cap_rr)

        elif _cap_fig_type == "Lifetime Comparison":
            _cap_t5_ready    = st.session_state.get("t5_ai_ready", False)
            _cap_t5_dir      = st.session_state.get("t5_ai_dir", "")
            _cap_t5_included = st.session_state.get("t5_ai_included", [])

            if not _cap_t5_ready or not _cap_t5_included:
                st.info(
                    "No Lifetime Compare data found in this session.  \n"
                    "Go to **Tab 5 → Lifetime Compare**, load a folder and include at least "
                    "one file, then return here."
                )
            else:
                st.success(
                    f"✅ Loaded from Tab 5 — `{_cap_t5_dir}`  "
                    f"({len(_cap_t5_included)} file(s) included)"
                )
                # Parse with Tab 5's simple parser for τ / η values
                import re as _re5
                _PAT5_TAU = _re5.compile(
                    r"τ1\s*\((?:Blue|UV\+Blue)\)\s*[:：]\s*([\d\.]+)\s*ns[,，]?\s*"
                    r"τ2\s*\(Yellow\)\s*[:：]\s*([\d\.]+)\s*(?:us|µs|μs)",
                    _re5.IGNORECASE,
                )
                _PAT5_BB = _re5.compile(
                    r"Radiative efficiency\s+(?:BB|UVB_BB)\s*[:：]\s*([\d\.]+)\s*%",
                    _re5.IGNORECASE,
                )
                _PAT5_YB = _re5.compile(
                    r"Radiative efficiency\s+YB\s*[:：]\s*([\d\.]+)\s*%",
                    _re5.IGNORECASE,
                )
                _cap_lc_records = []
                for _item in _cap_t5_included:
                    _fpath = os.path.join(_cap_t5_dir, _item["file"])
                    try:
                        _txt = open(_fpath, encoding="utf-8", errors="replace").read()
                        _mt  = _PAT5_TAU.search(_txt)
                        _mb  = _PAT5_BB.search(_txt)
                        _my  = _PAT5_YB.search(_txt)
                        _cap_lc_records.append({
                            "label":   _item["label"],
                            "tau1":    float(_mt.group(1)) if _mt else None,
                            "tau2":    float(_mt.group(2)) if _mt else None,
                            "eff_bb":  float(_mb.group(1)) if _mb else None,
                            "eff_yb":  float(_my.group(1)) if _my else None,
                        })
                    except Exception:
                        pass
                _cap_source_label = _cap_t5_dir
                _cap_source_ok    = True
                _cap_payload      = _t8_cap_payload_lifetime(_cap_lc_records, _cap_t5_dir)

        elif _cap_fig_type == "CIE Diagram":
            _cap_t6_ready  = st.session_state.get("t6_ai_ready", False)
            _cap_t6_points = st.session_state.get("t6_ai_points", [])

            if not _cap_t6_ready or not _cap_t6_points:
                st.info(
                    "No CIE data found in this session.  \n"
                    "Go to **Tab 6 → CIE Diagram**, load a folder of corrected spectra, "
                    "then return here."
                )
            else:
                st.success(f"✅ Loaded from Tab 6 — {len(_cap_t6_points)} sample(s)")
                _cap_source_label = f"{len(_cap_t6_points)} CIE points (Tab 6)"
                _cap_source_ok    = True
                _cap_payload      = _t8_cap_payload_cie(_cap_t6_points)

        elif _cap_fig_type == "Mapping Heatmap":
            _cap_t7_ready = st.session_state.get("t7_ai_ready", False)
            _cap_t7_x     = st.session_state.get("t7_ai_x",       None)
            _cap_t7_y     = st.session_state.get("t7_ai_y",       None)
            _cap_t7_data  = st.session_state.get("t7_ai_data",    None)

            if not _cap_t7_ready or _cap_t7_data is None:
                st.info(
                    "No mapping data found in this session.  \n"
                    "Go to **Tab 7 → Mapping Heatmap**, load a folder, then return here."
                )
            else:
                _cap_map_dir = st.session_state.get("mapping_dir", "")
                st.success(
                    f"✅ Loaded from Tab 7 — `{_cap_map_dir}`  "
                    f"({len(_cap_t7_x)} × {len(_cap_t7_y)} points)"
                )
                _cap_source_label = _cap_map_dir
                _cap_source_ok    = True
                _cap_payload = _t8_cap_payload_mapping(
                    _cap_t7_x, _cap_t7_y, _cap_t7_data,
                    do_detect=st.session_state.get("t7_ai_do_detect", False),
                    det_pct=float(st.session_state.get("t7_ai_det_pct", 25)),
                    det_abs=st.session_state.get("t7_ai_det_abs", None),
                )

        # ── Show computed payload preview ─────────────────────────────────────
        if _cap_source_ok and _cap_payload:
            with st.expander("📊 Computed figure summary (Python)", expanded=False):
                st.text(_cap_payload)

        # ── 3 · Researcher notes ──────────────────────────────────────────────
        if _cap_source_ok:
            st.markdown("#### 3 · Researcher notes  *(optional)*")
            _cap_notes = st.text_area(
                "Additional context for the AI writer "
                "(e.g. journal target, figure number, specific features to highlight)",
                value=st.session_state.get("t8_cap_notes", ""),
                height=80,
                key="t8_cap_notes_area",
            )
            st.session_state["t8_cap_notes"] = _cap_notes

            # ── 4 · Model ─────────────────────────────────────────────────────
            st.markdown("#### 4 · Model")
            _cap_key_s = "✅ API key configured" if _T8_GROQ_KEY else "❌ GROQ_API_KEY not found"
            _cap_pkg_s = "✅ Package ready"      if _T8_GROQ_PKG else "❌ langchain-openai not installed"
            st.caption(f"{_cap_key_s}   {_cap_pkg_s}")
            _cap_model = st.selectbox(
                "Groq model",
                options=_T8_GROQ_MODELS,
                index=(
                    _T8_GROQ_MODELS.index(
                        st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    )
                    if st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                    in _T8_GROQ_MODELS else 0
                ),
                key="t8_cap_groq_model_select",
            )
            st.session_state["t8_groq_model"] = _cap_model

            # ── 5 · Generate ──────────────────────────────────────────────────
            st.markdown("#### 5 · Generate")
            if st.button(
                "▶ Generate Caption & Results",
                type="primary",
                key="t8_cap_generate_btn",
                disabled=(not _T8_GROQ_KEY or not _T8_GROQ_PKG),
            ):
                _cap_user_prompt = (
                    f"Please generate a figure caption and results paragraph for the following "
                    f"GaN PL {_cap_fig_type} figure, using the required section tags.\n\n"
                )
                if st.session_state.get("t8_cap_notes", "").strip():
                    _cap_user_prompt += (
                        f"Researcher notes: {st.session_state['t8_cap_notes'].strip()}\n\n"
                    )
                _cap_user_prompt += _cap_payload

                _cap_chosen = st.session_state.get("t8_groq_model", _T8_GROQ_MODELS[0])
                with st.spinner(
                    f"Calling Groq ({_cap_chosen}) — this usually takes 5–20 seconds…"
                ):
                    _cap_raw = _t8_call_groq(
                        _T8_CAP_SYSTEM_PROMPT, _cap_user_prompt, _cap_chosen
                    )

                st.session_state["t8_cap_raw_output"] = _cap_raw
                if _cap_raw.startswith("[ERROR]"):
                    st.error(_cap_raw)
                    st.session_state["t8_cap_sections"] = {}
                else:
                    _cap_secs    = _t8_parse_cap_tags(_cap_raw)
                    _cap_missing = [t for t in _T8_CAP_TAGS
                                    if _cap_secs[t] == "(This section was not returned by the model.)"]

                    # Rescue call for missing tags
                    if _cap_missing:
                        _cap_rescue_prompt = (
                            "Your previous response was missing these required sections: "
                            + ", ".join(f"[{t}]" for t in _cap_missing)
                            + ".\n\nPlease provide ONLY those missing sections now, "
                            "using the exact tag format. Do not repeat sections already provided.\n\n"
                            + "\n".join(f"[{t}]\n(your content here)" for t in _cap_missing)
                        )
                        with st.spinner(
                            f"Some sections were missing — requesting completion "
                            f"({', '.join(_cap_missing)})…"
                        ):
                            _cap_rescue_raw = _t8_call_groq(
                                _T8_CAP_SYSTEM_PROMPT, _cap_rescue_prompt, _cap_chosen
                            )
                        if not _cap_rescue_raw.startswith("[ERROR]"):
                            _cap_rescue_secs = _t8_parse_cap_tags(_cap_rescue_raw)
                            for _ct in _cap_missing:
                                if _cap_rescue_secs[_ct] != "(This section was not returned by the model.)":
                                    _cap_secs[_ct] = _cap_rescue_secs[_ct]
                            st.session_state["t8_cap_raw_output"] = (
                                _cap_raw + "\n\n--- rescue call ---\n" + _cap_rescue_raw
                            )
                        _cap_still_missing = [t for t in _T8_CAP_TAGS
                                              if _cap_secs[t] == "(This section was not returned by the model.)"]
                        if _cap_still_missing:
                            st.warning(
                                f"⚠️ Section(s) still missing after retry: "
                                f"**{', '.join(_cap_still_missing)}**. "
                                f"Try switching to `qwen/qwen3-32b` or `llama-3.3-70b-versatile`."
                            )

                    st.session_state["t8_cap_sections"]     = _cap_secs
                    st.session_state["t8_cap_edit_FIGURE_CAPTION"]    = _cap_secs["FIGURE_CAPTION"]
                    st.session_state["t8_cap_edit_RESULTS_PARAGRAPH"] = _cap_secs["RESULTS_PARAGRAPH"]
                    st.session_state["t8_cap_edit_version"] = (
                        st.session_state.get("t8_cap_edit_version", 0) + 1
                    )

            # ── 6 · Editable output ───────────────────────────────────────────
            if st.session_state.get("t8_cap_sections"):
                st.markdown("#### 6 · Draft  *(editable)*")
                st.caption(
                    "Both sections are editable. "
                    "Re-running the generator will overwrite your edits."
                )
                _cap_ev = st.session_state.get("t8_cap_edit_version", 0)

                with st.expander("🖼️ Figure Caption", expanded=True):
                    _cap_fig_edited = st.text_area(
                        label="Figure Caption",
                        value=st.session_state.get("t8_cap_edit_FIGURE_CAPTION", ""),
                        height=140,
                        key=f"t8_cap_textarea_FIGURE_CAPTION_v{_cap_ev}",
                        label_visibility="collapsed",
                    )
                    st.session_state["t8_cap_edit_FIGURE_CAPTION"] = _cap_fig_edited

                with st.expander("📄 Results & Discussion Paragraph", expanded=True):
                    _cap_res_edited = st.text_area(
                        label="Results Paragraph",
                        value=st.session_state.get("t8_cap_edit_RESULTS_PARAGRAPH", ""),
                        height=180,
                        key=f"t8_cap_textarea_RESULTS_PARAGRAPH_v{_cap_ev}",
                        label_visibility="collapsed",
                    )
                    st.session_state["t8_cap_edit_RESULTS_PARAGRAPH"] = _cap_res_edited

                # ── 7 · Export ────────────────────────────────────────────────
                st.markdown("#### 7 · Export")
                _cap_export_lines = [
                    f"GaN PL AI Caption & Results — {_cap_fig_type}",
                    "=" * 60,
                    f"Figure type: {_cap_fig_type}",
                    f"Source:      {_cap_source_label}",
                    f"Model:       {st.session_state.get('t8_groq_model', '')}",
                    f"Notes:       {st.session_state.get('t8_cap_notes', '')}",
                    "",
                    "[FIGURE_CAPTION]",
                    st.session_state.get("t8_cap_edit_FIGURE_CAPTION", ""),
                    "",
                    "[RESULTS_PARAGRAPH]",
                    st.session_state.get("t8_cap_edit_RESULTS_PARAGRAPH", ""),
                    "",
                ]
                _cap_export_text = "\n".join(_cap_export_lines)

                _cap_dl1, _cap_dl2 = st.columns(2)
                with _cap_dl1:
                    st.download_button(
                        "⬇️ Download as .txt",
                        data=_cap_export_text.encode("utf-8"),
                        file_name="caption_and_results.txt",
                        mime="text/plain",
                        key="t8_cap_dl_txt",
                    )
                with _cap_dl2:
                    _fig_type_slug = _cap_fig_type.lower().replace(" ", "_")
                    _cap_md_lines = [
                        f"# GaN PL AI Caption & Results — {_cap_fig_type}",
                        "",
                        f"**Figure type:** {_cap_fig_type}  ",
                        f"**Source:** {_cap_source_label}  ",
                        f"**Model:** {st.session_state.get('t8_groq_model', '')}  ",
                        f"**Notes:** {st.session_state.get('t8_cap_notes', '')}",
                        "",
                        "## 🖼️ Figure Caption",
                        "",
                        st.session_state.get("t8_cap_edit_FIGURE_CAPTION", ""),
                        "",
                        "## 📄 Results & Discussion Paragraph",
                        "",
                        st.session_state.get("t8_cap_edit_RESULTS_PARAGRAPH", ""),
                        "",
                    ]
                    _cap_md_text = "\n".join(_cap_md_lines)
                    st.download_button(
                        "⬇️ Download as .md",
                        data=_cap_md_text.encode("utf-8"),
                        file_name="caption_and_results.md",
                        mime="text/markdown",
                        key="t8_cap_dl_md",
                    )

                with st.expander("🔧 Raw model output (debug)", expanded=False):
                    st.text(st.session_state.get("t8_cap_raw_output", ""))
