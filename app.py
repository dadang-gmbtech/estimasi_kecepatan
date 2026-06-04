import os
import sys
import time
import subprocess as _sp
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Auto-install imageio-ffmpeg ke Python yang sedang menjalankan app ini
try:
    import imageio_ffmpeg as _ioff
except ImportError:
    _sp.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
    import imageio_ffmpeg as _ioff

# ─────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vehicle Speed Tracker", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(140deg, #0d0d1a 0%, #111827 60%, #0f172a 100%); color: #e2e8f0; }
h1 { color: #a78bfa !important; font-weight: 700 !important; }
h2 { color: #818cf8 !important; font-weight: 600 !important; }
h3 { color: #6ee7b7 !important; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.8rem !important; font-weight: 700 !important; }
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important; font-size: 1.05rem !important; font-weight: 700 !important;
    padding: 0.65rem 2rem !important; border-radius: 14px !important; border: none !important;
    box-shadow: 0 4px 20px rgba(139,92,246,0.35); transition: transform 0.15s;
}
div.stButton > button:hover { transform: translateY(-2px); }
div[data-testid="stDownloadButton"] > button {
    background: rgba(99,102,241,0.15) !important; color: #a5b4fc !important;
    border: 1px solid #6366f1 !important; border-radius: 10px !important; font-weight: 600 !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(139,92,246,0.5); border-radius: 14px; padding: 1rem;
}
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; }
.stAlert { border-radius: 12px !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Speed Detection")
st.markdown(
    "Deteksi, lacak, dan estimasi kecepatan kendaraan menggunakan **YOLO11 + ByteTrack**. "
    "Kalibrasi **4 titik perspektif** untuk mengoreksi distorsi kamera dan mendapatkan "
    "kecepatan akurat dari semua jarak."
)
st.divider()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_RAW = os.path.join(BASE_DIR, "output_raw.mp4")
OUTPUT_WEB = os.path.join(BASE_DIR, "output_web.mp4")

# ─────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂  Pilih file video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    input_path = os.path.join(BASE_DIR, "input_video.mp4")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap0    = cv2.VideoCapture(input_path)
    fps_v   = cap0.get(cv2.CAP_PROP_FPS)
    vid_w   = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret0, first_frame = cap0.read()
    cap0.release()

    if not ret0:
        st.error("Gagal membaca frame pertama video.")
        st.stop()

    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    st.success(f"✅ **{uploaded_file.name}** — {vid_w}×{vid_h} px, {fps_v:.1f} FPS")

    # ─────────────────────────────────────────────────────────
    # KALIBRASI PERSPEKTIF 4 TITIK
    # ─────────────────────────────────────────────────────────
    st.subheader("📐 Kalibrasi Perspektif (4 Titik)")
    st.markdown("""
    Tandai **4 sudut** sebuah area persegi panjang di jalan yang Anda ketahui dimensi nyatanya
    (misalnya area parkir, lebar+panjang lajur, dll).

    > **Urutan titik:** Kiri-Atas → Kanan-Atas → Kanan-Bawah → Kiri-Bawah  
    > *(Bayangkan sebuah persegi panjang datar di permukaan jalan)*
    """)

    with st.expander("🖼️  Frame Pertama (referensi koordinat)", expanded=True):
        st.image(first_rgb, caption=f"{vid_w}×{vid_h} px — lihat koordinat x,y dari gambar ini",
                 use_container_width=True)

    # Default 4 titik berupa trapesium di tengah frame
    defs = [
        (vid_w // 4,     vid_h // 3),
        (3 * vid_w // 4, vid_h // 3),
        (3 * vid_w // 4, 2 * vid_h // 3),
        (vid_w // 4,     2 * vid_h // 3),
    ]
    labels_pt = ["Kiri-Atas (TL)", "Kanan-Atas (TR)", "Kanan-Bawah (BR)", "Kiri-Bawah (BL)"]
    colors_pt  = [(0, 200, 255), (0, 255, 100), (255, 80, 80), (255, 200, 0)]

    pts_img = []
    cols4 = st.columns(4)
    for i, (lbl, (dx, dy)) in enumerate(zip(labels_pt, defs)):
        with cols4[i]:
            st.markdown(f"**{lbl}**")
            px = st.number_input(f"X{i+1}", 0, vid_w - 1, dx, 1, key=f"px{i}")
            py = st.number_input(f"Y{i+1}", 0, vid_h - 1, dy, 1, key=f"py{i}")
            pts_img.append((px, py))

    st.markdown("**Dimensi Nyata Area yang Ditandai**")
    col_rw, col_rl, col_kmh = st.columns(3)
    with col_rw:
        real_width_m  = st.number_input("Lebar nyata (meter) — jarak TL↔TR / BL↔BR",
                                        0.1, 500.0, 6.0, 0.1, key="rw")
    with col_rl:
        real_length_m = st.number_input("Panjang nyata (meter) — jarak TL↔BL / TR↔BR",
                                        0.1, 500.0, 15.0, 0.1, key="rl")
    with col_kmh:
        use_kmh = st.checkbox("Tampilkan kecepatan dalam km/h", value=True)

    unit_label = "km/h" if use_kmh else "m/s"

    # ── Hitung homografi
    # Sumber: 4 titik di gambar (perspektif)
    src_pts = np.float32(pts_img)
    # Tujuan: persegi panjang bird's-eye (piksel representasi meter)
    # Kita buat 1 meter = 50 piksel di bird's-eye space (bisa apa saja, konsisten)
    SCALE_BEV = 50   # piksel per meter di bird's-eye view
    bev_w = int(real_width_m  * SCALE_BEV)
    bev_h = int(real_length_m * SCALE_BEV)
    dst_pts = np.float32([
        [0,      0     ],   # TL
        [bev_w,  0     ],   # TR
        [bev_w,  bev_h ],   # BR
        [0,      bev_h ],   # BL
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # ── Preview kalibrasi
    preview = first_rgb.copy()
    poly_pts = np.array(pts_img, dtype=np.int32)
    cv2.polylines(preview, [poly_pts], isClosed=True, color=(255, 220, 0), thickness=3)
    for i, ((px, py), color) in enumerate(zip(pts_img, colors_pt)):
        cv2.circle(preview, (px, py), 13, color, -1)
        cv2.circle(preview, (px, py), 15, (255, 255, 255), 2)
        cv2.putText(preview, f"P{i+1}", (px + 18, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    info_txt = f"Area: {real_width_m:.1f}m x {real_length_m:.1f}m"
    cv2.rectangle(preview, (8, 8), (len(info_txt) * 14 + 12, 44), (0, 0, 0), -1)
    cv2.putText(preview, info_txt, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
    st.image(preview, caption="Preview Kalibrasi Perspektif", use_container_width=True)

    # ── Bird's-eye preview
    bev_preview = cv2.warpPerspective(first_frame, H, (bev_w, bev_h))
    bev_rgb     = cv2.cvtColor(bev_preview, cv2.COLOR_BGR2RGB)
    with st.expander("🦅  Tampilan Bird's-Eye View (hasil homografi)"):
        st.image(bev_rgb, caption="Area kalibrasi setelah koreksi perspektif", use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Tombol proses
    # ─────────────────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        run = st.button("▶  Proses", use_container_width=True)

    if run:
        with st.spinner("Memuat model YOLO..."):
            model = YOLO("yolo11n.pt")

        cap      = cv2.VideoCapture(input_path)
        fps      = cap.get(cv2.CAP_PROP_FPS)
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_RAW, fourcc, fps, (width, height))

        # Riwayat centroid dalam koordinat DUNIA NYATA (meter)
        track_history_world = defaultdict(list)   # {id: [(x_m, y_m), ...]}
        speed_smooth        = defaultdict(list)    # {id: [speed buffer untuk smoothing]}
        vehicle_ids         = set()
        frame_vehicle_cnt   = []
        frame_times         = []
        all_speed_records   = []

        # Smoothing window — rata-rata N kecepatan terakhir untuk meredam jitter
        SMOOTH_N = 8

        progress_bar = st.progress(0, text="Memproses frame...")
        status_text  = st.empty()
        frame_num    = 0
        start_time   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            t_sec = frame_num / fps

            results = model.track(
                frame, persist=True,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
                verbose=False,
            )

            annotated    = frame.copy()
            active_count = 0

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)
                active_count = len(ids)

                for box, tid in zip(boxes, ids):
                    vehicle_ids.add(tid)
                    x1, y1, x2, y2 = map(int, box)
                    cx_px = (x1 + x2) // 2
                    cy_px = (y1 + y2) // 2

                    # ── Transformasi centroid ke koordinat dunia nyata (meter)
                    pt_img  = np.float32([[[cx_px, cy_px]]])
                    pt_bev  = cv2.perspectiveTransform(pt_img, H)[0][0]
                    # Konversi dari piksel BEV ke meter
                    wx = float(pt_bev[0]) / SCALE_BEV   # meter
                    wy = float(pt_bev[1]) / SCALE_BEV   # meter

                    track_history_world[tid].append((wx, wy))
                    if len(track_history_world[tid]) > 30:
                        track_history_world[tid].pop(0)
                    hist_w = track_history_world[tid]

                    # ── Estimasi kecepatan di ruang dunia nyata
                    speed_raw = 0.0
                    if len(hist_w) > 1:
                        win = min(7, len(hist_w))
                        pw1, pw2 = hist_w[-win], hist_w[-1]
                        dist_m = float(np.linalg.norm(
                            np.array(pw2) - np.array(pw1)))
                        dt = (win - 1) / fps
                        if dt > 0:
                            speed_ms  = dist_m / dt
                            speed_raw = speed_ms * 3.6 if use_kmh else speed_ms

                    # ── Smoothing kecepatan (rolling average)
                    speed_smooth[tid].append(speed_raw)
                    if len(speed_smooth[tid]) > SMOOTH_N:
                        speed_smooth[tid].pop(0)
                    speed = float(np.mean(speed_smooth[tid]))

                    all_speed_records.append((round(t_sec, 2), int(tid), round(speed, 2)))

                    label = f"ID:{tid}  {speed:.1f} {unit_label}"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
                    cv2.circle(annotated, (cx_px, cy_px), 5, (0, 0, 255), -1)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 14),
                                  (x1 + tw + 6, y1), (20, 20, 20), -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)
                    # Trajektori piksel untuk visual
                    if len(track_history_world[tid]) > 1:
                        # Rekonstruksi jalur piksel dari world coords (approximate)
                        pass

            # Overlay info
            cv2.putText(annotated, f"Kendaraan: {len(vehicle_ids)}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 230), 3)
            # Gambar area kalibrasi
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=(255, 220, 0), thickness=2)
            cv2.putText(annotated, f"Kalib: {real_width_m:.0f}mx{real_length_m:.0f}m",
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

            writer.write(annotated)
            frame_vehicle_cnt.append(active_count)
            frame_times.append(round(t_sec, 2))

            pct = int(frame_num / total_fr * 100)
            progress_bar.progress(min(pct, 100),
                                  text=f"Memproses frame {frame_num}/{total_fr} ({pct}%)")

        cap.release()
        writer.release()
        duration = time.time() - start_time
        progress_bar.progress(100, text="✅ Pemrosesan selesai!")

        # ── Re-encode ke H.264
        status_text.info("🔄 Mengkonversi video ke H.264 untuk browser...")
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            _sp.run(
                [ffmpeg_exe, "-y", "-i", OUTPUT_RAW,
                 "-vcodec", "libx264", "-crf", "23",
                 "-preset", "fast", "-movflags", "+faststart", OUTPUT_WEB],
                check=True, capture_output=True,
            )
            play_path = OUTPUT_WEB
            status_text.empty()
        except Exception as e:
            play_path = OUTPUT_RAW
            status_text.warning(f"⚠️ Konversi gagal ({e})")

        # ── DataFrame
        df_speed = pd.DataFrame(all_speed_records,
                                columns=["waktu_det", "id_kendaraan", "kecepatan"])
        df_count = pd.DataFrame({"waktu_det": frame_times,
                                 "jumlah_aktif": frame_vehicle_cnt})

        # ─────────────────────────────────────────────────────
        # TAMPILKAN HASIL
        # ─────────────────────────────────────────────────────
        st.success("✅ Pemrosesan selesai!")
        st.divider()

        # ── Metrik
        st.subheader("📊 Ringkasan Hasil")
        avg_spd = df_speed["kecepatan"].mean() if not df_speed.empty else 0
        max_spd = df_speed["kecepatan"].max()  if not df_speed.empty else 0
        mc, ma, mm, md, mw = st.columns(5)
        mc.metric("Total Kendaraan",       len(vehicle_ids))
        ma.metric("Rata-rata Kecepatan",   f"{avg_spd:.1f} {unit_label}")
        mm.metric("Kecepatan Tertinggi",   f"{max_spd:.1f} {unit_label}")
        md.metric("Durasi Video",           f"{total_fr/fps:.1f} det")
        mw.metric("Waktu Proses",           f"{duration:.1f} det")

        with st.expander("📐 Detail Kalibrasi"):
            ca, cb, cc = st.columns(3)
            ca.metric("Area Kalibrasi",  f"{real_width_m:.1f} × {real_length_m:.1f} m")
            cb.metric("Skala BEV",       f"{SCALE_BEV} px/m")
            cc.metric("Smoothing Window", f"{SMOOTH_N} frame")

        st.divider()

        # ── Video
        st.subheader("📹 Video Hasil Deteksi")
        with open(play_path, "rb") as vf:
            st.video(vf.read(), format="video/mp4")
        with open(play_path, "rb") as vf:
            st.download_button("⬇️  Download Video Hasil", data=vf,
                               file_name="output_deteksi.mp4", mime="video/mp4",
                               use_container_width=True)

        st.divider()

        # ── Grafik
        st.subheader("📈 Grafik Analisis")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"**Kecepatan per Kendaraan ({unit_label})**")
            if not df_speed.empty:
                pivot = (
                    df_speed.groupby(["waktu_det", "id_kendaraan"])["kecepatan"]
                    .mean().reset_index()
                    .pivot(index="waktu_det", columns="id_kendaraan", values="kecepatan")
                )
                pivot.columns = [f"ID {c}" for c in pivot.columns]
                st.line_chart(pivot, use_container_width=True, height=320)
        with col_g2:
            st.markdown("**Jumlah Kendaraan Aktif**")
            if not df_count.empty:
                df_plot = df_count.set_index("waktu_det")[["jumlah_aktif"]]
                df_plot.columns = ["Kendaraan Aktif"]
                st.area_chart(df_plot, use_container_width=True, height=320)

        st.markdown(f"**Distribusi Kecepatan ({unit_label})**")
        if not df_speed.empty:
            counts, edges = np.histogram(df_speed["kecepatan"].dropna(), bins=20)
            labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(counts))]
            df_hist = pd.DataFrame({"Rentang": labels, "Frekuensi": counts}).set_index("Rentang")
            st.bar_chart(df_hist, use_container_width=True, height=280)

        st.divider()
        with st.expander("🗂️  Data Kecepatan Lengkap"):
            st.dataframe(
                df_speed.rename(columns={
                    "waktu_det": "Waktu (det)",
                    "id_kendaraan": "ID Kendaraan",
                    "kecepatan": f"Kecepatan ({unit_label})",
                }),
                use_container_width=True, height=350,
            )
            csv = df_speed.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download CSV", data=csv,
                               file_name="data_kecepatan.csv", mime="text/csv")

        if os.path.exists(input_path):
            os.remove(input_path)

else:
    st.info("⬆️  Upload video untuk memulai deteksi kendaraan.")
