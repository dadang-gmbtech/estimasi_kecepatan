import os
import time
import subprocess as sp
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Compatibility patch for streamlit-drawable-canvas in newer Streamlit versions
import streamlit.elements.image as st_image
import streamlit.elements.lib.image_utils as image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig

try:
    original_image_to_url = image_utils.image_to_url
    def patched_image_to_url(image, *args, **kwargs):
        new_args = list(args)
        if len(new_args) > 0:
            layout_config = new_args[0]
            if not isinstance(layout_config, LayoutConfig):
                new_args[0] = LayoutConfig(width=layout_config)
        elif "layout_config" in kwargs:
            layout_config = kwargs["layout_config"]
            if not isinstance(layout_config, LayoutConfig):
                kwargs["layout_config"] = LayoutConfig(width=layout_config)
        return original_image_to_url(image, *new_args, **kwargs)
    st_image.image_to_url = patched_image_to_url
except AttributeError:
    pass

from streamlit_drawable_canvas import st_canvas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Speed Estimator Interactive", page_icon="🚗", layout="wide")
st.title("🚗 Interactive Vehicle Speed Estimator (Dynamic Depth-Scaling)")
st.markdown("Aplikasi ini menggunakan profil pergerakan kendaraan referensi untuk mengatasi masalah distorsi perspektif. Kecepatan dihitung menggunakan *Dynamic Depth-Scaling* sehingga hasilnya konsisten dari jauh maupun dekat kamera.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "temp_input.mp4")
OUTPUT_RAW = os.path.join(BASE_DIR, "output_interactive.mp4")
OUTPUT_WEB = os.path.join(BASE_DIR, "output_interactive_web.mp4")

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

if "video_uploaded" not in st.session_state:
    st.session_state["video_uploaded"] = False

uploaded_file = st.file_uploader("📂 Upload Video Kendaraan (.mp4)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with open(INPUT_VIDEO, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["video_uploaded"] = True

if st.session_state["video_uploaded"] and os.path.exists(INPUT_VIDEO):
    cap = cv2.VideoCapture(INPUT_VIDEO)
    ret, first_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ret:
        st.error("Gagal membaca frame dari video.")
        st.stop()

    st.markdown("### 1. Kalibrasi Referensi Dinamis")
    st.info("Pilih satu kendaraan yang terlihat cukup lama bergerak dari jauh ke dekat. Gambar kotak padanya, lalu masukkan kecepatan aktual kendaraan tersebut.")
    
    canvas_w = min(1000, w)
    scale_factor = canvas_w / float(w)
    canvas_h = int(h * scale_factor)

    rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    col1, col2 = st.columns([2, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_img,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key="canvas",
            display_toolbar=True
        )

    user_bbox = None
    with col2:
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            if not objects.empty and len(objects) > 0:
                last_rect = objects.iloc[-1]
                left = last_rect["left"] / scale_factor
                top = last_rect["top"] / scale_factor
                width_rect = last_rect["width"] / scale_factor
                height_rect = last_rect["height"] / scale_factor
                
                x1 = max(0, int(left))
                y1 = max(0, int(top))
                x2 = min(w, int(left + width_rect))
                y2 = min(h, int(top + height_rect))
                user_bbox = [x1, y1, x2, y2]
                
                st.success("✅ Kendaraan referensi telah diblok.")
                
                crop = first_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Kendaraan Referensi")

                real_speed = st.number_input("Kecepatan Aktual (km/jam):", min_value=1.0, value=60.0, step=1.0)
                
                if st.button("▶ Mulai Proses", use_container_width=True):
                    model = YOLO("yolo11s.pt")
                    
                    # Tahap 1: Ekstraksi Pelacakan Penuh (Full Tracking)
                    st.info("Tahap 1/2: Melakukan tracking pada seluruh video...")
                    progress_bar = st.progress(0)
                    
                    cap_proc = cv2.VideoCapture(INPUT_VIDEO)
                    track_history = defaultdict(list)
                    detections_by_frame = {}
                    
                    frame_idx = 0
                    while True:
                        ret, frame = cap_proc.read()
                        if not ret: break
                        
                        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 3, 5, 7], verbose=False)
                        frame_dets = []
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            ids = results[0].boxes.id.cpu().numpy().astype(int)
                            for box, tid in zip(boxes, ids):
                                cx = (box[0] + box[2]) / 2.0
                                cy = (box[1] + box[3]) / 2.0
                                frame_dets.append((tid, box, cx, cy))
                                track_history[tid].append((frame_idx, cx, cy, box))
                        
                        detections_by_frame[frame_idx] = frame_dets
                        frame_idx += 1
                        if frame_idx % 10 == 0:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                            
                    cap_proc.release()
                    progress_bar.empty()
                    
                    # Mencari Target ID
                    target_id = None
                    max_iou = 0.0
                    for f_idx in range(min(15, len(detections_by_frame))):
                        for tid, box, cx, cy in detections_by_frame.get(f_idx, []):
                            iou = calculate_iou(user_bbox, box)
                            if iou > max_iou and iou > 0.3:
                                max_iou = iou
                                target_id = tid
                        if target_id is not None: break

                    if target_id is None:
                        st.error("Gagal mencocokkan kendaraan yang digambar dengan hasil deteksi. Coba gambar kotak lebih pas.")
                        st.stop()
                        
                    st.success(f"Kendaraan Referensi terdeteksi sebagai ID: {target_id}")
                    
                    # Helper untuk kecepatan piksel (window=0.5 detik)
                    WINDOW = max(2, int(fps * 0.5))
                    def compute_px_speeds(hist):
                        speeds = []
                        for i in range(len(hist)):
                            f1, cx1, cy1, _ = hist[i]
                            start_idx = max(0, i - WINDOW)
                            f0, cx0, cy0, _ = hist[start_idx]
                            dt = (f1 - f0) / fps
                            if dt > 0:
                                dist = np.hypot(cx1 - cx0, cy1 - cy0)
                                speeds.append(dist / dt)
                            else:
                                speeds.append(0.0)
                        return speeds

                    # Membuat Profil Perspektif (Curve Fitting)
                    st.info("Membangun model Depth-Scaling...")
                    ref_hist = track_history[target_id]
                    ref_px_speeds = compute_px_speeds(ref_hist)
                    
                    valid_pts = [(hist[2], spd) for hist, spd in zip(ref_hist, ref_px_speeds) if spd > 1.0]
                    
                    if len(valid_pts) < 5:
                        st.error("Kendaraan referensi tidak terlihat cukup lama/jauh untuk membuat model depth-scaling. Coba pilih kendaraan lain.")
                        st.stop()
                        
                    Y_vals = np.array([pt[0] for pt in valid_pts])
                    V_vals = np.array([pt[1] for pt in valid_pts])
                    
                    # Fit polynomial degree 2 (V_px expected based on Y)
                    p_fit = np.polyfit(Y_vals, V_vals, 2)
                    
                    # Menghitung estimasi kecepatan semua kendaraan
                    all_vehicle_speeds = defaultdict(dict)
                    for tid, hist in track_history.items():
                        px_speeds = compute_px_speeds(hist)
                        
                        smooth_speeds = pd.Series(px_speeds).rolling(window=5, min_periods=1).mean().values
                        
                        for i, (f_idx, cx, cy, box) in enumerate(hist):
                            v_px = smooth_speeds[i]
                            
                            # Expected pixel speed at this Y using our fitted model
                            expected_v_px = np.polyval(p_fit, cy)
                            # Batasan aman agar tidak terjadi nilai negatif/nol akibat noise fitting
                            expected_v_px = max(expected_v_px, 1.0) 
                            
                            scale = real_speed / expected_v_px
                            speed_kmh = v_px * scale
                            all_vehicle_speeds[tid][f_idx] = speed_kmh

                    # Tahap 2: Menulis Hasil Video
                    st.info("Tahap 2/2: Membuat anotasi dan menyimpan video...")
                    cap_draw = cv2.VideoCapture(INPUT_VIDEO)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(OUTPUT_RAW, fourcc, fps, (w, h))
                    
                    progress_bar2 = st.progress(0)
                    all_records = []
                    
                    for f_idx in range(total_frames):
                        ret, frame = cap_draw.read()
                        if not ret: break
                        
                        annotated = frame.copy()
                        dets = detections_by_frame.get(f_idx, [])
                        
                        for (tid, box, cx, cy) in dets:
                            x1, y1, x2, y2 = map(int, box)
                            spd = all_vehicle_speeds[tid].get(f_idx, 0.0)
                            
                            all_records.append({"frame": f_idx, "time_s": f_idx/fps, "id": tid, "speed_kmh": spd})
                            
                            label = f"ID:{tid} {spd:.1f} km/h"
                            color = (0, 255, 0)
                            if tid == target_id:
                                color = (0, 165, 255) # Orange for reference
                            
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                        # Debug: Gambarkan kurva / skala kedalaman secara visual (opsional)
                        cv2.putText(annotated, "Dynamic Depth-Scaling Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                        writer.write(annotated)
                        if f_idx % 10 == 0:
                            progress_bar2.progress(min(f_idx / total_frames, 1.0))
                            
                    cap_draw.release()
                    writer.release()
                    progress_bar2.empty()
                    
                    st.info("🔄 Mengonversi video ke format web (H264)...")
                    try:
                        import imageio_ffmpeg
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        sp.run([ffmpeg_exe, "-y", "-i", OUTPUT_RAW, "-vcodec", "libx264", OUTPUT_WEB], check=True)
                        play_path = OUTPUT_WEB
                    except Exception as e:
                        st.warning(f"Konversi gagal: {e}")
                        play_path = OUTPUT_RAW
                    
                    st.session_state["play_path"] = play_path
                    st.session_state["all_records"] = all_records
                    st.session_state["calibrated"] = True
                    st.session_state["target_id"] = target_id
                    st.session_state["real_speed"] = real_speed

        else:
            st.warning("Silakan gambar kotak (bounding box) pada kendaraan referensi terlebih dahulu.")

    if "play_path" in st.session_state and os.path.exists(st.session_state["play_path"]):
        st.markdown("---")
        st.markdown("### 2. Hasil Estimasi")
        st.video(st.session_state["play_path"])
        
        df = pd.DataFrame(st.session_state["all_records"])
        if not df.empty:
            avg_speed = df.groupby("id")["speed_kmh"].mean().reset_index()
            
            target = st.session_state.get("target_id", None)
            real_speed = st.session_state.get("real_speed", None)
            
            if target is not None and real_speed is not None:
                st.markdown("#### Evaluasi Error Kendaraan Referensi")
                ref_df = df[df["id"] == target].copy()
                if not ref_df.empty:
                    ref_df["abs_error"] = abs(ref_df["speed_kmh"] - real_speed)
                    ref_df["sq_error"] = (ref_df["speed_kmh"] - real_speed) ** 2
                    mae = ref_df["abs_error"].mean()
                    rmse = np.sqrt(ref_df["sq_error"].mean())
                    
                    c1, c2 = st.columns(2)
                    c1.metric("MAE (Mean Absolute Error)", f"{mae:.2f} km/h")
                    c2.metric("RMSE (Root Mean Square Error)", f"{rmse:.2f} km/h")
                    
                    st.markdown("**Grafik Error Absolut (Selisih dengan Kecepatan Aktual)**")
                    st.line_chart(ref_df.set_index("time_s")["abs_error"])

            st.markdown("#### Grafik Kecepatan Waktu-ke-Waktu")
            top_ids = avg_speed.nlargest(5, "speed_kmh")["id"].tolist()
            if target is not None and target not in top_ids:
                top_ids.append(target)
                
            df_plot = df[df["id"].isin(top_ids)].pivot(index="time_s", columns="id", values="speed_kmh")
            df_plot.columns = [f"ID {c} (Ref)" if c == target else f"ID {c}" for c in df_plot.columns]
            st.line_chart(df_plot)

            st.markdown("#### Distribusi Kecepatan Rata-Rata")
            bins = range(0, int(avg_speed["speed_kmh"].max()) + 20, 10)
            hist, bin_edges = np.histogram(avg_speed["speed_kmh"], bins=bins)
            hist_df = pd.DataFrame({"Jumlah Kendaraan": hist}, index=[f"{bins[i]}-{bins[i+1]} km/h" for i in range(len(bins)-1)])
            st.bar_chart(hist_df)

            st.markdown("#### Statistik Kecepatan Rata-rata per Kendaraan")
            st.dataframe(avg_speed.head(20), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Semua Data CSV", data=csv, file_name="speed_interactive.csv", mime="text/csv")
