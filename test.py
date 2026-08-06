import os
import sys
import time
import random
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize single model instance (same as recognize_face.py / live_recognition.py)
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

REAL_FACES_DB = "faces_db"
MASKED_TARGET_DIRS = [r"d:\Masked\Masked Images", r"d:\Masked\New_Images"]
RECOGNITION_THRESHOLD = 0.35

MASK_FLAG_MAP = {
    "SUR": "Surgical Mask",
    "N95": "N95 Mask",
    "K95": "KN95 Mask",
    "CLT": "Cloth Mask",
    "GAS": "Gas Mask",
    "NONE": "Unmasked"
}

def load_database():
    db = {}
    print(f"[INFO] Loading registered embedding gallery from '{REAL_FACES_DB}'...")
    start_t = time.time()
    if os.path.exists(REAL_FACES_DB):
        for file in os.listdir(REAL_FACES_DB):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                db[name] = np.load(os.path.join(REAL_FACES_DB, file))
    elapsed = time.time() - start_t
    print(f"[INFO] Loaded {len(db)} gallery identities into memory in {elapsed:.2f} seconds.")
    return db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def parse_image_info(fname, src_dir):
    """Parse Ground Truth ID, Mask Flag, and Mask Type from filename."""
    if "__" in fname:
        parts = fname.split("__")
        gt_id = parts[0]
        flag = parts[1] if len(parts) >= 2 else "NONE"
    else:
        gt_id = os.path.splitext(fname)[0]
        flag = "NONE"

    mask_type = MASK_FLAG_MAP.get(flag, "Unmasked" if flag == "NONE" else flag)
    return gt_id, flag, mask_type

def run_masked_benchmark_evaluation(image_dirs=MASKED_TARGET_DIRS, max_samples=None, threshold=RECOGNITION_THRESHOLD):
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    db = load_database()
    if not db:
        print("[ERROR] Database is empty.")
        return

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_query_items = []

    for d in image_dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Directory not found: {d}")
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(valid_exts):
                fpath = os.path.join(d, fname)
                gt_id, flag, mask_type = parse_image_info(fname, d)
                all_query_items.append((fpath, fname, d, gt_id, flag, mask_type))

    if not all_query_items:
        print("[ERROR] No image files found across target directories.")
        return

    # Shuffle ALL query images in a single RANDOM order
    random.seed(42)  # Reproducible random seed
    random.shuffle(all_query_items)

    if max_samples and max_samples < len(all_query_items):
        all_query_items = all_query_items[:max_samples]

    total_queries = len(all_query_items)
    print("=" * 85)
    print("   MASK-TYPE RECOGNITION BENCHMARK EVALUATION (MASKED IMAGES + NEW IMAGES)")
    print("=" * 85)
    print(f" Target Image Directories : {image_dirs}")
    print(f" Combined Query Images   : {total_queries} (Shuffled in single random order)")
    print(f" Recognition Threshold   : {threshold}")
    print(f" Gallery DB Size         : {len(db)} identities")
    print("=" * 85)

    per_image_csv_path = "masked_recognition_per_image_results.csv"
    metrics_by_mask_csv_path = "masked_recognition_metrics_by_mask_type.csv"

    per_image_rows = []
    
    # Store metrics per mask flag
    flag_stats = {}
    for flag_code, category_name in MASK_FLAG_MAP.items():
        flag_stats[flag_code] = {
            "mask_type": category_name,
            "total": 0,
            "processed": 0,
            "no_face": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "time_sum_ms": 0.0
        }

    overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0
    overall_processed = 0
    overall_no_face = 0

    start_eval_time = time.time()

    for iteration, (fpath, fname, src_dir, gt_id, flag, mask_type) in enumerate(all_query_items, 1):
        if flag not in flag_stats:
            flag_stats[flag] = {
                "mask_type": mask_type,
                "total": 0, "processed": 0, "no_face": 0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0, "time_sum_ms": 0.0
            }

        flag_stats[flag]["total"] += 1
        query_t0 = time.perf_counter()

        img = cv2.imread(fpath)
        if img is None:
            print(f"[{iteration}/{total_queries}] Could not read image: {fname}")
            continue

        faces = app.get(img)
        query_elapsed_ms = (time.perf_counter() - query_t0) * 1000.0
        flag_stats[flag]["time_sum_ms"] += query_elapsed_ms

        if len(faces) == 0:
            overall_no_face += 1
            flag_stats[flag]["no_face"] += 1
            status = "FN" if gt_id in db else "TN"
            if status == "FN":
                overall_fn += 1
                flag_stats[flag]["fn"] += 1
            else:
                overall_tn += 1
                flag_stats[flag]["tn"] += 1

            per_image_rows.append({
                "Iteration": iteration,
                "Source_Folder": os.path.basename(src_dir),
                "Image_File": fname,
                "GroundTruth_ID": gt_id,
                "Mask_Flag": flag,
                "Mask_Type": mask_type,
                "Detected_ID": "NO_FACE_DETECTED",
                "Similarity_Score": 0.0,
                "Threshold": threshold,
                "Status": status,
                "Result_Description": "No Face Detected by SCRFD"
            })
            continue

        # Live embedding extraction (DO NOT USE PRE-SAVED .npy FOR TESTING)
        query_emb = faces[0].embedding
        overall_processed += 1
        flag_stats[flag]["processed"] += 1

        best_match = None
        best_score = 0.0

        # Cosine Max-Pooling Search against faces_db gallery
        for name, db_emb in db.items():
            if db_emb.ndim == 1:
                score = cosine_similarity(query_emb, db_emb)
            else:
                scores = [cosine_similarity(query_emb, view) for view in db_emb]
                score = max(scores) if scores else 0.0

            if score > best_score:
                best_score = score
                best_match = name

        # Evaluate match against threshold
        is_known_in_db = gt_id in db
        recognized = best_score >= threshold

        if recognized:
            if best_match == gt_id:
                status = "TP"
                result_desc = "CORRECT_MATCH"
                overall_tp += 1
                flag_stats[flag]["tp"] += 1
            else:
                status = "FP"
                result_desc = f"MISIDENTIFIED_AS_{best_match}"
                overall_fp += 1
                flag_stats[flag]["fp"] += 1
        else:
            if is_known_in_db:
                status = "FN"
                result_desc = "UNRECOGNIZED_KNOWN_IDENTITY"
                overall_fn += 1
                flag_stats[flag]["fn"] += 1
            else:
                status = "TN"
                result_desc = "CORRECTLY_REJECTED_UNKNOWN"
                overall_tn += 1
                flag_stats[flag]["tn"] += 1

        per_image_rows.append({
            "Iteration": iteration,
            "Source_Folder": os.path.basename(src_dir),
            "Image_File": fname,
            "GroundTruth_ID": gt_id,
            "Mask_Flag": flag,
            "Mask_Type": mask_type,
            "Detected_ID": best_match if recognized else "UNRECOGNIZED",
            "Similarity_Score": round(float(best_score), 4),
            "Threshold": threshold,
            "Status": status,
            "Result_Description": result_desc
        })

        if iteration % 100 == 0 or iteration == total_queries:
            print(f"  Processed {iteration}/{total_queries} queries... (TP={overall_tp}, FP={overall_fp}, FN={overall_fn}, TN={overall_tn})")

    total_eval_time = time.time() - start_eval_time

    # Write per-image log CSV
    with open(per_image_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Iteration", "Source_Folder", "Image_File", "GroundTruth_ID", "Mask_Flag", "Mask_Type",
            "Detected_ID", "Similarity_Score", "Threshold", "Status", "Result_Description"
        ])
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Compute Metrics per Mask Type
    summary_by_mask_rows = []

    def calc_metrics(tp, fp, tn, fn, total, processed, avg_ms, mask_name, flag_code):
        acc = ((tp + tn) / total) * 100.0 if total > 0 else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        tpr = rec
        fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        
        return {
            "Mask_Type": mask_name,
            "Mask_Flag": flag_code,
            "Total_Queries": total,
            "Faces_Processed": processed,
            "No_Face_Detected": total - processed,
            "Total_TP": tp,
            "Total_FP": fp,
            "Total_TN": tn,
            "Total_FN": fn,
            "Accuracy_Pct": round(acc, 2),
            "Precision_Pct": round(prec, 2),
            "Recall_Pct": round(rec, 2),
            "F1_Score": round(f1 / 100.0, 4),
            "TPR_Pct": round(tpr, 2),
            "FPR_Pct": round(fpr, 2),
            "Threshold": threshold,
            "Avg_Time_Per_Image_MS": round(avg_ms, 2)
        }

    for flag_code, stats in flag_stats.items():
        if stats["total"] == 0:
            continue
        avg_ms = (stats["time_sum_ms"] / stats["total"]) if stats["total"] > 0 else 0.0
        summary_by_mask_rows.append(calc_metrics(
            stats["tp"], stats["fp"], stats["tn"], stats["fn"],
            stats["total"], stats["processed"], avg_ms,
            stats["mask_type"], flag_code
        ))

    # Add OVERALL TOTAL row
    overall_avg_ms = (total_eval_time / total_queries) * 1000.0
    summary_by_mask_rows.append(calc_metrics(
        overall_tp, overall_fp, overall_tn, overall_fn,
        total_queries, overall_processed, overall_avg_ms,
        "OVERALL_TOTAL", "ALL"
    ))

    # Write summary by mask type CSV
    with open(metrics_by_mask_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_by_mask_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_by_mask_rows)

    print("\n" + "=" * 85)
    print("          MASK-TYPE BENCHMARK EVALUATION SUMMARY REPORT")
    print("=" * 85)
    print(f" Total Queries Evaluated : {total_queries}")
    print(f" Total Time Taken       : {total_eval_time:.2f} seconds")
    print("-" * 85)
    print(f"{'Mask Type':<18} | {'Flag':<5} | {'Queries':<7} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 85)
    for row in summary_by_mask_rows:
        print(f"{row['Mask_Type']:<18} | {row['Mask_Flag']:<5} | {row['Total_Queries']:<7} | {row['Accuracy_Pct']:>7.2f}% | {row['Precision_Pct']:>7.2f}% | {row['Recall_Pct']:>6.2f}% | {row['F1_Score']:>8.4f}")
    print("=" * 85)
    print(f" Per-Image Log CSV Saved    : {per_image_csv_path}")
    print(f" Mask-Type Summary CSV Saved: {metrics_by_mask_csv_path}")
    print("=" * 85)

if __name__ == "__main__":
    dirs = MASKED_TARGET_DIRS
    max_s = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip('"').strip("'")
        if arg.isdigit():
            max_s = int(arg)
        elif os.path.isdir(arg):
            dirs = [arg]
            if len(sys.argv) > 2 and sys.argv[2].isdigit():
                max_s = int(sys.argv[2])
    run_masked_benchmark_evaluation(image_dirs=dirs, max_samples=max_s)
