import os
import numpy as np

TARGET_DIRS = ["faces_db", "selected_targets_db"]

def reset_embeddings():
    print("=" * 70)
    print("  RESETTING EMBEDDINGS TO INITIAL UPLOADED STATE (ORIGINAL 3 VIEWS)")
    print("=" * 70)
    
    reset_count = 0
    total_files = 0
    
    for db_dir in TARGET_DIRS:
        if not os.path.exists(db_dir):
            continue
            
        print(f"\n[INFO] Checking directory: {db_dir}...")
        files = [f for f in os.listdir(db_dir) if f.endswith(".npy")]
        total_files += len(files)
        
        for f in files:
            file_path = os.path.join(db_dir, f)
            try:
                arr = np.load(file_path)
                # If embeddings were appended beyond initial 3 views (arr.shape[0] > 3)
                if arr.ndim > 1 and arr.shape[0] > 3:
                    initial_arr = arr[:3]  # Retain original 3 initial embeddings (clean, eye-occluded, mask-occluded)
                    np.save(file_path, initial_arr)
                    reset_count += 1
                    print(f"  [RESET] '{f}' ({db_dir}): Resetted shape from {arr.shape} -> {initial_arr.shape}")
            except Exception as e:
                print(f"  [ERROR] Failed processing {file_path}: {e}")
                
    print("\n" + "=" * 70)
    print("                      SUMMARY REPORT")
    print("=" * 70)
    print(f" Total Files Inspected : {total_files}")
    print(f" Total Embeddings Reset: {reset_count}")
    print(" All target embeddings have been restored to their initial state.")
    print("=" * 70)

if __name__ == "__main__":
    reset_embeddings()
