import os
import sys
import time
import shutil
from register_face import register_face, get_app

DEFAULT_IMAGE_FOLDER = r"d:\Masked\Real Images"

def batch_register_images(input_folder=DEFAULT_IMAGE_FOLDER, db_root="temp_face_database", emb_root="faces_db", keep_images=False):
    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    # Find all supported image files
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(valid_exts)]
    total_images = len(image_files)

    print("=" * 75)
    print("  BATCH FACE REGISTRATION (AUTOMATIC IMAGE CLEANUP -> KEEP EMBEDDINGS ONLY)")
    print("=" * 75)
    print(f"[INFO] Target Folder : {input_folder}")
    print(f"[INFO] Found Images  : {total_images} files")
    print(f"[INFO] Output Embs   : {emb_root}")
    print(f"[INFO] Keep Images   : {keep_images}")
    print("=" * 75)

    if total_images == 0:
        print("[WARN] No image files found in folder.")
        return

    # Initialize single FaceAnalysis model instance
    app = get_app()

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(input_folder, fname)
        
        # Extract ORIGINAL ID/name without altering it
        person_id = os.path.splitext(fname)[0]

        print(f"\n[{idx}/{total_images}] Processing ID: '{person_id}' ({fname})...")

        try:
            # Register face: copies image, creates eye occlusion & lower-face mask occlusion, and generates .npy
            embeddings = register_face(
                name=person_id,
                image_path=fpath,
                db_root=db_root,
                emb_root=emb_root,
                app=app
            )
            success_count += 1
            print(f"  --> SUCCESS: Registered '{person_id}' ({len(embeddings)} views) -> {emb_root}/{person_id}.npy")

            # DELETE the temporary face_database folder for this person (KEEP EMBEDDINGS ONLY)
            if not keep_images:
                person_folder = os.path.join(db_root, person_id)
                if os.path.exists(person_folder):
                    shutil.rmtree(person_folder)
                    print(f"  --> CLEANUP: Deleted temp image folder for '{person_id}' from {db_root}")

        except Exception as e:
            fail_count += 1
            print(f"  --> FAILED for '{person_id}': {e}")
            # Cleanup on failure
            if not keep_images:
                person_folder = os.path.join(db_root, person_id)
                if os.path.exists(person_folder):
                    shutil.rmtree(person_folder, ignore_errors=True)

    # Clean up empty temp db root if empty
    if not keep_images and os.path.exists(db_root):
        try:
            if not os.listdir(db_root):
                os.rmdir(db_root)
        except Exception:
            pass

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("                    BATCH REGISTRATION SUMMARY")
    print("=" * 75)
    print(f" Total Processed : {total_images}")
    print(f" Successfully Registered : {success_count}")
    print(f" Failed / Skipped       : {fail_count}")
    print(f" Total Time Taken       : {elapsed:.2f} seconds ({elapsed/total_images:.2f}s / image)")
    print(f" Final Saved Embeddings : {emb_root}/ (*.npy)")
    print("=" * 75)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_FOLDER
    folder = folder.strip('"').strip("'")
    batch_register_images(folder)
