# REHAB24-6 Pipeline

End-to-end flow for turning the raw REHAB24-6 dataset into a trained
repetition-level **correctness** classifier.

```mermaid
flowchart TD
    %% ---------- Raw dataset ----------
    subgraph RAW["data/REHAB24-6/ (raw dataset)"]
        SEG["Segmentation.csv<br/>(one row per repetition;<br/>frames, person_id, correctness, ...)"]
        SK3D["ExN/&lt;video_id&gt;-30fps.npy<br/>3D skeletons"]
        SK2D["ExN/&lt;video_id&gt;-c17/c18-30fps.npy<br/>2D skeletons (per camera)"]
        VID["ExN/&lt;video_id&gt;-Camera17/18-30fps.mp4<br/>RGB videos (per camera)"]
    end

    %% ---------- Manifest stage ----------
    SEG --> DS["dataset.py<br/>build_manifest_main()<br/>(expand reps × cam17/cam18,<br/>assign fixed person-based split)"]
    DS --> MAN["processed/manifest.csv<br/>(sample_id, split, paths,<br/>first/last_frame, correctness)"]
    DS --> SPLITS["processed/splits/<br/>train/val/test_keys.json<br/>train=1,2,3,4,5,7,10 · val=6 · test=8,9"]
    DS --> LBL["processed/labels/<br/>correctness.json"]

    %% ---------- Feature extraction (two branches) ----------
    MAN --> SKF["skeleton_features.py<br/>normalize (root=hips, scale by<br/>shoulder/hip span) + velocity +<br/>per-rep statistical summary"]
    SK3D --> SKF
    SK2D --> SKF
    SKF --> SKOUT["processed/skeleton_features/<br/>{split}/{sample_id}.npz"]

    MAN --> VMF["videomae_features.py<br/>sample clips → VideoMAE<br/>encoder → CLS embeddings<br/>(max-pool over clips, GPU)"]
    VID --> VMF
    VMF --> VMOUT["processed/videomae_features/<br/>{split}/{sample_id}.npz"]

    %% ---------- Optional fusion ----------
    SKOUT --> FUSE["fuse_features.py<br/>concat skeleton ⊕ videomae<br/>video_feature vectors"]
    VMOUT --> FUSE
    FUSE --> FUSEOUT["processed/fused_features/<br/>{split}/{sample_id}.npz"]

    %% ---------- Training ----------
    SKOUT -. "--feature-dir" .-> TRAIN
    VMOUT -. "--feature-dir" .-> TRAIN
    FUSEOUT -. "--feature-dir" .-> TRAIN
    SPLITS --> TRAIN
    LBL --> TRAIN
    MAN --> TRAIN
    TRAIN["train_correctness_classifier.py<br/>VideoFeatureClassifier MLP +<br/>pos_weight BCE, threshold search<br/>on val, early stopping"]
    TRAIN --> CKPT["processed/correctness_classifier.pt"]
    TRAIN --> PRED["processed/correctness_predictions.csv"]
    TRAIN --> METR["processed/correctness_metrics.json<br/>(overall + per-exercise)"]

    %% ---------- Colab export ----------
    MAN --> EXP["export_colab_package.py<br/>bundle manifest + splits + labels<br/>(optionally skeleton_features)"]
    SPLITS --> EXP
    LBL --> EXP
    EXP --> PKG["processed/colab_package/<br/>(+ README with run commands)"]

    classDef raw fill:#e8f0fe,stroke:#4285f4;
    classDef proc fill:#e6f4ea,stroke:#34a853;
    classDef code fill:#fef7e0,stroke:#fbbc04;
    class SEG,SK3D,SK2D,VID raw;
    class MAN,SPLITS,LBL,SKOUT,VMOUT,FUSEOUT,CKPT,PRED,METR,PKG proc;
    class DS,SKF,VMF,FUSE,TRAIN,EXP code;
```

## Stage summary

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Manifest | `dataset.py` | `Segmentation.csv` | `manifest.csv`, `splits/*_keys.json`, `labels/correctness.json` |
| 2a. Skeleton features | `skeleton_features.py` | 2D/3D skeleton `.npy` | `skeleton_features/{split}/*.npz` |
| 2b. VideoMAE features | `videomae_features.py` | Camera `.mp4` | `videomae_features/{split}/*.npz` |
| 3. Fusion (optional) | `fuse_features.py` | skeleton + videomae `.npz` | `fused_features/{split}/*.npz` |
| 4. Train classifier | `train_correctness_classifier.py` | any feature dir + splits + labels | `*.pt`, predictions CSV, metrics JSON |
| 5. Colab export | `export_colab_package.py` | manifest/splits/labels | `colab_package/` |

**Sample granularity:** one sample per repetition × camera (`cam17`, `cam18`).
**Split policy:** fixed, person-disjoint — train `{1,2,3,4,5,7,10}`, val `{6}`, test `{8,9}`.
**Task:** binary correctness classification per repetition.
