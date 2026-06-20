Below is a **dataset survey summary** for the datasets you listed, using the same 6 aspects:

1. What is it for?
2. Data type
3. Data description
4. Annotation
5. Exercises / actions
6. How can you use it?

---

## 1. Lymann/EgoExo-Fitness

| Aspect               | Summary                                                                                                                                                                                                                                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | For **full-body fitness action understanding**, especially learning from both **egocentric / first-person** and **exocentric / third-person** videos. It supports tasks like action recognition, action localization, cross-view sequence verification, skill determination, and execution verification. ([arXiv][1]) |
| **Data type**        | Synchronized first-person and third-person fitness videos. Hugging Face release provides **preprocessed 30 FPS video frames** and **frame-wise CLIP-B features**; raw videos are not directly available through the Hugging Face link according to the dataset page. ([Hugging Face][2])                              |
| **Data description** | 32 hours of data, 1,276 cross-view action sequence videos, over 6,000 single fitness actions, 40 adult participants, and 86 action sequences. Each sequence contains 3–6 continuous fitness actions. ([arXiv][1])                                                                                                     |
| **Annotation**       | Local annotation files are split by task: `meta_records.json` stores record IDs, views, frame paths, sequence ranges, and counts; `action_level_annotations.json` stores coarse action boundaries per record with `num_actions` and `action_info = [action_id, start_frame, end_frame]`; `subaction_level_annotations_ant13_style_v1.json` stores ActivityNet-style sub-step boundaries with `classes`, `database`, `label`, `segment_time`, `segment`, `fps`, and per-record metadata; `interpretable_action_judgement.json` stores fine-grained action judgments with `key_point_verification`, `action_quality_score`, `comment`, `action_name`, `action_guidance`, `annotator`, plus `st_ed_frame` and `frame_root`. ([arXiv][1]) |
| **Exercises**        | 12 fitness actions: kneeling push-ups, push-ups, kneeling torso twist, knee raise and abdominal muscles contract, shoulder bridge, sit-ups, leg reverse lunge, leg lunge with knee lift, sumo squat, jumping jacks, high knee, clap jacks. ([arXiv][1])                                                               |
| **How to use it**    | Best for computer vision / AI fitness coach research: action classification, temporal action localization, cross-view learning, action quality assessment, and “did the user follow the technical keypoints?” verification.                                                                                           |

**Good for your project if:** you want to study **fitness movement understanding from video**, especially if your topic involves AI coach, form correction, or comparing first-person and third-person views.

**Local annotation counts from `data/EgoExo-Fitness/raw_annotations`:**

| Exercise | Action boundary instances | Substep boundary instances | Judgement entries | Expert judgement annotations |
| --- | ---: | ---: | ---: | ---: |
| Kneeling pushing-ups | 97 | 480 | 72 | 127 |
| Push-ups | 60 | 339 | 62 | 100 |
| Kneeling Torso Twist | 126 | 594 | 0 | 0 |
| Knee Raise And Abdominal Muscles Contract | 89 | 447 | 0 | 0 |
| Shoulder Bridge | 77 | 441 | 77 | 130 |
| Sit-ups | 106 | 491 | 82 | 142 |
| Leg Reverse Lunge | 67 | 375 | 67 | 107 |
| Leg Lunge With Knee Lift | 81 | 338 | 64 | 110 |
| Sumo Squat | 55 | 328 | 55 | 93 |
| Jumping Jacks | 142 | 586 | 121 | 195 |
| High Knee | 98 | 383 | 68 | 120 |
| Clap Jacks | 88 | 495 | 74 | 112 |
| **Total** | **1,086** | **5,297** | **913** | **1,525** |

`Action boundary instances` are single-action temporal segments from `action_level_annotations.json`. `Substep boundary instances` are subaction/substep temporal segments from `subaction_level_annotations_ant13_style_v1.json`. `Judgement entries` are single-action records in `interpretable_action_judgement.json`; `expert judgement annotations` counts the number of human annotation objects inside those records.

---

## 2. UI-PRMD

| Aspect               | Summary                                                                                                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | A physical rehabilitation movement dataset for analyzing and assessing rehab exercise performance. It is commonly used for automatic exercise assessment and movement-quality evaluation. ([PMC][3])                                                 |
| **Data type**        | Skeleton / motion-capture data from **Vicon optical tracker** and **Kinect camera**. It includes full-body skeletal joint displacement data. ([GitHub][4])                                                                                           |
| **Data description** | 10 healthy subjects perform 10 common rehabilitation exercises. Each person performs both correct and incorrect / non-optimal repetitions. Each person performed 10 correct and 10 incorrect repetitions for each exercise. ([bura.brunel.ac.uk][5]) |
| **Annotation**       | Exercise class labels and repetition-level correctness labels: correct vs incorrect / non-optimal. Before any paper-specific cleaning, each exercise has 200 labeled repetitions: 10 subjects x 10 correct repetitions + 10 subjects x 10 incorrect repetitions. Across all 10 exercises, this gives 2,000 labeled repetitions. Some related work uses Vicon skeletal angles, such as 117-dimensional skeletal angles for deep squat quality assessment. ([GitHub][4]) |
| **Exercises**        | 10 exercises: deep squat, hurdle step, inline lunge, side lunge, sit to stand, active straight leg raise, shoulder abduction, shoulder extension, shoulder internal-external rotation, shoulder scaption. ([arXiv][6])                               |
| **How to use it**    | Good for skeleton-based machine learning: exercise classification, correct/incorrect classification, pose-based rehabilitation feedback, and comparing Kinect vs Vicon data.                                                                         |

**Good for your project if:** you want a **cleaner and smaller rehab dataset** with clear correct/incorrect labels.

---

## 3. REHAB24-6

| Aspect               | Summary                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | Designed for **human pose estimation evaluation** and **rehabilitation exercise feedback systems**. It focuses on exercise mistakes, different views, body heights, lighting conditions, and correctness-related tasks. ([Zenodo][7]) |
| **Data type**        | Multi-modal data: RGB videos from two cameras, 2D/3D motion-capture marker positions, and 2D/3D skeleton joint positions. ([Zenodo][7])                                                                                               |
| **Data description** | 65 recordings, 184,825 frames at 30 FPS, 10 subjects, 6 rehabilitation exercises, captured with 2 RGB cameras and 16 motion-capture cameras. ([Zenodo][7])                                                                            |
| **Annotation**       | 1,072 repetition annotations, including start/end frame segmentation, binary correctness label, exercise direction, and lighting-condition label. ([Zenodo][7])                                                                       |
| **Exercises**        | 6 exercises: arm abduction, arm VW, table push-ups, leg abduction, leg lunge, squats. ([Zenodo][8])                                                                                                                                   |
| **How to use it**    | Good for pose-estimation benchmarking, repetition segmentation, correct/incorrect exercise classification, and building an automatic rehab feedback system.                                                                           |

**Good for your project if:** you need **RGB video + skeleton + correctness labels**, not only skeleton data.

**Local annotation counts from `data/REHAB24-6/Segmentation.csv`:**

| Exercise ID | Exercise | Repetition annotations | Correct | Incorrect | Video records |
| --- | --- | ---: | ---: | ---: | ---: |
| Ex1 | arm abduction | 178 | 90 | 88 | 13 |
| Ex2 | arm VW | 208 | 94 | 114 | 12 |
| Ex3 | table push-ups | 107 | 52 | 55 | 10 |
| Ex4 | leg abduction | 210 | 120 | 90 | 12 |
| Ex5 | leg lunge | 174 | 78 | 96 | 9 |
| Ex6 | squats | 195 | 134 | 61 | 9 |
| **Total** |  | **1,072** | **568** | **504** | **65** |

Each row in `Segmentation.csv` is one repetition annotation with `video_id`, `repetition_number`, `exercise_id`, `person_id`, `first_frame`, `last_frame`, camera/view metadata, subtype metadata, and binary `correctness`.

---

## 4. IntelliRehabDS / IRDS

| Aspect               | Summary                                                                                                                                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What it is for**   | A physical rehabilitation movement dataset intended for automatic rehabilitation feedback, gesture classification, and movement correctness assessment. ([Zenodo][9])                                                                                                          |
| **Data type**        | 3D skeleton joint positions captured by Microsoft Kinect One at 30 FPS. It also provides depth-map images, simplified CSV format, and raw data with timestamp, tracking state, and 2D projections. ([bura.brunel.ac.uk][5])                                                    |
| **Data description** | 29 subjects: 15 real patients and 14 healthy controls. The dataset contains 2,589 files; most analysis uses 2,577 files with correctness labels 1 or 2. ([bura.brunel.ac.uk][5])                                                                                               |
| **Annotation**       | Gesture type, subject information, position label, and correctness label. Correctness values include correct, incorrect, and a third label for poorly executed/unclear movements. The paper notes correctness is mainly binary: correct vs incorrect. ([bura.brunel.ac.uk][5]) |
| **Exercises**        | 9 rehab movements: left/right elbow flexion, left/right shoulder flexion, left/right shoulder abduction, shoulder forward elevation, left side tap, right side tap. ([bura.brunel.ac.uk][5])                                                                                   |
| **How to use it**    | Good for skeleton time-series classification, patient vs healthy-control comparison, correct/incorrect prediction, and transfer learning for rehabilitation movement assessment.                                                                                               |

**Good for your project if:** you want **real patient data**, not only healthy subjects.

---

## 5. PHYTMO

| Aspect               | Summary                                                                                                                                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | For physical therapy motion monitoring using wearable inertial sensors. It supports exercise identification, exercise evaluation, and validation of IMU-based human motion algorithms. ([Zenodo][10])                 |
| **Data type**        | Magneto-inertial sensor data: acceleration, angular velocity / turn rate, magnetic field, plus optical reference system data for accurate 3D position and orientation. Files are stored in CSV format. ([Zenodo][10]) |
| **Data description** | 30 volunteers aged 20–70. The dataset includes 6 physical therapy exercises and 3 gait variations. Volunteers performed two series with at least 8 repetitions each. ([Zenodo][10])                                   |
| **Annotation**       | Exercise identity and performance correctness: correct vs wrong execution. The data also includes optical-system reference data for validating IMU-based motion estimation. ([Nature][11])                            |
| **Exercises**        | 6 exercises: knee flex-extension, squats, hip abduction, elbow flex-extension, extension of arms over head, squeezing. 3 gait variations: natural gait, infinity-symbol gait, heel-tiptoe gait. ([ResearchGate][12])  |
| **How to use it**    | Good for wearable-sensor ML, IMU signal processing, exercise recognition, correct/wrong movement classification, gait analysis, and validating sensor-fusion algorithms.                                              |

**Good for your project if:** you want to use **sensor / IMU data instead of video**.

---

## 6. UTD-MHAD

| Aspect                  | Summary                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What it is for**      | A multimodal human action recognition dataset designed for studying fusion of depth-camera data and wearable inertial-sensor data. ([jafari.tamu.edu][13])                                                                                 |
| **Data type**           | Four synchronized modalities: RGB video, depth video, skeleton joint positions, and inertial signals from Kinect and a wearable inertial sensor. ([jafari.tamu.edu][13])                                                                   |
| **Data description**    | 27 actions, 8 subjects, each subject repeats each action 4 times; after removing corrupted sequences, it has 861 action sequences. ([OpenDataLab][14])                                                                                     |
| **Annotation**          | Action class label for each sequence. It is mainly for action recognition, not specifically rehabilitation correctness assessment. ([The University of Texas at Dallas][15])                                                               |
| **Exercises / actions** | 27 general human actions, including arm swipes, hand waves, clapping, throwing, crossing arms, basketball shooting, jogging, walking, sit-to-stand, stand-to-sit, and other daily/sport actions. ([The University of Texas at Dallas][15]) |
| **How to use it**       | Good for multimodal action recognition, sensor fusion, RGB-depth-skeleton-IMU comparison, and benchmarking general HAR models.                                                                                                             |

**Good for your project if:** your topic is **general human action recognition**, not specifically rehabilitation correctness.

---

## 7. Fit3D

| Aspect               | Summary                                                                                                                                                                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | A large-scale fitness-training dataset for 3D human pose estimation, 3D human mesh reconstruction, exercise repetition segmentation, motion analysis, and automatic coaching / feedback. It was introduced with AIFit, a system for 3D human-interpretable fitness feedback. ([Fit3D][16])                                      |
| **Data type**        | Multi-view fitness exercise sequences with highly accurate ground-truth 3D skeletons, GHUM parameters, and SMPL-X human pose and shape parameters. ([Fit3D][16])                                                                                                                                                                |
| **Data description** | 611 multi-view sequences, at least 5 annotated repetitions per sequence, and 2,964,236 ground-truth 3D skeletons. The paper/site describes over 3 million images across more than 37 repeated exercises covering major muscle groups, performed by instructors and trainees. ([Fit3D][16])                                      |
| **Annotation**       | 3D motion-capture ground truth, human pose/shape parameters, repetition-level structure, and feedback-oriented labels/outputs used by the AIFit statistical coach. The dataset is meant to connect reconstructed 3D pose and motion to deviations from trainer standards and localized natural-language feedback. ([Fit3D][16]) |
| **Exercises**        | More than 37 repeated fitness exercises, covering major muscle groups. ([Fit3D][16])                                                                                                                                                                                                                                            |
| **How to use it**    | Best for fitness-tech systems that need real exercise motion rather than daily-action data: 3D pose/mesh reconstruction, repetition segmentation, form analysis, trainer-vs-trainee comparison, and interpretable feedback generation.                                                                                          |

**Good for your project if:** you want a **fitness-specific 3D motion dataset** for AI coaching, especially where exercise repetitions and form feedback matter more than generic action recognition.

**Local dataset statistics from `data/Fit3D/`:**

| Item | Count |
| --- | ---: |
| Subjects (total) | 11 |
| — train (full 3D GT) | 8: s03, s04, s05, s07, s08, s09, s10, s11 |
| — test (video + camera only, **no GT**) | 3: s02, s12, s13 |
| Synchronized cameras | 4 (50591643, 58860488, 60457274, 65906101) |
| Activity types per subject | 47 (28 named exercises + 19 warmup routines), identical set for every subject |
| Videos (total) | 1,645 |
| — train videos (8 subj × 4 cam × 47) | 1,504 |
| — test videos (3 subj × 47, single view) | 141 |
| `joints3d_25` GT sequences (train) | 376 (8 × 47) |
| **GT 3D skeletons / frames (train)** | **444,823** (25 joints each; ×4 cameras = 1,779,292 image-space projections) |
| SMPL-X parameter sequences (train) | 376 |
| GHUM (`gpp`) parameter sequences (train) | 376 |
| Sequences with repetition annotations (train) | 296 of 376 |
| Repetition-boundary frame indices (train) | 1,822 |
| **Repetitions (boundaries − 1 per sequence, ≥5/seq)** | **1,526** |

Per-subject GT 3D skeleton frames (train): s03 57,592 · s04 59,700 · s05 51,213 · s07 50,970 · s08 47,276 · s09 54,147 · s10 60,237 · s11 63,688.

Each train sequence ships `videos/{cam}/`, `camera_parameters/{cam}/` (extrinsics + intrinsics w/ & w/o distortion), `joints3d_25/`, `smplx/`, `gpp/`, and `rep_ann.json`. **Not in the local download:** per-repetition correctness/quality labels, natural-language feedback, and instructor/trainee tags — the AIFit feedback is derived by reference comparison, not shipped as labels. The test split additionally omits all 3D GT and rep annotations (videos + camera parameters only), so every 3D-GT experiment is confined to the 8 train subjects.

---

## 8. Squat Dataset / Waseda-style Squat Classification

| Aspect               | Summary                                                                                                                                                                                                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | The linked Zenodo Squat Dataset supports automated squat posture classification using pose estimation and ensemble learning. Separately, Ogata et al.'s squat-classification work focuses on fine-grained classification of good squats and common squat-form mistakes. ([Zenodo][17], [CVPRW][18])               |
| **Data type**        | Zenodo record: side-view squat images formatted to a 1:1 aspect ratio for preprocessing consistency. Ogata et al.: squat videos collected from several sources, including a detailed single-person setting, multiple people/scenes, and YouTube videos. ([Zenodo][17], [CVPRW][18])                               |
| **Data description** | Zenodo record: one `Dataset.zip` file, 824.1 MB, published November 8, 2025, under CC BY 4.0. The user-provided Waseda-style note describes short squat clips of about 10 seconds / 300 frames with roughly 3-5 squat repetitions, but that detail is not stated on the linked Zenodo page. ([Zenodo][17])          |
| **Annotation**       | Zenodo record: 3 classes: `Good` / correct posture, `Bad Back` / spinal misalignment, and `Bad Heel` / improper foot positioning. Ogata et al.: 7 labels: good squat plus inward knees, round back, warped back, upwards head, shallowness, and frontal knee. ([Zenodo][17], [CVPRW][18])                         |
| **Exercises**        | Squat only. The value is depth of posture taxonomy rather than breadth of exercise types.                                                                                                                                                                                                                        |
| **How to use it**    | Useful for squat-specific posture classification, single-exercise form correction, pose-estimation feature pipelines, and building a focused benchmark for high-risk / technically subtle strength-training movements.                                                                                            |

**Good for your project if:** you need **fine-grained squat-form error labels** rather than a broad exercise dataset.

**Note:** the provided Zenodo link and the seven-class Waseda-style squat taxonomy appear to describe related but not identical resources. Keep the source distinction clear when citing or using labels.

---

## 9. KIMORE

| Aspect               | Summary                                                                                                                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | A clinically oriented rehabilitation dataset for remote monitoring and movement assessment. It is designed to benchmark computational approaches that map motion features to clinical rehabilitation scores. ([PubMed][19], [GitHub][20])                                                                                              |
| **Data type**        | RGB videos, depth videos, and skeleton joint positions collected by an RGB-D / Kinect-style sensor. EGCN uses KIMORE as a skeleton-based rehabilitation exercise assessment dataset, including position and angle / orientation features. ([PubMed][19], [GitHub][20])                                                                 |
| **Data description** | 78 subjects: 44 healthy subjects and 34 subjects with motor dysfunctions. The exercises were selected for low-back-pain rehabilitation. Some later work describes the released data as 2,806 exercise repetitions across control experts, control non-experts, and a pain/postural-disorders group. ([PubMed][19])                      |
| **Annotation**       | For each exercise, KIMORE provides physician-defined clinically relevant features, clinical questionnaire / score information, and features validated against a stereophotogrammetric system. EGCN additionally uses manually segmented KIMORE skeleton data for model training and evaluation. ([PubMed][19], [GitHub][20])            |
| **Exercises**        | 5 physical rehabilitation exercises for low-back-pain scenarios. EGCN describes exercise-specific segmentation and separates left/right directions for some exercises. ([PubMed][19], [GitHub][20])                                                                                                                                     |
| **How to use it**    | Good for clinically grounded movement-quality assessment, patient-vs-healthy generalization, skeleton time-series modeling, clinical-score prediction, and evaluating whether AI feedback aligns with professional rehabilitation assessment rather than only simulated errors by healthy subjects.                                      |

**Good for your project if:** you need **real patient rehabilitation data with clinical scores**, not only healthy-subject demonstrations or synthetic wrong-form examples.

---

## 10. ExeCheck / ExeChecker

| Aspect               | Summary                                                                                                                                                                                                                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it is for**   | ExeCheck is the in-house dataset released with ExeChecker, a contrastive-learning framework for interpretable rehabilitation exercise assessment. The goal is not only to classify wrong execution, but to identify the joints involved in the incorrect movement and provide user-facing feedback. ([BU][21], [arXiv][22]) |
| **Data type**        | RGB-D videos (`.mkv`) plus skeletal joint information (`.json`) with joint positions and orientations. The skeleton JSON files are generated with the Microsoft Azure Kinect Body Tracking SDK. ([BU][21])                                                                                              |
| **Data description** | 10 rehabilitation exercises performed by 7 healthy subjects. Each exercise has paired correct and incorrect executions by the same subject, with 5 movement repetitions. The dataset package includes raw data, processing scripts, and a processed segmented / mirrored dataset used in the paper. ([BU][21]) |
| **Annotation**       | Paired correct/incorrect executions, repetition annotations (`RepSeg.csv`), metadata files, and exercise-specific joints of attention (JoA) describing the body joints relevant to each incorrect movement. ([BU][21])                                                                                   |
| **Exercises**        | 10 rehabilitation exercises. The project page describes incorrect-exercise procedures and corresponding joints of attention in its table; the downloadable package contains the exercise metadata used by the processing scripts. ([BU][21])                                                               |
| **How to use it**    | Good for skeleton-based rehab feedback, correct-vs-incorrect comparison, repetition segmentation, contrastive learning, localizing problematic joints, and testing whether a model can explain where a rehab movement went wrong.                                                                         |

**Good for your project if:** you want **paired correct/incorrect rehab executions with joint-level feedback targets**, especially for interpretable AI coaching rather than only binary correctness.

---

# Quick comparison

| Dataset                   | Best use                                                          | Main data type                          | Correct / incorrect label?                | Rehab-focused?      |
| ------------------------- | ----------------------------------------------------------------- | --------------------------------------- | ----------------------------------------- | ------------------- |
| **EgoExo-Fitness**        | Fitness action understanding, AI coach, cross-view video learning | Ego + exo videos, frames, CLIP features | Quality score + technical keypoint labels | No, fitness-focused |
| **UI-PRMD**               | Rehab exercise correctness classification                         | Kinect + Vicon skeleton                 | Yes                                       | Yes                 |
| **REHAB24-6**             | Pose estimation + rehab feedback                                  | RGB video + 2D/3D skeleton              | Yes                                       | Yes                 |
| **IntelliRehabDS**        | Patient rehab movement assessment                                 | Kinect 3D skeleton                      | Yes                                       | Yes                 |
| **PHYTMO**                | IMU-based rehab / gait analysis                                   | Wearable IMU + optical reference        | Yes                                       | Yes                 |
| **UTD-MHAD**              | General action recognition / sensor fusion                        | RGB + depth + skeleton + IMU            | Action label only                         | Not mainly rehab    |
| **Fit3D**                 | 3D fitness pose, mesh, repetition, and coaching feedback           | Multi-view images + 3D skeleton/mesh    | No shipped labels (3D GT + rep boundaries) | No, fitness-focused |
| **Squat Dataset/Waseda**  | Squat-specific posture classification                             | Side-view images / squat videos         | Yes, posture-error classes                | No, fitness-focused |
| **KIMORE**                | Clinical rehab movement assessment                                | RGB-D video + skeleton                  | Clinical scores/features                  | Yes                 |
| **ExeCheck/ExeChecker**   | Interpretable rehab feedback and joint-error localization          | RGB-D video + Azure Kinect skeleton     | Yes, paired correct/incorrect + JoA       | Yes                 |

[1]: https://arxiv.org/html/2406.08877v2 "EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding"
[2]: https://huggingface.co/datasets/Lymann/EgoExo-Fitness?utm_source=chatgpt.com "Lymann/EgoExo-Fitness · Datasets at Hugging Face"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5773117/?utm_source=chatgpt.com "A Data Set of Human Body Movements for Physical ... - PMC"
[4]: https://github.com/avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises "GitHub - avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises: A framework for quality assessment of exercises in physical rehabilitation based on skeletal joint displacements collected with a motion capture system. · GitHub"
[5]: https://bura.brunel.ac.uk/bitstream/2438/24189/1/FullText.pdf "IntelliRehabDS (IRDS)—A Dataset of Physical Rehabilitation Movements"
[6]: https://arxiv.org/html/2505.18412v1?utm_source=chatgpt.com "Rehabilitation Exercise Quality Assessment and Feedback ..."
[7]: https://zenodo.org/records/13305826 "REHAB24-6: A multi-modal dataset of physical rehabilitation exercises"
[8]: https://zenodo.org/records/13305826?utm_source=chatgpt.com "A multi-modal dataset of physical rehabilitation exercises"
[9]: https://zenodo.org/records/4610859?utm_source=chatgpt.com "IntelliRehabDS - A dataset of physical rehabilitation ..."
[10]: https://zenodo.org/records/6319979 "A database of physical therapy exercises with variability of execution collected by wearable sensors"
[11]: https://www.nature.com/articles/s41597-022-01387-2?utm_source=chatgpt.com "A database of physical therapy exercises with variability ..."
[12]: https://www.researchgate.net/figure/Exercises-considered-in-this-study-performed-by-one-of-the-volunteers-The-first-row_fig3_361060286?utm_source=chatgpt.com "Exercises considered in this study performed by one of the ..."
[13]: https://jafari.tamu.edu/wp-content/uploads/2019/06/ICIP2015-Chen-Final.pdf?utm_source=chatgpt.com "utd-mhad: a multimodal dataset for human action recognition"
[14]: https://opendatalab.com/OpenDataLab/UTD-MHAD_University_of_Texas_at_Dallas?utm_source=chatgpt.com "UTD-MHAD（University of Texas at Dallas）"
[15]: https://www.utdallas.edu/~kehtar/UTD-MHAD.html?utm_source=chatgpt.com "UTD Multimodal Human Action Dataset (UTD-MHAD)"
[16]: https://fit3d.imar.ro/ "Fit3D Dataset"
[17]: https://zenodo.org/records/17558630 "Squat Dataset"
[18]: https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Ogata_Temporal_Distance_Matrices_for_Squat_Classification_CVPRW_2019_paper.pdf "Temporal Distance Matrices for Squat Classification"
[19]: https://pubmed.ncbi.nlm.nih.gov/31217121/ "The KIMORE Dataset: KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation"
[20]: https://github.com/bruceyo/EGCN "EGCN"
[21]: https://www.cs.bu.edu/faculty/betke/ExeChecker/ "ExeChecker: Where Did I Go Wrong?"
[22]: https://arxiv.org/abs/2412.10573 "ExeChecker: Where Did I Go Wrong?"
