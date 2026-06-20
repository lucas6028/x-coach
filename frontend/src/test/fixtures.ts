import type { Analysis, Detection, Retrieval } from "../api";

export const mockDetection: Detection = {
  fault_id: "knees_inward_1",
  fault_name: "knees_inward",
  kg_query: "knee valgus",
  retrieval_mode: "kg",
  severity: 0.8,
  confidence: 0.9,
  observability: "visible",
  start_time: 1.0,
  end_time: 2.5,
  start_frame: 30,
  end_frame: 75,
  peak_frame: 50,
  phase: "descent",
  evidence: { valgus_angle: 0.35 },
};

export const mockRetrieval: Retrieval = {
  fault_id: "knees_inward_1",
  fault_name: "knees_inward",
  query_text: "knee valgus",
  retrieval_mode: "kg",
  context: {
    matched_nodes: ["hip_abductors"],
    results: [
      {
        summary: {
          causes: [{ node_id: "Weak hip abductors", label: "Weak hip abductors" }],
          risks: [{ node_id: "ACL strain", label: "ACL strain" }],
          corrections: [{ node_id: "Drive knees out", label: "Drive knees out" }],
        },
      },
    ],
    subgraph: {
      nodes: [{ node_id: "knees_inward", label: "Knee Valgus" }],
      edges: [],
    },
  },
};

export const mockAnalysis: Analysis = {
  video_id: "vid_001",
  source: "library",
  metadata: { fps: 30, width: 1280, height: 720, total_frames: 300 },
  view: { view_type: "side", view_confidence: 0.95 },
  quality: {
    lower_body_visibility_mean: 0.88,
    valid_frame_ratio: 0.92,
    valid_frames: 276,
    total_frames: 300,
  },
  detections: [mockDetection],
  retrievals: [mockRetrieval],
  pose: {
    fps: 30,
    width: 1280,
    height: 720,
    frames: [{ i: 0, lm: Array.from({ length: 33 }, () => [0.5, 0.5, 1.0] as [number, number, number]) }],
  },
};

export const mockCleanAnalysis: Analysis = {
  ...mockAnalysis,
  video_id: "vid_clean",
  detections: [],
  retrievals: [],
};
