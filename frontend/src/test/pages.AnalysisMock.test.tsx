import { describe, it, expect, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import AnalysisMock from "../pages/AnalysisMock";

// The comp is static demo content, so what is worth pinning is not "does it fetch" but the two
// things a design page can silently lose: the readouts the reference specifies (the numbers a
// reviewer checks the layout against) and the promise that it stays inert — no API call, no
// session read. AppLayout renders real chrome, so panel-scoped queries keep the nav out.

beforeEach(() => localStorage.clear());

const panel = (name: string) => within(screen.getByRole("region", { name }));

describe("AnalysisMock — page header", () => {
  it("renders the title, subtitle and breadcrumb trail", () => {
    renderWithProviders(<AnalysisMock />);
    expect(screen.getByRole("heading", { name: "Squat Motion Analysis", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Get real-time feedback and improve your form")).toBeInTheDocument();

    const crumbs = within(screen.getByRole("navigation", { name: "Breadcrumb" }));
    expect(crumbs.getByRole("link", { name: "Home" })).toHaveAttribute("href", "/");
    expect(crumbs.getByRole("link", { name: "Workout" })).toHaveAttribute("href", "/app");
    expect(crumbs.getByText("Squat Analysis")).toHaveAttribute("aria-current", "page");
  });

  it("sends the primary CTA into the real studio", () => {
    renderWithProviders(<AnalysisMock />);
    expect(screen.getByRole("link", { name: /Start \/ Upload Video/ })).toHaveAttribute("href", "/app");
  });

  it("shows the exercise and device selectors", () => {
    renderWithProviders(<AnalysisMock />);
    expect(within(screen.getByRole("group", { name: "Exercise" })).getByText("Squat")).toBeInTheDocument();
    expect(within(screen.getByRole("group", { name: "Device" })).getByText("Webcam")).toBeInTheDocument();
  });
});

describe("AnalysisMock — analysis stage", () => {
  it("overlays both detected errors on the stage", () => {
    renderWithProviders(<AnalysisMock />);
    const stage = panel("Live analysis");
    expect(stage.getByText("Detected Errors")).toBeInTheDocument();
    expect(stage.getByText("Knee angle too small")).toBeInTheDocument();
    expect(stage.getByText("Knees collapse forward")).toBeInTheDocument();
    expect(stage.getByText("Back not neutral")).toBeInTheDocument();
    expect(stage.getByText("Slight forward lean")).toBeInTheDocument();
  });

  it("reports the form score as a labelled dial plus its verdict", () => {
    renderWithProviders(<AnalysisMock />);
    const stage = panel("Live analysis");
    expect(stage.getByRole("img", { name: "Form score 68 percent" })).toBeInTheDocument();
    expect(stage.getByText("68%")).toBeInTheDocument();
    expect(stage.getByText("Good")).toBeInTheDocument();
  });

  it("shows the live badge and transport readout", () => {
    renderWithProviders(<AnalysisMock />);
    const stage = panel("Live analysis");
    expect(stage.getByText("Live Analysis")).toBeInTheDocument();
    expect(stage.getByText("0:12 / 0:30")).toBeInTheDocument();
  });
});

describe("AnalysisMock — summary cards", () => {
  it("lists the previous sessions with their form scores", () => {
    renderWithProviders(<AnalysisMock />);
    // Scoped to the card: "68%" is also the stage's form-score dial.
    const sessions = within(
      screen.getByText("Your Previous Sessions").closest("section") as HTMLElement
    );
    expect(sessions.getByText("Jan 20, 2025")).toBeInTheDocument();
    expect(sessions.getByText("Jan 18, 2025")).toBeInTheDocument();
    expect(sessions.getByText("Jan 15, 2025")).toBeInTheDocument();
    expect(sessions.getByText("68%")).toBeInTheDocument();
    expect(sessions.getByText("72%")).toBeInTheDocument();
    expect(sessions.getByText("65%")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View All" })).toHaveAttribute("href", "/history");
  });

  it("renders each key metric with its value and optimal band", () => {
    renderWithProviders(<AnalysisMock />);
    // Scoped to the card: "Knee Angle" and "72°" are also drawn as the stage's angle callout.
    const metrics = within(screen.getByText("Key Metrics").closest("section") as HTMLElement);
    expect(metrics.getByText("Knee Angle")).toBeInTheDocument();
    expect(metrics.getByText("72°")).toBeInTheDocument();
    expect(metrics.getByText("Optimal 80–100°")).toBeInTheDocument();
    expect(metrics.getByText("Back Angle")).toBeInTheDocument();
    expect(metrics.getByText("8°")).toBeInTheDocument();
    expect(metrics.getByText("Depth")).toBeInTheDocument();
    expect(metrics.getByText("Hips below parallel")).toBeInTheDocument();
  });

  it("numbers the three improvement tips", () => {
    renderWithProviders(<AnalysisMock />);
    expect(screen.getByText("Tips for Improvement")).toBeInTheDocument();
    const tips = screen.getByText("Tips for Improvement").closest("section");
    expect(within(tips as HTMLElement).getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByText("Push your hips back")).toBeInTheDocument();
    expect(screen.getByText("Keep chest up")).toBeInTheDocument();
    expect(screen.getByText("Knees in line with toes")).toBeInTheDocument();
  });
});

describe("AnalysisMock — coach column", () => {
  it("renders the conversation, the grounded feedback and its sources", () => {
    renderWithProviders(<AnalysisMock />);
    const coach = panel("AI Fitness Coach");
    expect(coach.getByText("Online")).toBeInTheDocument();
    expect(coach.getByText(/I've analyzed your squat/)).toBeInTheDocument();
    expect(coach.getByText("Yes, please!")).toBeInTheDocument();
    expect(coach.getByText("Knee angle too small (72°)")).toBeInTheDocument();
    expect(coach.getByText("Related Insights")).toBeInTheDocument();
    expect(
      coach.getByText(/The Effect of Knee Position on Squat Performance/)
    ).toBeInTheDocument();
  });

  it("accepts typing in the composer but sends nothing — the comp has no conversation", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AnalysisMock />);
    const coach = panel("AI Fitness Coach");
    const box = coach.getByRole("textbox", { name: "Ask me anything about your workout" });

    await user.type(box, "how deep should I go?");
    expect(box).toHaveValue("how deep should I go?");

    // Submitting is a no-op: the draft survives and no new message joins the thread.
    await user.click(coach.getByRole("button", { name: "Send message" }));
    expect(box).toHaveValue("how deep should I go?");
    expect(coach.queryByText("how deep should I go?", { selector: "p" })).not.toBeInTheDocument();
  });
});
