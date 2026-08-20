import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import DemoIntro from "../components/DemoIntro";
import { renderWithProviders } from "./renderWithProviders";

// Task 11 gave the movement selector its own dedicated suite (App.movement.test.tsx, exercised
// through App so the URL-driven wiring is covered end to end). These props are fixed to an
// always-available, already-loaded Squat so the pre-existing assertions below — none of which are
// about movement selection — keep observing the same dropzone/loader behavior as before.
//
// The <select> itself now lives in the studio's page header (StudioTitleBar), not here: the
// reference design moved every analysis control up into the header, and two selectors for one
// setting is how they drift apart. So DemoIntro no longer takes `movements`/`onMovementChange`,
// and the tier it forwards to CaptureStudio is supplied from above too.
const movementProps = {
  movement: "Squat",
  movementError: "",
  movementsLoaded: true,
  tier: "heavy" as const,
};

describe("DemoIntro", () => {
  it("shows the heading", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Squat in about 20 seconds."
    );
  });

  // Finding 2 of the 2026-07-25 review: this heading sits directly above the movement selector and
  // stayed hardcoded to "squat" regardless of what was selected. Pin that it now names the actual
  // selection, and that the dropzone below it (see UploadDropzone.test.tsx) agrees.
  it("names the selected movement in the heading, not a hardcoded squat", () => {
    renderWithProviders(
      <DemoIntro
        onBlob={vi.fn()}
        onError={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movement="Push-up"
      />
    );
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Push-up in about 20 seconds."
    );
    expect(screen.getByText(/Drop a Push-up video/i)).toBeInTheDocument();
  });

  it("shows the sub-heading description", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByText(/Upload or record a clip/)).toBeInTheDocument();
  });

  // The sample library is gone: the studio's empty state is now the capture panel alone. Pinned
  // because the button, its "or" divider and the copy promising a labeled sample were three
  // separate places that had to agree, and a leftover of any one of them offers a route into a
  // picker that no longer exists.
  it("offers no sample-clip route", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.queryByRole("button", { name: /sample/i })).toBeNull();
    expect(screen.queryByText(/labeled sample/i)).toBeNull();
  });

  it("swaps the dropzone for the Lumen scan loader while loading", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={true} statusMsg="Extracting pose…" error="" {...movementProps} />
    );
    // The Lumen waiting state takes over the upload target, carrying the status message…
    expect(screen.getByRole("status")).toHaveAttribute("aria-label", "Extracting pose…");
    // …and the idle dropzone prompt is gone.
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });

  it("shows the Lumen scan loader while the movement catalog is still loading", () => {
    renderWithProviders(
      <DemoIntro
        onBlob={vi.fn()}
        onError={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movementsLoaded={false}
      />
    );
    // Same "we don't know yet" guard as the loading case: the dropzone must not appear before
    // the catalog fetch settles, so a slow network can't let an unconfirmed movement upload.
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });

  it("shows the error panel when error is non-empty", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="Video too short" {...movementProps} />
    );
    expect(screen.getByText("That clip did not go through")).toBeInTheDocument();
    expect(screen.getByText("Video too short")).toBeInTheDocument();
  });

  it("does not show the error panel when error is empty", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.queryByText("That clip did not go through")).not.toBeInTheDocument();
  });

  it("renders the three 'what comes back' step cards", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByText("Skeleton and faults")).toBeInTheDocument();
    expect(screen.getByText("Grounded feedback")).toBeInTheDocument();
    expect(screen.getByText("Knowledge graph")).toBeInTheDocument();
  });

  // The selector moved to the studio's page header (see the note on `movementProps`); this pins
  // that it did NOT get left behind here too, which would give the app two of them and break
  // App.movement.test.tsx's single-select lookup.
  it("no longer renders a movement select — that lives in the page header", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.queryByLabelText(/movement/i)).toBeNull();
  });

  it("shows the movementError panel instead of the dropzone, and hides no other content", () => {
    renderWithProviders(
      <DemoIntro
        onBlob={vi.fn()}
        onError={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movement="Lunge"
        movementError='"Lunge" cannot be analysed yet. Pick one of the available movements.'
      />
    );
    expect(
      screen.getByText('"Lunge" cannot be analysed yet. Pick one of the available movements.')
    ).toBeInTheDocument();
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });
});
