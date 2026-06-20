import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LibraryPicker from "../components/LibraryPicker";
import { renderWithProviders } from "./renderWithProviders";

function mockLibraryFetch(items = [{ video_id: "vid_001", split: "test", view_type: "side", fault_count: 1, faults: ["knees_inward"] }]) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ total: items.length, items }),
  } as Response);
}

afterEach(() => vi.restoreAllMocks());

describe("LibraryPicker", () => {
  it("shows the title", async () => {
    mockLibraryFetch([]);
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    expect(screen.getByText(/Sample Library/i)).toBeInTheDocument();
    // Flush the fetch-driven state update so it settles inside the test, not after teardown.
    await waitFor(() => expect(screen.queryByText("Loading…")).not.toBeInTheDocument());
  });

  it("shows loading state initially", async () => {
    mockLibraryFetch([]);
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
    // Let the (empty) fetch resolve so the loading→loaded transition runs inside act().
    await waitFor(() => expect(screen.queryByText("Loading…")).not.toBeInTheDocument());
  });

  it("renders library items after loading", async () => {
    mockLibraryFetch();
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("vid_001")).toBeInTheDocument());
  });

  it("renders fault tags on library items", async () => {
    mockLibraryFetch();
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("knees_inward")).toBeInTheDocument());
  });

  it("renders 'clean' label for items with no faults", async () => {
    mockLibraryFetch([{ video_id: "vid_clean", split: "test", view_type: "front", fault_count: 0, faults: [] }]);
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("clean")).toBeInTheDocument());
  });

  it("calls onClose when the close button is clicked", async () => {
    mockLibraryFetch([]);
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(<LibraryPicker onClose={onClose} onPick={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose when the backdrop is clicked", async () => {
    mockLibraryFetch([]);
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container } = renderWithProviders(<LibraryPicker onClose={onClose} onPick={vi.fn()} />);
    // The backdrop is the outermost fixed div
    await user.click(container.firstChild as HTMLElement);
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onPick with the video id when an item is clicked", async () => {
    mockLibraryFetch();
    const user = userEvent.setup();
    const onPick = vi.fn();
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={onPick} />);
    await waitFor(() => screen.getByText("vid_001"));
    await user.click(screen.getByText("vid_001").closest("button")!);
    expect(onPick).toHaveBeenCalledWith("vid_001");
  });

  it("shows the total count in the title", async () => {
    mockLibraryFetch();
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    await waitFor(() => expect(screen.getByText(/\(1\)/)).toBeInTheDocument());
  });

  it("shows filter buttons", async () => {
    mockLibraryFetch([]);
    renderWithProviders(<LibraryPicker onClose={vi.fn()} onPick={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "All" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Knee Valgus" })).toBeInTheDocument();
    });
  });
});
