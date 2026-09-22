import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import FaultCard from "../components/FaultCard";
import { renderWithProviders } from "./renderWithProviders";
import { mockDetection, mockRetrieval } from "./fixtures";

// The card is where a verdict becomes checkable: the rule's own paper and the finding its
// threshold rests on are printed under the causal ladder, straight from the detection.
describe("FaultCard citation", () => {
  it("shows the finding and the reference the rule was built from", () => {
    renderWithProviders(
      <FaultCard d={mockDetection} retrieval={mockRetrieval} active={false} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText(/ACL injury risk with 73% specificity/)).toBeInTheDocument();
    expect(screen.getByTestId("fault-citation")).toHaveTextContent("Ford KR, et al. (2015)");
  });

  // Older stored analyses predate the fields, and a few rules cite nothing: the section is
  // absent rather than an empty labelled block.
  it("omits the section when the detection carries no citation", () => {
    const { citation: _c, citation_support: _s, ...bare } = mockDetection;
    renderWithProviders(
      <FaultCard d={bare} retrieval={mockRetrieval} active={false} onSeek={vi.fn()} />
    );
    expect(screen.queryByText("Evidence")).not.toBeInTheDocument();
    expect(screen.queryByTestId("fault-citation")).not.toBeInTheDocument();
  });

  // The backend sends "" (the dataclass default), not undefined, for a rule that cites nothing.
  it("treats an empty citation string as no citation", () => {
    renderWithProviders(
      <FaultCard
        d={{ ...mockDetection, citation: "", citation_support: "" }}
        retrieval={mockRetrieval}
        active={false}
        onSeek={vi.fn()}
      />
    );
    expect(screen.queryByText("Evidence")).not.toBeInTheDocument();
  });

  // A reference without a one-line finding still earns the section: the paper is the point.
  it("shows the reference alone when the rule has no support line", () => {
    renderWithProviders(
      <FaultCard
        d={{ ...mockDetection, citation_support: "" }}
        retrieval={mockRetrieval}
        active={false}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByTestId("fault-citation")).toHaveTextContent("Ford KR, et al. (2015)");
    expect(screen.queryByText(/ACL injury risk/)).not.toBeInTheDocument();
  });
});
