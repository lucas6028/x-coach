import ReactMarkdown, { type Components } from "react-markdown";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";

// The coach's answer is LLM text rendered as HTML, so it MUST be sanitized. We derive from
// rehype-sanitize's default (GitHub) schema — which strips <script>, event handlers, and
// javascript: URLs — and additionally drop <a>/<img>: the coach speaks only from the analysis and
// has no reason to emit links or images, so removing them keeps the rendered surface minimal.
// (react-markdown already escapes raw HTML by default since rehype-raw is not enabled; the sanitize
// pass is defence-in-depth and is what actually enforces the no-links rule on real markdown links.)
const schema = {
  ...defaultSchema,
  tagNames: (defaultSchema.tagNames ?? []).filter((t) => t !== "a" && t !== "img"),
};

// Map each element to a token-styled node (no @tailwindcss/typography dependency). `node` is dropped
// so it never lands on a DOM element. The coach emits short answers: paragraphs, bold cues, lists,
// inline code/timecodes — headings collapse to the same quiet subhead.
const components: Components = {
  p: ({ node: _n, ...props }) => (
    <p className="text-[15px] leading-relaxed text-content [&:not(:first-child)]:mt-2" {...props} />
  ),
  ul: ({ node: _n, ...props }) => (
    <ul className="mt-2 list-disc space-y-1 pl-5 text-[15px] leading-relaxed text-content" {...props} />
  ),
  ol: ({ node: _n, ...props }) => (
    <ol
      className="mt-2 list-decimal space-y-1 pl-5 text-[15px] leading-relaxed text-content"
      {...props}
    />
  ),
  li: ({ node: _n, ...props }) => <li className="marker:text-faint" {...props} />,
  strong: ({ node: _n, ...props }) => <strong className="font-semibold text-content" {...props} />,
  em: ({ node: _n, ...props }) => <em className="italic" {...props} />,
  code: ({ node: _n, ...props }) => (
    <code
      className="rounded bg-content/[0.06] px-1 py-0.5 font-mono text-[13px] text-primary"
      {...props}
    />
  ),
  h1: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h2: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h3: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
};

// Render a coach message as sanitized markdown. Tolerant of partial input mid-stream (unterminated
// emphasis just renders as text).
export default function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown rehypePlugins={[[rehypeSanitize, schema]]} components={components}>
      {children}
    </ReactMarkdown>
  );
}
