import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";

// The coach's answer is LLM text rendered as HTML, so it MUST be sanitized. We derive from
// rehype-sanitize's default (GitHub) schema — which strips <script>, event handlers, and
// javascript: URLs — and additionally drop <a>/<img>: the coach speaks only from the analysis and
// has no reason to emit links or images, so removing them keeps the rendered surface minimal.
// (react-markdown already escapes raw HTML by default since rehype-raw is not enabled; the sanitize
// pass is defence-in-depth and is what actually enforces the no-links rule on real markdown links —
// including the bare URLs that remark-gfm auto-links, whose <a> is stripped the same way.)
const schema = {
  ...defaultSchema,
  tagNames: (defaultSchema.tagNames ?? []).filter((t) => t !== "a" && t !== "img"),
  // remark-gfm tags task lists with these classes; allow just those exact values through (not
  // arbitrary className) so the components below can drop the disc bullet for a checkbox row.
  attributes: {
    ...defaultSchema.attributes,
    ul: [...(defaultSchema.attributes?.ul ?? []), ["className", "contains-task-list"]],
    li: [...(defaultSchema.attributes?.li ?? []), ["className", "task-list-item"]],
  },
};

// Map each element to a token-styled node (no @tailwindcss/typography dependency). `node` is dropped
// so it never lands on a DOM element. Covers the coach's full markdown surface: paragraphs, bold
// cues, lists (incl. gfm task lists), inline/block code, tables, blockquotes, rules, strikethrough
// — all heading levels collapse to the same quiet subhead.
const components: Components = {
  p: ({ node: _n, ...props }) => (
    <p className="text-[15px] leading-relaxed text-content [&:not(:first-child)]:mt-2" {...props} />
  ),
  ul: ({ node: _n, className, ...props }) => {
    // A gfm task list (contains-task-list) drops the disc bullet — the checkbox is the marker.
    const isTaskList = (className ?? "").includes("contains-task-list");
    return (
      <ul
        className={`mt-2 space-y-1 text-[15px] leading-relaxed text-content ${
          isTaskList ? "list-none pl-1" : "list-disc pl-5"
        }`}
        {...props}
      />
    );
  },
  ol: ({ node: _n, ...props }) => (
    <ol
      className="mt-2 list-decimal space-y-1 pl-5 text-[15px] leading-relaxed text-content"
      {...props}
    />
  ),
  li: ({ node: _n, className, ...props }) => {
    const isTaskItem = (className ?? "").includes("task-list-item");
    return <li className={isTaskItem ? "list-none" : "marker:text-faint"} {...props} />;
  },
  strong: ({ node: _n, ...props }) => <strong className="font-semibold text-content" {...props} />,
  em: ({ node: _n, ...props }) => <em className="italic" {...props} />,
  del: ({ node: _n, ...props }) => <del className="text-faint line-through" {...props} />,
  // The gfm task-list checkbox (disabled). Keep it inert and aligned with its label.
  input: ({ node: _n, ...props }) => (
    <input className="mr-1.5 align-middle accent-primary" {...props} />
  ),
  // One `code` handles both inline and fenced blocks (react-markdown routes both here). A fenced/
  // indented block carries a language-* class or spans multiple lines → render plain inside the
  // styled <pre>; an inline span gets the pill. String(children) is the code text either way.
  code: ({ node: _n, className, children, ...props }) => {
    const isBlock = /language-/.test(className ?? "") || String(children).includes("\n");
    return isBlock ? (
      <code className={`font-mono text-[13px] text-content/90 ${className ?? ""}`} {...props}>
        {children}
      </code>
    ) : (
      <code
        className="rounded bg-content/[0.06] px-1 py-0.5 font-mono text-[13px] text-primary"
        {...props}
      >
        {children}
      </code>
    );
  },
  // The <pre> wrapper of a fenced block: the scrollable, padded surface (the inner <code> renders
  // plain, so there's no double background).
  pre: ({ node: _n, ...props }) => (
    <pre
      className="mt-2 overflow-x-auto rounded-md bg-content/[0.06] p-3 leading-relaxed"
      {...props}
    />
  ),
  blockquote: ({ node: _n, ...props }) => (
    <blockquote
      className="mt-2 border-l-2 border-content/15 pl-3 text-[15px] italic text-content/80"
      {...props}
    />
  ),
  hr: ({ node: _n, ...props }) => <hr className="my-3 border-content/10" {...props} />,
  // h1–h6 all collapse to the same quiet subhead.
  h1: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h2: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h3: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h4: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h5: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  h6: ({ node: _n, ...props }) => <h3 className="mt-3 text-sm font-semibold text-content" {...props} />,
  // GFM tables (via remark-gfm). The table scrolls inside its own container so a wide table never
  // pushes the chat column horizontally.
  table: ({ node: _n, ...props }) => (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full border-collapse text-[14px] text-content" {...props} />
    </div>
  ),
  th: ({ node: _n, ...props }) => (
    <th
      className="border border-content/10 px-2 py-1 text-left font-semibold text-content"
      {...props}
    />
  ),
  td: ({ node: _n, ...props }) => (
    <td className="border border-content/10 px-2 py-1 align-top" {...props} />
  ),
};

// Render a coach message as sanitized markdown. Tolerant of partial input mid-stream (unterminated
// emphasis just renders as text).
export default function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[[rehypeSanitize, schema]]}
      components={components}
    >
      {children}
    </ReactMarkdown>
  );
}
