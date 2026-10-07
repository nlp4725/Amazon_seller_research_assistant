import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// The report's last section (see SYSTEM in analysis_agent.py), however Haiku formats the
// heading: "BOTTOM LINE:", "**BOTTOM LINE**", "## Bottom Line", ...
const BOTTOM_LINE = /^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*bottom line[ \t]*:?[ \t]*(?:\*\*)?[ \t]*:?[ \t]*/im

export function splitBottomLine(content) { // -> [prose before, bottom line text or null]
  const m = BOTTOM_LINE.exec(content)
  if (!m) return [content, null] // heading not found: everything is prose, no highlighted box
  const rest = content.slice(m.index + m[0].length).trim()
  return rest ? [content.slice(0, m.index).trim(), rest] : [content, null]
}

export function Markdown({ children }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ table: (props) => <div className="table-scroll"><table {...props} /></div> }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
