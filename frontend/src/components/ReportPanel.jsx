import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function formatTimestamp(date) {
  const pad = (n) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function buildFilename(title) {
  return title
    .replace(/[📊🟢]/gu, '')
    .trim()
    .replace(/ /g, '_')
    .replace(/:/g, '') + '.txt'
}

export default function ReportPanel({ panel, onClear }) {
  const subject = panel.title.replace(/[#*]/g, '').trim()
  const body = panel.content.replace(/[#*|]/g, '')
  const mailto = `mailto:nasilipurcell@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`

  const handleDownload = () => {
    const timestamp = formatTimestamp(new Date())
    const content = `${panel.title}\nGenerated: ${timestamp}\n\n${panel.content}`
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = buildFilename(panel.title)
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  return (
    <div className="report-panel">
      <div className="report-panel-title">{panel.title}</div>

      <div className="report-panel-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{panel.content}</ReactMarkdown>
      </div>

      <div className="report-panel-actions">
        <a className="btn" href={mailto}>📧 Email me now</a>
        <button className="btn" onClick={handleDownload}>⬇️ Download report</button>
      </div>
      <button className="btn btn-clear" onClick={onClear}>✕ Clear</button>
    </div>
  )
}
