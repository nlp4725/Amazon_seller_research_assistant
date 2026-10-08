import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

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
