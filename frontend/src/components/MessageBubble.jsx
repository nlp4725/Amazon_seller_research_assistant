import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function MessageBubble({ role, content, isLoading }) {
  const isUser = role === 'user'

  return (
    <div className={`message-row ${isUser ? 'from-user' : 'from-assistant'}`}>
      {isUser ? (
        <span className="avatar avatar-user">🧑</span>
      ) : (
        <img className="avatar" src="/assistant_avatar.svg" alt="" />
      )}
      <div className="message-content">
        {isLoading ? (
          <span className="loading-dots" aria-label="Analyzing…">
            <span></span><span></span><span></span>
          </span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        )}
      </div>
    </div>
  )
}
