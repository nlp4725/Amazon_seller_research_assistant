import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

export default function MessageList({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages.length, isLoading])

  return (
    <div className="message-list">
      {messages.map((m, i) => (
        <MessageBubble key={i} role={m.role} content={m.content} />
      ))}
      {isLoading && <MessageBubble role="assistant" isLoading />}
      <div ref={bottomRef} />
    </div>
  )
}
