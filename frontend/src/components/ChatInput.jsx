import { useRef, useState } from 'react'

const MODES = [ // fast/low-precision vs slow/high-precision retrieval -- see retrieval_pipeline.md
  { value: 'simple', label: 'Fast scan', title: 'Scans every candidate in the category and reranks it. Quicker, may miss or over-include some products.' },
  { value: 'structured', label: 'Thorough scan', title: 'Classifies every real category path against your query, then reranks only the uncertain ones. Slower, more precise product counts.' },
]

export default function ChatInput({ onSend, disabled, mode, onModeChange }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const resize = (el) => {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  const handleChange = (e) => {
    setValue(e.target.value)
    resize(e.target)
  }

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="chat-input-bar">
      <div className="mode-toggle" role="radiogroup" aria-label="Retrieval mode">
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            className={`mode-toggle-option ${mode === m.value ? 'active' : ''}`}
            onClick={() => onModeChange(m.value)}
            title={m.title}
            aria-pressed={mode === m.value}
            disabled={disabled}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="chat-input-inner">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          placeholder="Ask about any niche or category…"
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          type="button"
          className="send-button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 19V5M12 5L5 12M12 5L19 12" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
