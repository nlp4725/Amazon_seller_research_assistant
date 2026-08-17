export async function sendChatMessage(messages, mode = 'simple') {
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, mode }),
    })

    if (res.status === 429) {
      return { reply: "You've sent too many requests. Please wait a moment before trying again.", ok: false }
    }
    if (res.status === 400) {
      const data = await res.json().catch(() => ({}))
      return { reply: data.error || 'Invalid request.', ok: false }
    }
    if (!res.ok) {
      return { reply: 'Something went wrong. Please try again.', ok: false }
    }

    const data = await res.json()
    return { reply: data.reply, ok: true }
  } catch {
    return { reply: 'Something went wrong. Please try again.', ok: false }
  }
}
