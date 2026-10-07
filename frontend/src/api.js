// One niche question in, one report out. Goes through /api/chat as a single-message
// conversation -- the backend's cat_selector + niche_report flow is unchanged. Retrieval is
// always the structured pipeline (TypeSafe Jev classifies paths and filters titles).
export async function analyzeNiche(query) {
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: query }], mode: 'structured' }), // Jev scan -- the only method the UI offers
    })

    if (res.status === 429) {
      return { reply: "You've run too many searches. Please wait a moment before trying again.", ok: false }
    }
    if (res.status === 400) {
      const data = await res.json().catch(() => ({}))
      return { reply: data.error || 'Invalid request.', ok: false }
    }
    if (!res.ok) {
      return { reply: 'Something went wrong. Please try again.', ok: false }
    }

    const data = await res.json()
    return { reply: data.reply, data: data.data ?? null, ok: true }
  } catch {
    return { reply: 'Something went wrong. Please try again.', ok: false }
  }
}
