// Three calls, fastest first:
//   analyzeNiche    -> dashboard data, no prose (a few seconds)
//   writeBottomLine -> the Overview card's 2–3 sentences, started automatically after analyze
//   writeFullReport -> the long write-up, only when the user asks for it (~10 s)
// The two writers get back the `report` analyze returned, so they narrate exactly what's on screen.

async function post(path, body) {
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (res.status === 429) {
      return { ok: false, error: "You've made too many requests. Please wait a minute and try again." }
    }
    const data = await res.json().catch(() => ({}))
    if (!res.ok) return { ok: false, error: data.error || 'Something went wrong. Please try again.' }
    return { ok: true, ...data }
  } catch {
    return { ok: false, error: 'Something went wrong. Please try again.' }
  }
}

export const analyzeNiche = (query) => post('/api/analyze', { query }) // {clarify} | {report, data}

export const writeBottomLine = (query, report) => post('/api/bottom-line', { query, report }) // {text}

export const writeFullReport = (query, report) => post('/api/report', { query, report }) // {text}
