import { useEffect, useRef, useState } from 'react'
import Landing from './components/Landing.jsx' // entry page: hero + search + "You will see"
import Results from './components/Results.jsx' // dashboard for one analysed niche
import { analyzeNiche, writeBottomLine, writeFullReport } from './api.js'

const urlQuery = () => new URLSearchParams(window.location.search).get('q') ?? '' // ?q=<idea> -> shareable/reloadable result
const WRITERS = { bottomLine: writeBottomLine, fullReport: writeFullReport }

// search: {id, query, status: loading|done|clarify|error, message, data, report,
//          bottomLine: {status, text, error}, fullReport: {status, text, error}}
// Writer status: idle | loading | done | error.
export default function App() {
  const [search, setSearch] = useState(null)
  const seq = useRef(0) // id of the latest search; responses for older searches are dropped

  const update = (id, patch) => setSearch((cur) => (cur?.id === id ? { ...cur, ...patch(cur) } : cur))

  const write = async (id, kind, query, report) => {
    update(id, () => ({ [kind]: { status: 'loading' } }))
    const res = await WRITERS[kind](query, report)
    update(id, () => ({ [kind]: res.ok ? { status: 'done', text: res.text } : { status: 'error', error: res.error } }))
  }

  const analyze = async (query, { push = true } = {}) => {
    const q = query.trim()
    if (!q) return
    if (push) window.history.pushState(null, '', `?q=${encodeURIComponent(q)}`) // back/forward replays without pushing
    const id = ++seq.current
    setSearch({ id, query: q, status: 'loading' })

    const res = await analyzeNiche(q)
    if (!res.ok) return update(id, () => ({ status: 'error', message: res.error }))
    if (res.clarify) return update(id, () => ({ status: 'clarify', message: res.clarify }))
    if (!res.data) return update(id, () => ({ status: 'clarify', message: res.report?.error ?? 'No matching products found.' }))

    update(id, () => ({
      status: 'done', data: res.data, report: res.report,
      bottomLine: { status: 'loading' }, fullReport: { status: 'idle' },
    }))
    write(id, 'bottomLine', q, res.report) // fills the Overview card a moment after the dashboard
  }

  const showLanding = () => {
    seq.current++ // drop any in-flight responses
    setSearch(null)
  }

  const goHome = () => {
    window.history.pushState(null, '', window.location.pathname)
    showLanding()
  }

  const deepLinked = useRef(false) // StrictMode runs effects twice in dev -- don't fire the search twice
  useEffect(() => {
    if (urlQuery() && !deepLinked.current) analyze(urlQuery(), { push: false }) // deep link: run the search in the URL
    deepLinked.current = true
    const onPop = () => (urlQuery() ? analyze(urlQuery(), { push: false }) : showLanding()) // browser back/forward
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return search
    ? (
      <Results
        search={search}
        onAnalyze={analyze}
        onHome={goHome}
        onWrite={(kind) => write(search.id, kind, search.query, search.report)}
      />
    )
    : <Landing onAnalyze={analyze} />
}
