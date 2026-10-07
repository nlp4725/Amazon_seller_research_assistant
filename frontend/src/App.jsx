import { useEffect, useRef, useState } from 'react'
import Landing from './components/Landing.jsx' // entry page: hero + search + "You will see"
import Results from './components/Results.jsx' // dashboard for one analysed niche
import { analyzeNiche } from './api.js' // calls the Flask REST API's /api/chat endpoint

const urlQuery = () => new URLSearchParams(window.location.search).get('q') ?? '' // ?q=<idea> -> shareable/reloadable result

export default function App() {
  const [search, setSearch] = useState(null) // {query, status: loading|done|clarify|error, reply, data}

  const analyze = async (query, { push = true } = {}) => {
    const q = query.trim()
    if (!q) return
    if (push) window.history.pushState(null, '', `?q=${encodeURIComponent(q)}`) // back/forward replays without pushing
    setSearch({ query: q, status: 'loading' })

    const { reply, data, ok } = await analyzeNiche(q)
    const status = !ok ? 'error' : data ? 'done' : 'clarify' // no data = clarifying question or refusal
    // Ignore a stale response if the user started another search meanwhile.
    setSearch((cur) => (cur?.query === q ? { query: q, status, reply, data } : cur))
  }

  const goHome = () => {
    window.history.pushState(null, '', window.location.pathname)
    setSearch(null)
  }

  const deepLinked = useRef(false) // StrictMode runs effects twice in dev -- don't fire the search twice
  useEffect(() => {
    if (urlQuery() && !deepLinked.current) analyze(urlQuery(), { push: false }) // deep link: run the search in the URL
    deepLinked.current = true
    const onPop = () => (urlQuery() ? analyze(urlQuery(), { push: false }) : setSearch(null)) // browser back/forward
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return search
    ? <Results search={search} onAnalyze={analyze} onHome={goHome} />
    : <Landing onAnalyze={analyze} />
}
