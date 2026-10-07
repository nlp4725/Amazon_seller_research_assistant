import { useEffect, useState } from 'react'
import Logo from './Logo.jsx'
import ChartCard from './blocks/ChartCard.jsx'
import LaunchChart from './blocks/LaunchChart.jsx'
import PriceChart from './blocks/PriceChart.jsx'
import CumulativeChart from './blocks/CumulativeChart.jsx'
import ThemeTable from './blocks/ThemeTable.jsx'
import SellerTable from './blocks/SellerTable.jsx'
import ProductTable from './blocks/ProductTable.jsx'
import { Markdown, splitBottomLine } from './blocks/Summary.jsx'
import { fmtPct, fmtPrice } from './blocks/format.js'
import { EXAMPLES } from '../examples.js'

const TABS = ['Overview', 'Sellers', 'Themes', 'Launches']

function TopBar({ query, busy, onAnalyze, onHome }) {
  const [value, setValue] = useState(query)
  useEffect(() => setValue(query), [query]) // follow chip clicks and back/forward

  return (
    <header className="topbar">
      <Logo onClick={onHome} />
      <form className="topbar-search" onSubmit={(e) => { e.preventDefault(); onAnalyze(value) }}>
        <span className="topbar-search-label">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M20 20l-3.5-3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Niche
        </span>
        <input value={value} onChange={(e) => setValue(e.target.value)} maxLength={1000} aria-label="Product idea" />
        <button type="submit" className="btn-primary topbar-submit" disabled={busy || !value.trim()}>Analyze</button>
      </form>
    </header>
  )
}

function StatCards({ stats }) {
  const cards = [
    {
      label: stats.is_match_count ? 'Matching launches' : 'Products analysed',
      value: stats.products.toLocaleString(),
      sub: stats.pct_of_category != null ? `${fmtPct(stats.pct_of_category)} of category launches` : 'whole category',
    },
    { label: 'Category launches', value: stats.category_total.toLocaleString(), sub: stats.categories.join(' + ') },
    { label: 'Median price', value: fmtPrice(stats.median_price), sub: 'of these launches' },
    {
      label: 'Unique sellers',
      value: stats.unique_sellers.toLocaleString(),
      sub: stats.top_sellers_n ? `Top ${stats.top_sellers_n} hold ${fmtPct(stats.top_sellers_pct)}` : null,
    },
    stats.candidates_considered != null
      ? { label: 'Candidates checked', value: stats.candidates_considered.toLocaleString(), sub: 'classified and filtered by Jev' }
      : { label: 'Candidates checked', value: '—', sub: 'whole category, no product filter' },
  ]

  return (
    <div className="stat-cards">
      {cards.map((c) => (
        <div key={c.label} className="stat-card">
          <p className="stat-label">{c.label}</p>
          <p className="stat-value">{c.value}</p>
          {c.sub && <p className="stat-sub" title={c.sub}>{c.sub}</p>}
        </div>
      ))}
    </div>
  )
}

function Overview({ data, reply }) {
  const [prose, bottomLine] = splitBottomLine(reply)
  return (
    <div className="overview">
      <div className="grid grid-wide">
        <ChartCard title="New Launches Per Month" subtitle={`Monthly standalone product launches. ${data.note}`}>
          <LaunchChart data={data.launches_by_month} />
        </ChartCard>
        <ChartCard title="Bottom Line" subtitle="Analyst read of this niche" className="panel-bottomline">
          {bottomLine ? <Markdown>{bottomLine}</Markdown> : <p className="panel-empty">See the full analysis below.</p>}
        </ChartCard>
      </div>
      <div className="grid grid-even">
        <ChartCard
          title="Price Distribution" subtitle="Where new launches are priced"
          badge={data.price_peak && `Peak: ${data.price_peak}`}
        >
          <PriceChart data={data.price_distribution} peak={data.price_peak} />
        </ChartCard>
        <ChartCard title="Cumulative Launches" subtitle="Total matching launches over time">
          <CumulativeChart data={data.launches_by_month} />
        </ChartCard>
      </div>
      <ChartCard title="Full Analysis" subtitle="Written from the numbers above">
        <Markdown>{prose}</Markdown>
      </ChartCard>
    </div>
  )
}

function TabPanel({ tab, data, reply }) {
  if (tab === 'Overview') return <Overview data={data} reply={reply} />
  if (tab === 'Sellers') {
    return (
      <ChartCard title="Top Sellers" subtitle="Ranked by number of launches. Share is of all analysed launches.">
        {data.top_sellers.length ? <SellerTable sellers={data.top_sellers} /> : <p className="panel-empty">No identified sellers.</p>}
      </ChartCard>
    )
  }
  if (tab === 'Themes') {
    return (
      <ChartCard title="Sub-niche Themes" subtitle="Launches grouped by title similarity, with yearly counts and growth">
        {data.themes.length ? <ThemeTable themes={data.themes} /> : <p className="panel-empty">Too few launches to group into themes.</p>}
      </ChartCard>
    )
  }
  return (
    <ChartCard title="Most Recent Launches" subtitle="Newest matching standalone listings">
      {data.recent_launches.length ? <ProductTable products={data.recent_launches} /> : <p className="panel-empty">No launches.</p>}
    </ChartCard>
  )
}

function Loading({ query }) {
  return (
    <div aria-busy="true">
      <p className="loading-text">Analyzing “{query}”… this usually takes 20–40 seconds.</p>
      <div className="stat-cards">
        {[0, 1, 2, 3, 4].map((i) => <div key={i} className="stat-card skeleton" />)}
      </div>
      <div className="grid grid-wide">
        <div className="panel skeleton skeleton-tall" />
        <div className="panel skeleton skeleton-tall" />
      </div>
    </div>
  )
}

export default function Results({ search, onAnalyze, onHome }) {
  const [tab, setTab] = useState('Overview')
  useEffect(() => setTab('Overview'), [search.query]) // a new search starts on Overview

  const { query, status, data, reply } = search
  const busy = status === 'loading'

  return (
    <div className="results">
      <TopBar query={query} busy={busy} onAnalyze={onAnalyze} onHome={onHome} />

      <main className="results-main">
        <div className="results-head">
          <h1 className="results-title">“{query}”</h1>
          <span className="pill">Amazon US</span>
          <span className="pill">2024–2026 launches</span>
          <span className="pill">Standalone listings only</span>
        </div>
        <p className="results-meta">
          Amazon US launch database{data ? ` · ${data.stats.categories.join(' + ')} · ${data.stats.category_total.toLocaleString()} category launches indexed` : ''}
        </p>
        <div className="examples">
          <span className="examples-label">Try:</span>
          {EXAMPLES.filter((ex) => ex !== query).map((ex) => (
            <button key={ex} type="button" className="chip" disabled={busy} onClick={() => onAnalyze(ex)}>{ex}</button>
          ))}
        </div>

        {busy && <Loading query={query} />}
        {(status === 'clarify' || status === 'error') && (
          <div className={`notice ${status === 'error' ? 'notice-error' : ''}`}>
            <Markdown>{reply}</Markdown>
            {status === 'clarify' && <p className="notice-hint">Refine your search above and press Analyze.</p>}
          </div>
        )}
        {status === 'done' && (
          <>
            <StatCards stats={data.stats} />
            <nav className="tabs" role="tablist">
              {TABS.map((t) => (
                <button
                  key={t} type="button" role="tab" aria-selected={tab === t}
                  className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </nav>
            <TabPanel tab={tab} data={data} reply={reply} />
          </>
        )}
      </main>
    </div>
  )
}
