import { useState } from 'react'
import Logo from './Logo.jsx'
import { EXAMPLES } from '../examples.js'

const YOU_WILL_SEE = [ // only what the report actually contains -- no score or BSR we don't have
  'Launch volume and share of category over the past 24 months',
  'New launches per month',
  'Price distribution of new launches',
  'Top sellers ranked by share of launches',
  'Sub-niche themes with growth rates and trend signals',
  'Most recent product launches',
]

export default function Landing({ onAnalyze }) {
  const [value, setValue] = useState('')

  return (
    <div className="landing">
      <header className="landing-nav">
        <Logo />
        <nav className="landing-links">
          <a href="#">Features</a>
          <a href="#">Pricing</a>
          <a href="#">Docs</a>
        </nav>
        <div className="landing-actions">
          <a className="btn-outline" href="#">Log In</a>
          <a className="btn-dark" href="#">Start for free</a>
        </div>
      </header>

      <main className="landing-main">
        <p className="breadcrumb">Free Tools <span>/</span> Niche Analyzer</p>
        <h1 className="landing-title">Niche Analyzer</h1>
        <p className="landing-powered">Amazon US</p>
        <p className="landing-lede">
          Type any product idea and instantly see how many sellers are entering the market, what
          sub-niches are growing, and who the top players are.
        </p>

        <form className="landing-search" onSubmit={(e) => { e.preventDefault(); onAnalyze(value) }}>
          <input
            className="landing-input"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Enter a product idea..."
            maxLength={1000}
            aria-label="Product idea"
          />
          <div className="country-pill" title="Our data covers Amazon US only">
            <span aria-hidden="true">🇺🇸</span> United States
          </div>
          <button type="submit" className="btn-primary landing-submit" disabled={!value.trim()}>Analyze Niche</button>
        </form>

        <div className="landing-meta">
          <div className="examples">
            <span className="examples-label">For example:</span>
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" className="chip" onClick={() => onAnalyze(ex)}>{ex}</button>
            ))}
          </div>
        </div>

        <hr className="landing-rule" />

        <h2 className="see-title">You will see:</h2>
        <ul className="see-list">
          {YOU_WILL_SEE.map((item) => (
            <li key={item}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M5 12.5l4.5 4.5L19 7.5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {item}
            </li>
          ))}
        </ul>
      </main>
    </div>
  )
}
