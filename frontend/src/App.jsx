import { useEffect, useMemo, useState } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from 'recharts'

// `??` (not `||`) so an explicitly empty VITE_API_URL stays empty, producing
// relative /api/* fetches when frontend and backend share a domain (Vercel).
// Unset (local dev) still falls through to the local uvicorn default.
const API = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const CLASS_ORDER = ['Economy', 'Premium Economy', 'Business', 'First']

// Muted greys for non-selected classes; the currently-selected cabin is the
// only one drawn in gold. No second accent colour.
const CLASS_GREY = {
  Economy: '#9b9aa1',
  'Premium Economy': '#7c7b82',
  Business: '#5c5b62',
  First: '#413f45',
}

const inr = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
})

const inrCompact = (v) => `₹${Math.round(v).toLocaleString('en-IN')}`

async function postJSON(path, body, signal) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (res.status === 404) {
    const err = new Error('No flights on record for this route and cabin.')
    err.notFound = true
    throw err
  }
  if (!res.ok) {
    throw new Error(`Request failed (${res.status})`)
  }
  return res.json()
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="chart-tooltip">
      <div className="tt-days">{label} days before departure</div>
      <div className="tt-price">{inrCompact(payload[0].value)}</div>
    </div>
  )
}

function CompareTooltip({ active, payload, label, activeClass }) {
  if (!active || !payload || !payload.length) return null
  const rows = [...payload].sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  return (
    <div className="chart-tooltip">
      <div className="tt-days">{label} days before departure</div>
      {rows.map((r) => (
        <div className="tt-row" key={r.dataKey}>
          <span
            className="tt-swatch"
            style={{ background: r.dataKey === activeClass ? '#e8b04b' : CLASS_GREY[r.dataKey] }}
          />
          <span>{r.dataKey}</span>
          <span style={{ marginLeft: 'auto', color: '#ece9e4' }}>{r.value?.toFixed(0)}</span>
        </div>
      ))}
    </div>
  )
}

// Merge each class's [{days, index}] series into one row-per-day array,
// keyed by travel_class, for a multi-line recharts chart.
function buildCompareSeries(classes) {
  const byDay = new Map()
  for (let d = 0; d <= 120; d++) byDay.set(d, { days: d })
  for (const c of classes) {
    for (const pt of c.indexed) {
      byDay.get(pt.days)[c.travel_class] = pt.index
    }
  }
  return Array.from(byDay.values())
}

export default function App() {
  const [options, setOptions] = useState(null)
  const [optionsError, setOptionsError] = useState(null)

  const [source, setSource] = useState('Delhi')
  const [destination, setDestination] = useState('Mumbai')
  const [travelClass, setTravelClass] = useState('Economy')
  const [daysBefore, setDaysBefore] = useState(30)

  const [predictData, setPredictData] = useState(null)
  const [predictError, setPredictError] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)

  const [forecastData, setForecastData] = useState(null)
  const [forecastError, setForecastError] = useState(null)

  const [compareData, setCompareData] = useState(null)
  const [compareError, setCompareError] = useState(null)

  // Load dropdown options once
  useEffect(() => {
    fetch(`${API}/api/options`)
      .then((r) => r.json())
      .then(setOptions)
      .catch((e) => setOptionsError(e.message))
  }, [])

  // Predict: re-fetch on every control change, including the slider
  useEffect(() => {
    const controller = new AbortController()
    setPredictLoading(true)
    postJSON(
      '/api/predict',
      { source, destination, travel_class: travelClass, days_before: daysBefore },
      controller.signal
    )
      .then((data) => {
        setPredictData(data)
        setPredictError(null)
      })
      .catch((e) => {
        if (e.name === 'AbortError') return
        setPredictData(null)
        setPredictError(e.notFound ? e.message : 'Could not reach the prediction service.')
      })
      .finally(() => setPredictLoading(false))
    return () => controller.abort()
  }, [source, destination, travelClass, daysBefore])

  // Forecast: only depends on route + cabin, independent of the slider
  useEffect(() => {
    const controller = new AbortController()
    postJSON(
      '/api/forecast',
      { source, destination, travel_class: travelClass, days_before: daysBefore },
      controller.signal
    )
      .then((data) => {
        setForecastData(data)
        setForecastError(null)
      })
      .catch((e) => {
        if (e.name === 'AbortError') return
        setForecastData(null)
        setForecastError(e.notFound ? e.message : 'Could not reach the forecast service.')
      })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, destination, travelClass])

  // Compare: cabin-class curves for the route, independent of cabin + slider
  useEffect(() => {
    const controller = new AbortController()
    postJSON('/api/compare', { source, destination }, controller.signal)
      .then((data) => {
        setCompareData(data)
        setCompareError(null)
      })
      .catch((e) => {
        if (e.name === 'AbortError') return
        setCompareData(null)
        setCompareError(e.notFound ? e.message : 'Could not reach the comparison service.')
      })
    return () => controller.abort()
  }, [source, destination])

  const compareSeries = useMemo(
    () => (compareData ? buildCompareSeries(compareData.classes) : null),
    [compareData]
  )

  if (optionsError) {
    return (
      <div className="app-loading">
        <pre style={{ fontFamily: 'Inter', fontStyle: 'normal', fontSize: 14, color: '#b5645c' }}>
          Error: {optionsError}
        </pre>
      </div>
    )
  }

  if (!options) {
    return <div className="app-loading">Loading AeroLens…</div>
  }

  const notFound = predictError && predictData === null
  const showChart = forecastData && !forecastError

  return (
    <>
      <div className="mesh-bg">
        <div className="blob blob-blue" />
        <div className="blob blob-purple" />
        <div className="blob blob-teal" />
        <div className="blob blob-gold" />
      </div>
      <div className="main-scrim" />
      <div className="grid-overlay" />

      <div className="app">
        <aside className="sidebar">
          <div>
            <div className="wordmark">AeroLens</div>
            <div className="wordmark-sub">Flight Price Intelligence</div>
          </div>

          <div className="controls">
            <div className="control-group">
              <label className="eyebrow" htmlFor="source">From</label>
              <div className="select-wrap">
                <select
                  id="source"
                  className="control-select"
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                >
                  {options.sources.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="control-group">
              <label className="eyebrow" htmlFor="destination">To</label>
              <div className="select-wrap">
                <select
                  id="destination"
                  className="control-select"
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                >
                  {options.destinations.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="control-group">
              <label className="eyebrow" htmlFor="class">Cabin</label>
              <div className="select-wrap">
                <select
                  id="class"
                  className="control-select"
                  value={travelClass}
                  onChange={(e) => setTravelClass(e.target.value)}
                >
                  {options.classes.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="control-group">
              <div className="slider-head">
                <span className="eyebrow">Lead Time</span>
                <span className="slider-value">
                  {daysBefore}
                  <sub>days</sub>
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={120}
                value={daysBefore}
                className="control-slider"
                onChange={(e) => setDaysBefore(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="sidebar-foot">
            Predictions from an XGBoost model trained on 78,221 cleaned
            historical fares. Forecast curves average 100 real flights per
            route to smooth model step-noise.
          </div>
        </aside>

        <main className="main">
          <div className="eyebrow-line">
            {source} → {destination}
            <span className="dot">·</span>
            {travelClass}
            <span className="dot">·</span>
            {daysBefore} days out
          </div>

          {notFound ? (
            <div className="notice-panel">
              <div className="notice-title">No Data</div>
              {predictError}
            </div>
          ) : (
            <>
              <div className={`hero-block${predictLoading ? ' is-loading' : ''}`}>
                <div className="hero-price">
                  {predictData ? inr.format(predictData.predicted_price) : '—'}
                </div>
                <div className="sub-line">
                  {predictData ? (
                    <>
                      <span className="accent">{inrCompact(predictData.low)}</span>
                      {' – '}
                      <span className="accent">{inrCompact(predictData.high)}</span>
                      {'  ·  '}
                      {predictData.distance_km.toLocaleString('en-IN')} km
                      {'  ·  '}
                      {predictData.n_reference_flights.toLocaleString('en-IN')} reference flights
                    </>
                  ) : (
                    'Fetching prediction…'
                  )}
                </div>
              </div>

              <div className={`stats-row${predictLoading ? ' hero-block is-loading' : ''}`}>
                <div className="stat-panel">
                  <div className="stat-value">
                    {forecastData ? `${forecastData.book_by_days}d` : '—'}
                  </div>
                  <div className="stat-label">Book By</div>
                  <div className="stat-caption">
                    Latest lead time at which price is still within 5% of its
                    floor. Booking earlier buys little.
                  </div>
                </div>

                <div className="stat-panel">
                  <div className="stat-value">
                    {forecastData ? `${forecastData.premium_pct}%` : '—'}
                  </div>
                  <div className="stat-label">Last-Minute Premium</div>
                  <div className="stat-caption">
                    Average price booking 0–3 days out versus 60–120 days
                    out.
                  </div>
                </div>

                <div className="stat-panel">
                  <div className="stat-value">
                    {forecastData ? inrCompact(forecastData.early_price) : '—'}
                  </div>
                  <div className="stat-label">Early Price</div>
                  <div className="stat-caption">
                    Average predicted price when booking 60–120 days ahead
                    of departure.
                  </div>
                </div>
              </div>

              <div className="chart-panel">
                <div className="chart-header">
                  <span className="chart-title">Forecast Curve · 0–120 Days</span>
                  <span className="chart-note">Smoothed, 7-day rolling mean</span>
                </div>

                {showChart ? (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={forecastData.curve} margin={{ top: 34, right: 16, left: -8, bottom: 8 }}>
                      <XAxis
                        dataKey="days"
                        reversed
                        ticks={[0, 20, 40, 60, 80, 100, 120]}
                        tick={{ fill: '#6a6a72', fontSize: 11, fontFamily: 'Inter' }}
                        axisLine={{ stroke: 'rgba(255,255,255,0.09)' }}
                        tickLine={false}
                        label={{
                          value: 'DAYS BEFORE DEPARTURE',
                          position: 'insideBottom',
                          offset: -4,
                          fill: '#6a6a72',
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        domain={['dataMin - 1000', 'dataMax + 500']}
                        tick={{ fill: '#6a6a72', fontSize: 11, fontFamily: 'Inter' }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(v) => inrCompact(v)}
                        width={64}
                      />
                      <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.15)' }} />
                      <ReferenceLine
                        x={forecastData.book_by_days}
                        stroke="#e8b04b"
                        strokeDasharray="3 4"
                        strokeOpacity={0.55}
                        label={{
                          value: `Book by ${forecastData.book_by_days}d`,
                          position: 'top',
                          fill: '#e8b04b',
                          fontSize: 11,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#e8b04b"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 320, display: 'flex', alignItems: 'center', color: '#6a6a72', fontSize: 13 }}>
                    {forecastError || 'Loading forecast curve…'}
                  </div>
                )}
              </div>

              <div className="chart-panel" style={{ marginTop: 24 }}>
                <div className="chart-header">
                  <span className="chart-title">Cabin Class Comparison · Indexed</span>
                  <span className="chart-note">Floor = 100</span>
                </div>

                {compareSeries ? (
                  <>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={compareSeries} margin={{ top: 10, right: 16, left: -8, bottom: 8 }}>
                        <XAxis
                          dataKey="days"
                          reversed
                          ticks={[0, 20, 40, 60, 80, 100, 120]}
                          tick={{ fill: '#6a6a72', fontSize: 11, fontFamily: 'Inter' }}
                          axisLine={{ stroke: 'rgba(255,255,255,0.09)' }}
                          tickLine={false}
                          label={{
                            value: 'DAYS BEFORE DEPARTURE',
                            position: 'insideBottom',
                            offset: -4,
                            fill: '#6a6a72',
                            fontSize: 10,
                          }}
                        />
                        <YAxis
                          domain={['dataMin - 5', 'dataMax + 5']}
                          tick={{ fill: '#6a6a72', fontSize: 11, fontFamily: 'Inter' }}
                          axisLine={false}
                          tickLine={false}
                          width={52}
                          tickFormatter={(v) => v.toFixed(0)}
                        />
                        <Tooltip content={<CompareTooltip activeClass={travelClass} />} cursor={{ stroke: 'rgba(255,255,255,0.15)' }} />
                        {compareData.classes.map((c) => {
                          const isActive = c.travel_class === travelClass
                          return (
                            <Line
                              key={c.travel_class}
                              type="monotone"
                              dataKey={c.travel_class}
                              stroke={isActive ? '#e8b04b' : CLASS_GREY[c.travel_class]}
                              strokeWidth={isActive ? 2.5 : 1.3}
                              strokeOpacity={isActive ? 1 : 0.7}
                              dot={false}
                              isAnimationActive={false}
                            />
                          )
                        })}
                      </LineChart>
                    </ResponsiveContainer>

                    <div className="compare-legend">
                      {CLASS_ORDER.filter((cls) =>
                        compareData.classes.some((c) => c.travel_class === cls)
                      ).map((cls) => {
                        const c = compareData.classes.find((x) => x.travel_class === cls)
                        const isActive = cls === travelClass
                        return (
                          <div key={cls} className={`compare-legend-item${isActive ? ' is-active' : ''}`}>
                            <span
                              className="compare-legend-swatch"
                              style={{ background: isActive ? '#e8b04b' : CLASS_GREY[cls] }}
                            />
                            <span>{cls}</span>
                            <span className="compare-legend-premium" style={{ color: isActive ? '#e8b04b' : undefined }}>
                              {c.premium_pct}%
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </>
                ) : (
                  <div style={{ height: 280, display: 'flex', alignItems: 'center', color: '#6a6a72', fontSize: 13 }}>
                    {compareError || 'Loading cabin comparison…'}
                  </div>
                )}
              </div>
            </>
          )}

          <div className="transparency-footer">
            Model R&sup2; 0.897 · Median error ~15% · Trained on 78,221 cleaned historical flights · XGBoost regression
          </div>
        </main>
      </div>
    </>
  )
}
