const COND_COLORS = {
  reduced_mobility:           '#d97706',
  heart_failure:              '#dc2626',
  COPD:                       '#2563eb',
  sleep_disordered_breathing: '#7c3aed',
}
const COND_LABELS = {
  reduced_mobility:           'Reduced Mobility',
  heart_failure:              'Heart Failure',
  COPD:                       'COPD',
  sleep_disordered_breathing: 'Sleep-Disordered Breathing',
}

const CONDITION_TO_STATE = {
  reduced_mobility: 'M_mobility',
  heart_failure: 'CP_cardio',
  COPD: 'CP_cardio',
  sleep_disordered_breathing: 'SDB_sleep',
}

const LATENT_STATE_LABELS = {
  M_mobility: 'M - Mobility',
  CP_cardio: 'CP - Cardiopulmonary',
  SDB_sleep: 'SDB - Sleep',
}

const DAILY_SUMMARY_COLUMNS = [
  { key: 'day', label: 'Day' },
  { key: 'ground_truth_label', label: 'Label', format: d => d.replace(/_/g, ' ') },
  { key: 'active_duration', label: 'Active duration' },
  { key: 'sleep_window', label: 'Sleep window' },
  { key: 'night_bathroom_visits', label: 'Night bathroom visits' },
  { key: 'breathing_rate_mean', label: 'Breathing rate' },
  { key: 'bed_exit_events', label: 'Bed exit events' },
]

const conditions = {}
const charts = {}
let globalDays = []

function toggleCond(el) {
  const c = el.dataset.cond
  if (el.classList.contains('active')) {
    el.classList.remove('active')
    delete conditions[c]
  } else {
    el.classList.add('active')
    conditions[c] = 'mild'
  }
}

function setSev(e, cond, sev) {
  e.stopPropagation()
  conditions[cond] = sev
  const card = document.querySelector(`[data-cond="${cond}"]`)
  card.querySelectorAll('.sev-btn').forEach(b => b.classList.remove('active'))
  e.target.classList.add('active')
}

function makeChart(id, datasets, min, max) {
  if (charts[id]) charts[id].destroy()
  const ctx = document.getElementById(id).getContext('2d')
  charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels: globalDays.map(d => d.day), datasets },
    options: {
      responsive: true,
      animation: { duration: 300 },
      plugins: { legend: { display: datasets.length > 1, labels: { boxWidth: 10, font: { size: 11 }, color: '#6b7280' } } },
      scales: {
        x: { ticks: { color: '#9ca3af', maxTicksLimit: 8 }, grid: { color: '#f3f4f6' } },
        y: { ticks: { color: '#9ca3af' }, grid: { color: '#f3f4f6' }, min, max },
      },
      elements: { point: { radius: 0 }, line: { tension: 0.3 } }
    }
  })
}

async function runSim() {
  if (!Object.keys(conditions).length) {
    alert('Select at least one condition.')
    return
  }

  document.getElementById('empty').style.display   = 'none'
  document.getElementById('results').style.display = 'none'
  document.getElementById('loading').style.display = 'block'

  const res = await fetch('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      resident_id:     document.getElementById('res-id').value || 'R001',
      conditions,
      simulation_days: parseInt(document.getElementById('days').value),
    }),
  })

  const data = await res.json()
  globalDays = data.days

  // Badge
  document.getElementById('b-id').textContent   = data.resident_id
  document.getElementById('b-days').textContent = `${data.simulation_days} days`
  const tags = document.getElementById('b-tags')
  tags.innerHTML = ''
  for (const [c, sev] of Object.entries(data.conditions)) {
    const t = document.createElement('span')
    t.className = 'tag'
    t.style.background = COND_COLORS[c] + '22'
    t.style.color       = COND_COLORS[c]
    t.textContent       = `${COND_LABELS[c]} - ${sev}`
    tags.appendChild(t)
  }

  // Stats
  const days = data.days
  const acute  = days.filter(d => d.ground_truth_label !== 'normal').length
  const sleep  = days.reduce((a, b) => a + b.sleep_window, 0) / days.length
  const active = days.reduce((a, b) => a + b.active_duration, 0) / days.length
  const bath   = days.reduce((a, b) => a + b.night_bathroom_visits, 0) / days.length
  document.getElementById('s-events').textContent = acute
  document.getElementById('s-sleep').textContent  = sleep.toFixed(1)
  document.getElementById('s-active').textContent = Math.round(active)
  document.getElementById('s-bath').textContent   = bath.toFixed(1)

  // Charts
  const traj = data.trajectories
  const nDays = data.simulation_days
  const thresh = Array(nDays).fill(0.3)

  const latentGroups = new Map()
  for (const [cond, sev] of Object.entries(data.conditions)) {
    const stateKey = CONDITION_TO_STATE[cond]
    if (!stateKey) continue
    if (!latentGroups.has(stateKey)) {
      latentGroups.set(stateKey, { color: COND_COLORS[cond], labels: [] })
    }
    latentGroups.get(stateKey).labels.push(COND_LABELS[cond])
  }

  const latentDatasets = Array.from(latentGroups.entries()).map(([stateKey, info]) => ({
    label: info.labels.length > 1
      ? `${LATENT_STATE_LABELS[stateKey]} · ${info.labels.join(', ')}`
      : `${LATENT_STATE_LABELS[stateKey]} · ${info.labels[0]}`,
    data: traj[stateKey],
    borderColor: info.color,
    borderWidth: 2,
  }))

  makeChart('c-traj', [
    ...latentDatasets,
    { label: 'Threshold (0.3)',      data: thresh, borderColor: '#d1d5db', borderDash: [4,4], borderWidth: 1 },
  ], 0, 1)

  makeChart('c-active', [{ data: days.map(d => d.active_duration),         borderColor: '#059669', borderWidth: 2, fill: true, backgroundColor: 'rgba(5,150,105,.08)' }])
  makeChart('c-bath',   [{ data: days.map(d => d.night_bathroom_visits),   borderColor: '#7c3aed', borderWidth: 2, fill: true, backgroundColor: 'rgba(124,58,237,.08)' }])
  makeChart('c-br',     [{ data: days.map(d => d.breathing_rate_mean),     borderColor: '#2563eb', borderWidth: 2, fill: true, backgroundColor: 'rgba(37,99,235,.08)' }])
  makeChart('c-frag',   [{ data: days.map(d => d.sleep_fragmentation_idx), borderColor: '#dc2626', borderWidth: 2, fill: true, backgroundColor: 'rgba(220,38,38,.08)' }], 0, 1)

  // Table
  const headRow = document.getElementById('table-head-row')
  headRow.innerHTML = DAILY_SUMMARY_COLUMNS.map(col => `<th>${col.label}</th>`).join('')

  const tbody = document.getElementById('tbody')
  tbody.innerHTML = ''
  days.slice(0, 30).forEach(d => {
    const tr = document.createElement('tr')
    tr.innerHTML = DAILY_SUMMARY_COLUMNS.map(col => {
      const value = col.format ? col.format(d[col.key]) : d[col.key]
      const className = col.key === 'ground_truth_label' ? d.ground_truth_label : ''
      return `<td class="${className}">${value}</td>`
    }).join('')
    tbody.appendChild(tr)
  })

  document.getElementById('loading').style.display  = 'none'
  document.getElementById('results').style.display  = 'flex'
}

function downloadCSV() {
  if (!globalDays.length) return
  const exclude = new Set(['M_severity','CP_severity','SDB_severity','ground_truth_label'])
  const keys = Object.keys(globalDays[0]).filter(k => !exclude.has(k))
  const csv  = [keys.join(','), ...globalDays.map(r => keys.map(k => r[k]).join(','))].join('\n')
  const a = document.createElement('a')
  a.href     = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv)
  a.download = 'synthetic_resident.csv'
  a.click()
}
