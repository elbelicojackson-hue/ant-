/* ═══════════════════════════════════════════════════════════════
   NeuralComm Live — SSE Event Handler & Real-Time Visualization
   Pure vanilla JS, no frameworks, no dependencies
   ═══════════════════════════════════════════════════════════════ */

// ── Global State ──────────────────────────────────────────────
const STATE = {
  sessionId: '',
  question: '',
  status: 'idle',        // idle | running | finished | stopped | error
  models: [],
  round: 0,
  roundPhase: '',
  elapsed: 0,
  elapsedTimer: null,
  sessionStart: 0,

  // Chains: keyed by agent_id
  chains: {},           // { agent_id: { chain_id, version, status, steps: [...] } }

  // Attacks: ordered list
  attacks: [],          // [{ round, attacker, target_agent, target_step, success, reason, votes }]

  // Entropy: latest 10-dim vector
  entropy: {
    semantic: 0, causal: 0, boundary: 0, temporal: 0,
    dependency: 0, divergence: 0, information: 0,
    propagation: 0, evidence: 0, impact: 0, total: 0,
  },

  // Round stats
  stats: { attacks: 0, successful: 0, active_chains: 0, broken_chains: 0 },

  // Consensus
  consensus: { alpha: null, fragile: null },

  // Convergence
  convergence: { can_converge: false, reason: '' },

  // Verification records
  verifications: [],    // [{ round, reason, content }]

  // Final results
  consensusClaims: [],
  disputes: [],
};

// ── DOM Refs ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const DOM = {
  statusBadge:   $('status-badge'),
  elapsed:       $('elapsed'),
  sessionLabel:  $('session-label'),
  questionInput: $('question-input'),
  runBtn:        $('run-btn'),
  stopBtn:       $('stop-btn'),

  statRound:     $('stat-round'),
  statAttacks:   $('stat-attacks'),
  statActive:    $('stat-active'),
  statBroken:    $('stat-broken'),
  statAlpha:     $('stat-alpha'),

  chainsContainer: $('chains-container'),
  chainsCount:     $('chains-count'),

  attacksContainer: $('attacks-container'),
  attacksCount:     $('attacks-count'),

  verifyContainer: $('verify-container'),

  convergenceBlock: $('convergence-block'),
  convergenceText:  $('convergence-text'),
  consensusClaims:  $('consensus-claims'),

  logContainer: $('log-container'),
};

// ── Initialization ────────────────────────────────────────────
function init() {
  DOM.runBtn.addEventListener('click', startSession);
  DOM.stopBtn.addEventListener('click', stopSession);
  $('clear-log-btn').addEventListener('click', clearLog);
  DOM.questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startSession();
  });
}

// ── Session Control ───────────────────────────────────────────
async function startSession() {
  const question = DOM.questionInput.value.trim();
  if (!question) return;

  setRunning(true);
  resetState();
  STATE.question = question;
  addLog('info', `Starting: "${question.slice(0, 60)}..."`);

  try {
    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, max_rounds: 50, convergence_threshold: 0.25, verifier: 'auto' }),
    });
    const data = await resp.json();
    STATE.sessionId = data.session_id;
    DOM.sessionLabel.textContent = `Session ${STATE.sessionId}`;
    addLog('info', `Session: ${STATE.sessionId}`);
    connectSSE(STATE.sessionId);
  } catch (err) {
    addLog('error', `Failed to start: ${err.message}`);
    setRunning(false);
  }
}

async function stopSession() {
  if (!STATE.sessionId) return;
  addLog('info', 'Stopping...');
  try {
    await fetch(`/api/stop/${STATE.sessionId}`, { method: 'POST' });
  } catch (err) {
    addLog('error', `Stop failed: ${err.message}`);
  }
}

function setRunning(running) {
  if (running) {
    STATE.status = 'running';
    DOM.statusBadge.setAttribute('data-status', 'running');
    DOM.statusBadge.textContent = 'Running';
    DOM.runBtn.disabled = true;
    DOM.stopBtn.disabled = false;
    DOM.questionInput.disabled = true;
    STATE.sessionStart = Date.now();
    STATE.elapsedTimer = setInterval(updateElapsed, 200);
  } else {
    STATE.status = 'idle';
    DOM.statusBadge.setAttribute('data-status', 'idle');
    DOM.statusBadge.textContent = 'Idle';
    DOM.runBtn.disabled = false;
    DOM.stopBtn.disabled = true;
    DOM.questionInput.disabled = false;
    if (STATE.elapsedTimer) { clearInterval(STATE.elapsedTimer); STATE.elapsedTimer = null; }
  }
}

function updateElapsed() {
  if (STATE.sessionStart) {
    STATE.elapsed = ((Date.now() - STATE.sessionStart) / 1000).toFixed(1);
    DOM.elapsed.textContent = STATE.elapsed + 's';
  }
}

function resetState() {
  STATE.round = 0;
  STATE.chains = {};
  STATE.attacks = [];
  STATE.verifications = [];
  STATE.entropy = {
    semantic: 0, causal: 0, boundary: 0, temporal: 0,
    dependency: 0, divergence: 0, information: 0,
    propagation: 0, evidence: 0, impact: 0, total: 0,
  };
  STATE.stats = { attacks: 0, successful: 0, active_chains: 0, broken_chains: 0 };
  STATE.consensus = { alpha: null, fragile: null };
  STATE.convergence = { can_converge: false, reason: '' };
  STATE.consensusClaims = [];
  STATE.disputes = [];

  DOM.chainsContainer.innerHTML = '<div class="empty-hint">等待模型构建推理链...</div>';
  DOM.chainsCount.textContent = '0 active';
  DOM.attacksContainer.innerHTML = '<div class="empty-hint">等待攻击事件...</div>';
  DOM.attacksCount.textContent = '0 attacks';
  DOM.verifyContainer.innerHTML = '<div class="empty-hint dim">DeepSeek 查证记录将显示在这里</div>';
  DOM.consensusClaims.innerHTML = '<div class="empty-hint dim">运行完成后显示...</div>';
  DOM.convergenceText.textContent = '等待...';
  $('alpha-value').textContent = '—';
  $('fragile-value').textContent = '—';
  resetEntropyDisplay();
  resetStats();
  DOM.logContainer.innerHTML = '<div class="empty-hint dim">事件日志将实时显示在这里...</div>';
}

// ── SSE Connection ────────────────────────────────────────────
function connectSSE(sessionId) {
  const url = `/api/stream/${sessionId}`;
  addLog('info', `SSE connecting: ${url}`);

  const es = new EventSource(url);

  es.addEventListener('session_started', (e) => {
    const d = JSON.parse(e.data);
    STATE.models = d.models || [];
    STATE.question = d.question || STATE.question;
    DOM.sessionLabel.textContent = `Session ${d.session_id}`;
    addLog('session', `${d.models.length} models | mode: ${d.mode}`);
  });

  es.addEventListener('round_started', (e) => {
    const d = JSON.parse(e.data);
    STATE.round = d.round;
    STATE.roundPhase = d.phase || '';
    addLog('round', `R${d.round}${d.phase ? ' — ' + d.phase : ''}`);
  });

  es.addEventListener('model_responded', (e) => {
    const d = JSON.parse(e.data);
    addLog('model', `${d.agent_id} → ${d.content_length} chars (R${d.round})`);
  });

  es.addEventListener('chain_built', (e) => {
    const d = JSON.parse(e.data);
    STATE.chains[d.agent_id] = {
      chain_id: d.chain_id,
      agent_id: d.agent_id,
      version: d.version,
      status: 'active',
      steps: d.steps.map(s => ({
        index: s.index, content: s.content, status: s.status, attacked_count: 0,
      })),
    };
    addLog('chain', `${d.agent_id} v${d.version}: ${d.steps.length} steps`);
    renderChains();
  });

  es.addEventListener('chain_repaired', (e) => {
    const d = JSON.parse(e.data);
    if (STATE.chains[d.agent_id]) {
      STATE.chains[d.agent_id].version = d.new_version;
      STATE.chains[d.agent_id].steps = d.new_steps.map(s => ({
        index: s.index, content: s.content, status: s.status, attacked_count: 0,
      }));
      STATE.chains[d.agent_id].status = 'active';
    }
    addLog('chain', `${d.agent_id} v${d.old_version}→v${d.new_version}`);
    renderChains();
  });

  es.addEventListener('attack_started', (e) => {
    const d = JSON.parse(e.data);
    addLog('attack', `⚔ ${d.attacker_id} → ${d.target_agent}.Step${d.target_step}: ${(d.reason || '').substring(0, 60)}`);
  });

  es.addEventListener('vote_started', (e) => {
    const d = JSON.parse(e.data);
    addLog('vote', `🗳 投票: ${d.phase} — ${d.defender || d.judge || ''}`);
  });

  es.addEventListener('vote_cast', (e) => {
    const d = JSON.parse(e.data);
    const icon = d.verdict === 'collapsed' ? '✗' : '✓';
    addLog('vote', `  ${icon} ${d.voter} (${d.role}): ${d.verdict}`);
  });

  es.addEventListener('attack_result', (e) => {
    const d = JSON.parse(e.data);
    STATE.attacks.push({
      round: d.round,
      attacker: d.attacker_id,
      target_agent: d.target_agent,
      target_step: d.target_step,
      success: d.success,
      reason: d.reason || '',
      votes: d.votes || '',
    });

    const chain = STATE.chains[d.target_agent];
    if (chain) {
      const step = chain.steps.find(s => s.index === d.target_step);
      if (step) {
        step.attacked_count = (step.attacked_count || 0) + 1;
        if (d.success) {
          step.status = 'collapsed';
        } else if (step.attacked_count >= 2) {
          step.status = 'fortified';
        }
      }
    }

    const verb = d.success ? '✗ COLLAPSED' : '✓ DEFENDED';
    addLog('judge', `${d.attacker_id}→${d.target_agent}.Step${d.target_step}: ${verb} (${d.votes})`);
    renderAttacks();
    renderChains();
  });

  es.addEventListener('entropy_update', (e) => {
    const d = JSON.parse(e.data);
    STATE.entropy = d.entropy;
    renderEntropy();
  });

  es.addEventListener('round_summary', (e) => {
    const d = JSON.parse(e.data);
    STATE.stats.attacks = d.attacks_count;
    STATE.stats.successful = d.successful_count;
    STATE.stats.active_chains = d.active_chains;
    STATE.stats.broken_chains = d.broken_chains;
    renderStats();
  });

  es.addEventListener('convergence_check', (e) => {
    const d = JSON.parse(e.data);
    STATE.convergence.can_converge = d.can_converge;
    STATE.convergence.reason = d.reason || '';
    if (d.fragile_score !== undefined) {
      STATE.consensus.fragile = d.fragile_score;
      $('fragile-value').textContent = d.fragile_score.toFixed(3);
    }
    DOM.convergenceText.textContent = `${d.can_converge ? '✓' : '✗'} ${d.reason}`;
    addLog('converge', `R${d.round}: can=${d.can_converge} frag=${d.fragile_score?.toFixed(3) || '?'}`);
  });

  es.addEventListener('verification_triggered', (e) => {
    const d = JSON.parse(e.data);
    STATE.verifications.push({ round: d.round, reason: d.reason, content: d.content || '' });
    renderVerifications();
    addLog('verify', `R${d.round}: ${d.reason}`);
  });

  es.addEventListener('session_finished', (e) => {
    const d = JSON.parse(e.data);
    STATE.status = 'finished';
    DOM.statusBadge.setAttribute('data-status', 'finished');
    DOM.statusBadge.textContent = 'Finished';
    DOM.runBtn.disabled = false;
    DOM.stopBtn.disabled = true;
    DOM.questionInput.disabled = false;
    if (STATE.elapsedTimer) { clearInterval(STATE.elapsedTimer); STATE.elapsedTimer = null; }

    STATE.consensusClaims = d.consensus || [];
    STATE.disputes = d.disputes || [];

    addLog('session', `FINISHED: ${d.convergence_reason} | ${d.total_rounds}r | ${d.elapsed_seconds}s`);
    if (d.attacks_summary) {
      addLog('session', `Atk: ${d.attacks_summary.total}t ${d.attacks_summary.successful}s`);
    }
    renderConsensusClaims();
    es.close();
  });

  es.addEventListener('session_error', (e) => {
    const d = JSON.parse(e.data);
    STATE.status = 'error';
    DOM.statusBadge.setAttribute('data-status', 'error');
    DOM.statusBadge.textContent = 'Error';
    DOM.runBtn.disabled = false;
    DOM.stopBtn.disabled = true;
    DOM.questionInput.disabled = false;
    if (STATE.elapsedTimer) { clearInterval(STATE.elapsedTimer); STATE.elapsedTimer = null; }
    addLog('error', `SESSION ERROR: ${d.error}`);
    es.close();
  });

  es.addEventListener('heartbeat', () => { /* keepalive */ });

  es.onerror = () => {
    if (STATE.status === 'running') {
      addLog('warn', 'SSE disconnected, retrying...');
    }
  };
}

// ── Renderers ─────────────────────────────────────────────────

function renderChains() {
  const entries = Object.entries(STATE.chains);
  if (entries.length === 0) {
    DOM.chainsContainer.innerHTML = '<div class="empty-hint">等待模型构建推理链...</div>';
    DOM.chainsCount.textContent = '0 active';
    return;
  }

  const active = entries.filter(([_, c]) => c.status === 'active');
  DOM.chainsCount.textContent = `${active.length} active`;

  DOM.chainsContainer.innerHTML = entries.map(([agentId, chain]) => {
    const cardClass = chain.status === 'broken' ? 'chain-card broken' : 'chain-card';
    const statusIconMap = { active: '●', broken: '✗', archived: '○' };
    const statusIcon = statusIconMap[chain.status] || '?';

    const stepsHTML = (chain.steps || []).map(s => {
      const iconMap = { active: '○', collapsed: '✗', fortified: '◆' };
      const icon = iconMap[s.status] || '?';
      const iconClass = `step-icon ${s.status}`;
      const meta = (s.attacked_count || 0) > 0 ? `(${s.attacked_count})` : '';
      return `
        <div class="chain-step">
          <span class="${iconClass}">${icon}</span>
          <span class="step-index">S${s.index}</span>
          <span class="step-content">${escHtml(s.content.slice(0, 80))}</span>
          <span class="step-meta">${meta}</span>
        </div>`;
    }).join('');

    return `
      <div class="${cardClass}">
        <div class="chain-hdr">
          <span class="chain-agent">${escHtml(agentId)}</span>
          <span class="chain-version">v${chain.version}</span>
          <span class="chain-status ${chain.status}">${statusIcon} ${chain.status}</span>
        </div>
        ${stepsHTML}
      </div>`;
  }).join('');
}

function renderAttacks() {
  if (STATE.attacks.length === 0) {
    DOM.attacksContainer.innerHTML = '<div class="empty-hint">等待攻击事件...</div>';
    DOM.attacksCount.textContent = '0 attacks';
    return;
  }

  DOM.attacksCount.textContent = `${STATE.attacks.length} attacks`;

  DOM.attacksContainer.innerHTML = STATE.attacks.slice().reverse().map(a => {
    const cls = a.success ? 'attack-entry success' : 'attack-entry failed';
    const icon = a.success ? '✗' : '✓';
    const voteCls = a.success ? 'win' : 'lose';
    return `
      <div class="${cls}">
        <div class="attack-topline">
          <span class="attack-agents">${icon} ${escHtml(a.attacker)} → ${escHtml(a.target_agent)}.Step${a.target_step}</span>
          <span class="attack-votes ${voteCls}">${a.votes}</span>
        </div>
        <div class="attack-reason">${escHtml(a.reason.slice(0, 150))}</div>
      </div>`;
  }).join('');
}

function renderEntropy() {
  const e = STATE.entropy;
  const dims = [
    'semantic', 'causal', 'boundary', 'temporal', 'dependency',
    'divergence', 'information', 'propagation', 'evidence', 'impact',
  ];

  dims.forEach(dim => {
    const el = document.getElementById(`e-${dim}`);
    if (el) {
      const v = e[dim] ?? 0;
      el.textContent = v.toFixed(3);
      el.className = 'val';
      if (v > 0.6) el.classList.add('val-high');
      else if (v > 0.35) el.classList.add('val-mid');
      else el.classList.add('val-low');
    }
  });

  const totalEl = document.getElementById('e-total');
  if (totalEl) {
    const t = e.total ?? 0;
    totalEl.textContent = t.toFixed(3);
    totalEl.className = 'val';
    if (t > 0.5) totalEl.classList.add('val-high');
    else if (t > 0.3) totalEl.classList.add('val-mid');
    else totalEl.classList.add('val-low');
  }
}

function resetEntropyDisplay() {
  const dims = [
    'semantic', 'causal', 'boundary', 'temporal', 'dependency',
    'divergence', 'information', 'propagation', 'evidence', 'impact', 'total',
  ];
  dims.forEach(dim => {
    const el = document.getElementById(`e-${dim}`);
    if (el) { el.textContent = '—'; el.className = 'val'; }
  });
}

function renderStats() {
  const s = STATE.stats;
  DOM.statRound.textContent = STATE.round || '—';
  DOM.statAttacks.textContent = s.attacks;
  DOM.statActive.textContent = s.active_chains;
  DOM.statBroken.textContent = s.broken_chains;
}

function resetStats() {
  DOM.statRound.textContent = '—';
  DOM.statAttacks.textContent = '—';
  DOM.statActive.textContent = '—';
  DOM.statBroken.textContent = '—';
  DOM.statAlpha.textContent = '—';
}

function renderVerifications() {
  if (STATE.verifications.length === 0) {
    DOM.verifyContainer.innerHTML = '<div class="empty-hint dim">DeepSeek 查证记录将显示在这里</div>';
    return;
  }
  DOM.verifyContainer.innerHTML = STATE.verifications.map(v => `
    <div class="verify-entry">
      [R${v.round}] ${escHtml(v.reason)}
      ${v.content ? '<br><span style="opacity:0.7">' + escHtml(v.content.slice(0, 200)) + '</span>' : ''}
    </div>
  `).join('');
}

function renderConsensusClaims() {
  if (STATE.consensusClaims.length === 0 && STATE.disputes.length === 0) {
    DOM.consensusClaims.innerHTML = '<div class="empty-hint dim">运行完成后显示...</div>';
    return;
  }
  let html = '';
  if (STATE.consensusClaims.length > 0) {
    html += STATE.consensusClaims.map(c =>
      `<div class="claim-item" style="color:var(--green)">✓ ${escHtml(c)}</div>`
    ).join('');
  }
  if (STATE.disputes.length > 0) {
    html += STATE.disputes.map(d =>
      `<div class="claim-item" style="color:var(--orange)">? ${escHtml(d)}</div>`
    ).join('');
  }
  DOM.consensusClaims.innerHTML = html;
}

// ── Event Log ─────────────────────────────────────────────────
function addLog(type, message) {
  const now = new Date();
  const ts = now.toTimeString().slice(0, 8) + '.' + String(now.getMilliseconds()).padStart(3, '0');

  const emptyHint = DOM.logContainer.querySelector('.empty-hint');
  if (emptyHint) emptyHint.remove();

  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="ts">${ts}</span> <span class="evt">[${type}]</span> <span class="txt">${escHtml(message)}</span>`;

  DOM.logContainer.appendChild(line);

  // Keep max 200 lines
  while (DOM.logContainer.children.length > 200) {
    DOM.logContainer.firstChild.remove();
  }

  // Auto-scroll log
  const scrollParent = DOM.logContainer.parentElement;
  if (scrollParent) scrollParent.scrollTop = scrollParent.scrollHeight;
}

function clearLog() {
  DOM.logContainer.innerHTML = '<div class="empty-hint dim">事件日志已清除</div>';
}

// ── Utility ───────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Boot ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
