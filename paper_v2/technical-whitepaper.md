# NCP Technical Whitepaper: Architecture, Algorithms, and Design Philosophy

**Version 1.0 | May 2026**
**Neural Consensus Protocol Working Group**

---

## 1. Design Philosophy

### 1.1 The Core Principle: Algorithms Control Structure, Not Behavior

NCP's fundamental design principle distinguishes it from all existing multi-agent frameworks:

> **The protocol layer never tells an agent what to think or what to do. It only controls what each agent can see.**

This is not a philosophical preference — it's a mathematical necessity. If the protocol injects instructions ("attack X's Step 3"), the resulting behavior is no longer emergent; it's scripted. The protocol becomes a centralized controller disguised as a distributed system.

Instead, NCP achieves strategic behavior through **information routing**:
- UCB1 scores determine which chains are displayed first (attention bias)
- Unverified steps are marked with ⚠ (information annotation, not instruction)
- Fortified steps are marked with ✓ (reducing wasted attacks)
- Evidence pool is visible to all (ammunition, not orders)

The agent's decision to attack, defend, or concede is entirely its own. The protocol only shapes the information landscape.

### 1.2 Why Not Natural Language?

Traditional multi-agent systems communicate in natural language. This is wrong for three reasons:

1. **Ambiguity**: "I disagree with your point" — which point? What specifically is wrong?
2. **Unverifiability**: "I agree" — genuine agreement or sycophancy? No way to distinguish.
3. **Non-composability**: A conclusion from Session A cannot be formally referenced in Session B.

NCP replaces natural language communication with **state transitions on addressable data structures**. The "language" of NCP is:

```
PROPOSE_CHAIN → CHALLENGE(target_address) → VOTE(collapsed/defended) → REPAIR(new_version)
```

Natural language still exists inside `step.content`, but the protocol layer doesn't parse it. It only cares about: who challenged whom, what was the vote, did the state change.

### 1.3 The Trust Model

NCP's trust model is fundamentally different from traditional AI systems:

| Traditional AI | NCP |
|---|---|
| "Trust me, I'm a large model" | "Here's the evidence chain, judge for yourself" |
| Black box → answer | Structured process → auditable conclusion |
| Single point of failure | N independent verifiers |
| No recourse when wrong | Every step can be challenged retroactively |

Trust in NCP is not binary (trust/don't trust). It's **graduated and evidence-based**:
- A claim with α=0.9, 12 attacks survived, 3 external evidence items → high trust
- A claim with α=0.6, 0 attacks, 0 evidence → low trust (fragile consensus)
- A claim that was once α=0.9 but its dependency was revoked → degraded trust

---

## 2. Protocol Architecture (7-Layer Stack)

```
┌─────────────────────────────────────────────────────────┐
│ L7: Observability Layer                                  │
│     SSE real-time events, Web UI, audit logs             │
├─────────────────────────────────────────────────────────┤
│ L6: Persistence Layer                                    │
│     Consensus store, decay, cross-session reference      │
├─────────────────────────────────────────────────────────┤
│ L5: Consensus Layer                                      │
│     α calculation, claim extraction, convergence         │
├─────────────────────────────────────────────────────────┤
│ L4: Entropy Layer                                        │
│     10-dim entropy vector, phase detection, scheduling   │
├─────────────────────────────────────────────────────────┤
│ L3: Judgment Layer                                       │
│     Decentralized parallel voting, majority decision     │
├─────────────────────────────────────────────────────────┤
│ L2: Attack Layer                                         │
│     UCB1 routing, precision targeting, collapse propagation │
├─────────────────────────────────────────────────────────┤
│ L1: Chain Layer                                          │
│     Addressable assertion chains, versioning, status     │
├─────────────────────────────────────────────────────────┤
│ L0: Adapter Layer                                        │
│     Unified API for heterogeneous models                 │
└─────────────────────────────────────────────────────────┘
```

Each layer has a single responsibility and communicates with adjacent layers through well-defined interfaces.

---

## 3. Core Algorithms

### 3.1 UCB1 Attack Scheduling

**Problem**: Without guidance, agents attack the same "obvious" target repeatedly, leaving other steps unverified.

**Solution**: Model the attack target selection as a multi-armed bandit problem.

```
UCB(step_i) = exploitation_i + C × √(ln(N) / n_i) + recency_i

where:
  exploitation_i = collapse_count_i / attack_count_i  (historical collapse rate)
  C = √2  (exploration coefficient)
  N = total attacks across all steps
  n_i = attacks on step_i
  recency_i = 0.1 × ln(1 + rounds_since_last_attack_i)
```

**Key property**: If `n_i = 0` (never attacked), UCB → ∞, guaranteeing exploration.

**Implementation**: UCB scores are NOT injected into prompts. They determine which chains are displayed first in the agent's visible context (information routing).

### 3.2 Decentralized Parallel Voting

**Problem**: Centralized arbitration (a "judge" model) creates a single point of failure and serial bottleneck.

**Solution**: All non-attacker nodes vote simultaneously.

```python
# Pseudocode
async def judge_attack(attack, all_agents):
    voters = [a for a in all_agents if a != attack.attacker]
    
    # All voters receive the same prompt simultaneously
    votes = await asyncio.gather(*[
        voter.evaluate(attack.target_step, attack.reason)
        for voter in voters
    ])
    
    # Simple majority
    collapse_votes = sum(1 for v in votes if v == "collapsed")
    defend_votes = sum(1 for v in votes if v == "defended")
    
    return collapse_votes > defend_votes
```

**Latency**: O(1) regardless of participant count (all votes are parallel).

**Scaling property**: More agents → more voters per attack → more robust decisions → faster convergence (not slower).

### 3.3 Convergence Detection

**The key insight**: Attack success is not "failure to converge" — it's "actively converging" (eliminating uncertainty).

Three convergence conditions (any one triggers termination):

```
Condition 1: VERIFIED_CONVERGENCE
  coverage > 50% AND verified_ratio > 40% AND round ≥ 2
  
  where:
    coverage = steps_attacked_at_least_once / total_steps
    verified_ratio = steps_with_attack_count > 0 / total_steps

Condition 2: CHAINS_STABLE
  no_successful_attack_rounds ≥ 2 AND round ≥ 3

Condition 3: TRUE_CONSENSUS (α-based)
  consensus_alpha ≥ 0.65 AND round ≥ min_rounds
  
  where:
    α = Σ(weight_i × agree_i) / Σ(weight_i)
    (computed by claim extraction across all active chains)
```

**Why this is correct**: With 30 agents, one round of attacks covers 30 steps simultaneously. If coverage > 50% after round 1, the system can converge in round 2. More agents → faster coverage → faster convergence.

### 3.4 ELO-Based Dynamic Reputation

```
After each attack:
  expected_attacker = 1 / (1 + 10^((R_defender - R_attacker) / 400))
  
  if attack_success:
    R_attacker += K × (1 - expected_attacker)
    R_defender += K × (0 - (1 - expected_attacker))
  else:
    R_attacker += K × (0 - expected_attacker)
    R_defender += K × (1 - (1 - expected_attacker))

Vote weight = f(R) ∈ [0.5, 2.0]
```

**Effect**: Agents whose chains are never broken gain higher vote weight. Agents that repeatedly collapse lose influence. This is emergent reputation, not assigned authority.

### 3.5 Consensus Persistence with Decay

```
confidence(t) = max(FLOOR, 1.0 - DECAY_RATE × age_days + REFERENCE_BOOST × ref_count)

where:
  DECAY_RATE = 0.005/day
  FLOOR = 0.3
  REFERENCE_BOOST = 0.02/reference
  CHALLENGE_PENALTY = 0.15/successful_challenge
```

**Properties**:
- Unused claims decay to 0.3 in ~140 days
- Frequently referenced claims stay near 1.0 indefinitely
- Successfully challenged claims drop rapidly (can reach 0 → revoked)
- Revocation propagates to dependent claims

### 3.6 External Truth Anchoring

```
Tool Registry:
  web_search    → Firecrawl search + DeepSeek summarization → GroundedFact
  paper_verify  → Firecrawl site:pubmed search → GroundedFact
  firecrawl_scrape → Direct URL content extraction → GroundedFact

GroundedFact properties:
  - Cannot be overturned by CHALLENGE alone
  - Requires counter-evidence from another external tool
  - Has source_urls for human verification
  - Injected into all agents' visible context as "★ verified external fact"
```

**Why this matters**: This is the solution to "optimizing consensus ≠ optimizing truth". External facts provide an anchor that internal agreement cannot override.

---

## 4. The 10-Dimensional Entropy Vector

Each dimension captures a different type of uncertainty:

| Dimension | What it measures | Drives what action |
|---|---|---|
| semantic | Inter-model understanding consistency | — |
| causal | Reasoning path uniqueness | Need stronger model |
| boundary | Conclusion scope clarity | — |
| temporal | Time sensitivity | Need verification |
| dependency | Unverified assumption ratio | Need more attacks |
| divergence | Inter-model disagreement | Continue debate |
| information | Missing key information | Trigger search |
| propagation | Error spread range | Urgent repair |
| evidence | Evidence reliability | Trigger external tool |
| impact | Attack shock magnitude | Amplifies all others |

**The amplification property**:
```
total_entropy = base_mean × (1 + impact × 0.5)
```

When impact is high (attacks are causing large shifts), all uncertainty is amplified. This prevents premature convergence during active restructuring.

---

## 5. State Machine

### 5.1 Chain States

```
                    PROPOSE_CHAIN
                         │
                         ▼
                    ┌─────────┐
                    │  ACTIVE  │◄──── REPAIR (version+1)
                    └────┬────┘           ▲
                         │                │
                    CHALLENGE              │
                         │                │
                         ▼                │
                    ┌─────────┐           │
                    │  VOTED  │           │
                    └────┬────┘           │
                    ┌────┴────┐           │
                    │         │           │
              majority      majority      │
              collapse      defend        │
                    │         │           │
                    ▼         ▼           │
              ┌─────────┐  (step.attacked_count++)
              │  BROKEN  │    │           │
              └────┬────┘    │           │
                   │         │           │
                   ▼         │           │
              ┌──────────┐   │           │
              │ ARCHIVED  │   │           │
              └──────────┘   └───────────┘
```

### 5.2 Step States

```
active → (attacked but defended) → fortified (after 2+ defenses)
active → (attacked and collapsed) → collapsed
```

### 5.3 Session States

```
idle → running → {converged | stagnated | max_rounds | user_stopped | error}
```

---

## 6. Protocol Messages (NCP v1.0)

| Message | Direction | Purpose |
|---|---|---|
| PROPOSE_CHAIN | Agent → Network | Submit a new reasoning chain |
| CHALLENGE | Agent → Target | Attack a specific step |
| VOTE | Agent → Network | Vote on an attack's validity |
| REPAIR | Agent → Network | Submit repaired chain (version+1) |
| VERIFY | System → Tool | Request external fact-checking |
| EVIDENCE | Tool → Network | Submit verified external fact |
| CONVERGE | System → Network | Declare consensus reached |
| JOIN/LEAVE | Agent → Network | Dynamic participation |

**Transport-layer independence**: These messages can be serialized as JSON and transmitted over HTTP, WebSocket, NATS, Kafka, or libp2p. The protocol doesn't care about the transport.

---

## 7. Comparison with Existing Protocols

| | MCP (Anthropic) | A2A (Google) | NCP (Ours) |
|---|---|---|---|
| Purpose | Tool calling | Task delegation | Cognitive consensus |
| Communication unit | Function call | Task/Artifact | Addressable assertion |
| Interaction pattern | Request-Response | Delegate-Report | Challenge-Vote-Repair |
| Consensus mechanism | None | None | α-weighted majority |
| State management | None | Task status | Chain version state machine |
| Trust model | Trust the tool | Trust the delegate | Verify through adversarial testing |
| Auditability | Call logs | Task logs | Complete evidence chains |
| Composability | None | None | Cross-session persistence + decay |

**NCP complements MCP and A2A** — it doesn't replace them:
- Agent uses MCP to call tools (search, compute)
- Agent uses A2A to delegate subtasks
- Agents use NCP to reach consensus on reasoning conclusions

---

## 8. Known Limitations and Mitigations

### 8.1 Consensus ≠ Truth

**Problem**: The protocol optimizes for agreement, not correctness.

**Mitigation**: External truth anchoring. GroundedFacts from Firecrawl/DeepSeek provide an external reference that internal consensus cannot override. The system explicitly distinguishes "internally agreed" from "externally verified".

### 8.2 Training Data Homogeneity

**Problem**: All models share similar training data → shared blind spots → false consensus on shared errors.

**Mitigation**: Information-routing heterogeneity. Different agents see different subsets of the evidence pool, producing structurally different reasoning even from similar knowledge bases.

### 8.3 Game-Theoretic Incentive Misalignment

**Problem**: Under certain conditions, "always attack" or "always concede" may be Nash equilibria.

**Mitigation**: 
- Decentralized voting removes the attacker's automatic vote (attacks must convince others)
- ELO reputation penalizes frivolous attacks (failed attacks lower reputation)
- NO_ATTACK is a valid and credited action (verification contribution)

### 8.4 Prompt-Induced Attack Pressure

**Problem**: The attack prompt creates pressure to always find something to attack, even when no real flaw exists.

**Mitigation**: The prompt asks for VALID/WEAK/INVALID evaluation of each step, not "find something to attack". NO_ATTACK with reasoning is a first-class output.

---

## 9. Scaling Properties

| Agents | Attacks/Round | Coverage/Round | Expected Convergence |
|---|---|---|---|
| 3 | 3 | ~15% | 4-6 rounds |
| 7 | 7 | ~35% | 2-3 rounds |
| 15 | 15 | ~70% | 1-2 rounds |
| 30 | 30 | ~95% | 1 round |

**The scaling law**: More agents → more parallel attacks → higher coverage per round → faster convergence. This is the opposite of traditional multi-agent systems where more agents → more rounds of discussion → slower convergence.

**Latency scaling**: Because voting is parallel (`asyncio.gather`), adding more agents adds zero latency to the judgment phase. The bottleneck is always the slowest single model's response time, not the number of models.

---

## 10. Implementation Reference

```
neuralcomm/
├── core/
│   ├── arena.py              # Chain-Centric engine (main loop)
│   ├── strategy.py           # UCB1 scheduler + ELO reputation + evidence pool
│   ├── consensus_store.py    # Persistence + decay + cross-session
│   ├── tools.py              # External tool registry (Firecrawl + DeepSeek)
│   ├── entropy.py            # 10-dim entropy vector
│   ├── adapter.py            # Unified model API (7+ providers)
│   └── state_machine.py      # State tracking
├── protocol/
│   └── NCP-SPEC-v1.md        # Formal protocol specification
├── web/
│   ├── server.py             # FastAPI + SSE real-time backend
│   └── static/               # Observability UI
├── benchmark/
│   └── truthfulqa_test.py    # Accuracy benchmark
└── werewolf/
    └── game.py               # Social deduction game (protocol validation)
```

---

## 11. Future Directions

1. **Standardized Benchmark**: TruthfulQA / MMLU hard subset comparison (single model vs NCP)
2. **Transport Layer Implementations**: HTTP binding, WebSocket binding, libp2p binding
3. **Agent Discovery**: DNS-SD / DHT-based agent discovery for open networks
4. **Formal Verification**: TLA+ specification of protocol invariants
5. **Token Efficiency**: CC Engine architecture (diff-only transmission, 66% reduction)

---

*NCP Technical Whitepaper v1.0 — May 2026*
*https://github.com/elbelicojackson-hue/ant-*
