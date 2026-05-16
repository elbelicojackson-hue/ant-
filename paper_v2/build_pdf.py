"""
build_pdf.py — 用 reportlab 生成 NCP 数学规范 PDF
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "NCP-Mathematical-Foundations.pdf")

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                        leftMargin=2.5*cm, rightMargin=2.5*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontSize=18, spaceAfter=6))
styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=12,
                          alignment=TA_CENTER, textColor=colors.grey, spaceAfter=20))
styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'], fontSize=16, spaceBefore=20))
styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=13, spaceBefore=14))
styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'], fontSize=11, spaceBefore=10))
styles.add(ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=10,
                          leading=14, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='Formula', parent=styles['Normal'], fontSize=10,
                          leading=14, leftIndent=30, fontName='Courier', spaceAfter=8, spaceBefore=8))
styles.add(ParagraphStyle(name='Def', parent=styles['Normal'], fontSize=10,
                          leading=14, leftIndent=20, borderWidth=1, borderColor=colors.lightgrey,
                          borderPadding=8, backColor=colors.Color(0.97, 0.97, 1.0)))

story = []

# Title
story.append(Paragraph("Neural Consensus Protocol (NCP)", styles['Title2']))
story.append(Paragraph("Mathematical Foundations and Algorithm Specification", styles['Subtitle']))
story.append(Paragraph("Version 1.0 — May 2026 | NCP Working Group", styles['Subtitle']))
story.append(Spacer(1, 20))

# ═══════════════════════════════════════════════════════════════
# Section 1
story.append(Paragraph("1. Formal Definitions", styles['H1']))

story.append(Paragraph("<b>Definition 1 (Reasoning Chain).</b> A reasoning chain C is a tuple (id, a, v, S, σ) where: "
    "id ∈ H is a unique identifier, a ∈ A is the owning agent, v ∈ N⁺ is the version, "
    "S = (s₁, s₂, ..., sₖ) is an ordered sequence of steps, "
    "σ ∈ {active, broken, archived} is the chain status.", styles['Def']))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Definition 2 (Step).</b> A step sᵢ is a tuple (i, cᵢ, τᵢ, nᵢ) where: "
    "i ∈ N⁺ is the index, cᵢ is the assertion content, "
    "τᵢ ∈ {active, collapsed, fortified} is the status, nᵢ ∈ N is the attack count.", styles['Def']))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Definition 3 (Attack).</b> An attack X is a tuple (a_src, C_tgt, i_tgt, r, ω) where: "
    "a_src is the attacker, C_tgt is the target chain, i_tgt is the target step, "
    "r is the reason, ω ∈ {0,1} is the outcome.", styles['Def']))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Definition 4 (Address).</b> Every step has a unique address:", styles['Body']))
story.append(Paragraph("addr(s) = agent.v{version}.Step{index}    e.g., GPT-5.4.v2.Step3", styles['Formula']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("2. Entropy Calculations", styles['H1']))

story.append(Paragraph("2.1 10-Dimensional Entropy Vector", styles['H2']))
story.append(Paragraph("The system state is characterized by:", styles['Body']))
story.append(Paragraph("H(t) = (H_sem, H_cau, H_bnd, H_tmp, H_dep, H_div, H_inf, H_prp, H_evi, H_imp)", styles['Formula']))
story.append(Paragraph("where each Hᵢ ∈ [0, 1] measures a distinct type of uncertainty.", styles['Body']))

story.append(Paragraph("2.2 Divergence Entropy (Stance-Based)", styles['H2']))
story.append(Paragraph("For each active chain, define a stance vector v_a ∈ {-1, 0, +1}^d:", styles['Body']))
story.append(Paragraph("v_{a,j} = +1 if positive signals dominate, -1 if negative, 0 otherwise", styles['Formula']))
story.append(Paragraph("Stance agreement between agents a, b:", styles['Body']))
story.append(Paragraph("agree(a,b) = (1/&#124;D&#124;) × Σ_{j∈D} match(v_{a,j}, v_{b,j})", styles['Formula']))
story.append(Paragraph("where match = 1 if equal, 0.5 if one is 0, 0 otherwise.", styles['Body']))
story.append(Paragraph("Divergence entropy:", styles['Body']))
story.append(Paragraph("H_div = 1 - (2 / &#124;A&#124;(&#124;A&#124;-1)) × Σ_{a&lt;b} agree(a,b)", styles['Formula']))

story.append(Paragraph("2.3 Total Entropy with Impact Amplification", styles['H2']))
story.append(Paragraph("<b>Theorem 1.</b> The total entropy is amplified by impact:", styles['Body']))
story.append(Paragraph("H_total = min(1, H_mean × (1 + 0.5 × H_imp))", styles['Formula']))
story.append(Paragraph("Amplification range: 1.0× (no impact) to 1.5× (max impact).", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("3. UCB1 Attack Scheduling", styles['H1']))

story.append(Paragraph("3.1 Multi-Armed Bandit Formulation", styles['H2']))
story.append(Paragraph("<b>Definition 5 (UCB1 Score).</b> For step sᵢ:", styles['Body']))
story.append(Paragraph("UCB(sᵢ) = cᵢ/nᵢ + √2 × √(ln(N)/nᵢ) + 0.1 × ln(1 + Δtᵢ)", styles['Formula']))
story.append(Paragraph("where cᵢ = collapse count, nᵢ = attack count, N = total attacks, Δtᵢ = rounds since last attack.", styles['Body']))

story.append(Paragraph("<b>Property 1 (Exploration Guarantee).</b> If nᵢ = 0, then UCB(sᵢ) = +∞. "
    "Every step is attacked at least once before any step is attacked twice.", styles['Def']))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Theorem 2 (Coverage Bound).</b> With &#124;A&#124; agents and &#124;S&#124; total steps, after r rounds:", styles['Body']))
story.append(Paragraph("coverage(r) ≥ min(1, r × &#124;A&#124; / &#124;S&#124;)", styles['Formula']))
story.append(Paragraph("With 30 agents and 210 steps: full coverage in ⌈210/30⌉ = 7 rounds.", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("4. Decentralized Parallel Voting", styles['H1']))

story.append(Paragraph("<b>Definition 6.</b> Voting set: V = A \\ {a_src} (all except attacker)", styles['Body']))
story.append(Paragraph("Each voter independently: vote(v) ∈ {collapsed, defended}", styles['Formula']))
story.append(Paragraph("<b>Majority Decision:</b>", styles['Body']))
story.append(Paragraph("ω = 1 (collapsed) if |{v: vote(v)=collapsed}| > |{v: vote(v)=defended}|, else 0", styles['Formula']))

story.append(Paragraph("<b>Property 2 (No Privileged Node).</b> No judge, no arbiter. Attacker cannot vote.", styles['Def']))
story.append(Spacer(1, 8))
story.append(Paragraph("<b>Property 3 (Parallel Execution).</b> latency(voting) = max_{v∈V} latency(v) = O(1) w.r.t. &#124;V&#124;", styles['Def']))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Theorem 3 (Byzantine Fault Tolerance).</b> With n voters, tolerates ⌊(n-1)/2⌋ Byzantine nodes.", styles['Body']))
story.append(Paragraph("Proof: Byzantine voter contributes at most 1 wrong vote. Honest majority n-f > n/2 when f < n/2. □", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("5. Convergence Theory", styles['H1']))

story.append(Paragraph("<b>Definition 7 (Verified Convergence).</b> System converges when:", styles['Body']))
story.append(Paragraph("coverage(t) > θ_c  AND  verified_ratio(t) > θ_v  AND  t ≥ t_min", styles['Formula']))
story.append(Paragraph("Default: θ_c = 0.5, θ_v = 0.4, t_min = 2", styles['Body']))

story.append(Paragraph("<b>Theorem 4 (Convergence Speed).</b> Expected rounds:", styles['Body']))
story.append(Paragraph("E[T] ≤ ⌈θ_c × &#124;S&#124; / &#124;A&#124;⌉ + 1", styles['Formula']))
story.append(Paragraph("<b>Corollary:</b> 30 agents, 49 steps, θ_c=0.5: E[T] ≤ 2 rounds.", styles['Body']))
story.append(Paragraph("This proves: <b>more agents → faster convergence</b>.", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("6. ELO Reputation System", styles['H1']))

story.append(Paragraph("After attack between attacker a and defender d:", styles['Body']))
story.append(Paragraph("E_a = 1 / (1 + 10^((R_d - R_a)/400))", styles['Formula']))
story.append(Paragraph("R_a' = R_a + K × (ω - E_a)", styles['Formula']))
story.append(Paragraph("R_d' = R_d + K × ((1-ω) - E_d)", styles['Formula']))
story.append(Paragraph("K = 32, ω = 1 if attack succeeds, 0 otherwise.", styles['Body']))
story.append(Spacer(1, 8))
story.append(Paragraph("Vote weight mapping:", styles['Body']))
story.append(Paragraph("w(a) = clip((R_a - 800)/400 + 0.5, 0.5, 2.0)", styles['Formula']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("7. Consensus Persistence with Decay", styles['H1']))

story.append(Paragraph("Confidence decay function:", styles['Body']))
story.append(Paragraph("φ(t) = max(φ_floor, 1.0 - λ(t-t₀) + μ × n_ref(t))", styles['Formula']))
story.append(Paragraph("λ = 0.005/day, φ_floor = 0.3, μ = 0.02/reference", styles['Body']))
story.append(Spacer(1, 8))
story.append(Paragraph("Challenge penalty: φ' = φ - 0.15", styles['Formula']))
story.append(Paragraph("Revocation propagation: if φ' ≤ 0.1, all dependents lose 0.2", styles['Body']))

story.append(Paragraph("7.1 Cross-Session α Calculation", styles['H2']))
story.append(Paragraph("α(c) = Σ_{a∈A} w(a) × 1[c ∈ C_a] / Σ_{a∈A} w(a)", styles['Formula']))
story.append(Paragraph("Convergence threshold: α ≥ 0.65 → TRUE_CONSENSUS", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("8. Protocol Invariants", styles['H1']))

invariants = [
    "I1: ∀C: v_C ≥ 1 (version is positive)",
    "I2: ∀s: n_s ≥ 0 (attack count non-negative)",
    "I3: τ_s = fortified ⟹ n_s ≥ 2 (fortification requires 2+ survived attacks)",
    "I4: σ_C = broken ⟹ ∃s ∈ C.S: τ_s = collapsed",
    "I5: REPAIR(C) ⟹ v_{C'} = v_C + 1 (repair increments version)",
    "I6: σ_C = archived ⟹ C cannot be attacked",
    "I7: vote(a_src, X) = ⊥ (attacker cannot vote on own attack)",
    "I8: H_total ∈ [0, 1] (entropy is bounded)",
    "I9: α(c) ∈ [0, 1] (consensus degree is bounded)",
    "I10: φ(t) ≥ φ_floor unless revoked",
]
for inv in invariants:
    story.append(Paragraph(inv, styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("9. Complexity Summary", styles['H1']))

data = [
    ['Component', 'Per-Round Complexity', 'Bottleneck'],
    ['Chain parsing', 'O(&#124;A&#124; × k)', 'Regex'],
    ['UCB1 scoring', 'O(&#124;S&#124;)', 'Step traversal'],
    ['Attack routing', 'O(&#124;A&#124; × &#124;S&#124;)', 'Display build'],
    ['Parallel voting', 'O(1) latency', 'Slowest model'],
    ['Entropy', 'O(&#124;A&#124; × k)', 'Stance extraction'],
    ['α extraction', 'O(1) call', 'Single model'],
    ['Consensus store', 'O(&#124;C_store&#124;)', 'Keyword match'],
]
t = Table(data, colWidths=[4*cm, 4*cm, 4*cm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.4)),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
]))
story.append(t)
story.append(Spacer(1, 12))
story.append(Paragraph("<b>True bottleneck:</b> LLM inference latency (~10-30s). All algorithmic overhead is sub-second.", styles['Body']))

# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("10. Scaling Law", styles['H1']))

data2 = [
    ['Agents', 'Attacks/Round', 'Coverage/Round', 'Expected Convergence'],
    ['3', '3', '~15%', '4-6 rounds'],
    ['7', '7', '~35%', '2-3 rounds'],
    ['15', '15', '~70%', '1-2 rounds'],
    ['30', '30', '~95%', '1 round'],
]
t2 = Table(data2, colWidths=[3*cm, 3.5*cm, 3.5*cm, 4*cm])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.2)),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 1.0, 0.95)]),
]))
story.append(t2)
story.append(Spacer(1, 12))
story.append(Paragraph("<b>The fundamental scaling law:</b> More agents → more parallel attacks → "
    "higher coverage per round → faster convergence. This is the opposite of traditional "
    "multi-agent systems.", styles['Body']))

# Build
doc.build(story)
print(f"PDF generated: {OUTPUT}")
