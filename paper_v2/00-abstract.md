# NCP: Neural Consensus Protocol — A Decentralized Multi-Agent Reasoning Consensus Protocol with Auditable Evidence Chains

## Abstract

We present the Neural Consensus Protocol (NCP), a decentralized multi-agent consensus protocol that enables heterogeneous AI agents to reach verifiable agreement on complex reasoning tasks through structured adversarial verification. Unlike existing multi-agent frameworks that produce unverifiable dialogue (MCP for tool calling, A2A for task delegation), NCP defines a minimal protocol for *cognitive consensus*: how independent reasoning engines converge on shared conclusions with complete audit trails.

NCP introduces five architectural innovations: (1) **Addressable Assertion Chains** — each agent maintains a versioned reasoning chain where every step has a unique address (`agent.version.step`) and can be independently challenged; (2) **Decentralized Parallel Voting** — attack validity is determined by all non-attacker nodes voting simultaneously via `asyncio.gather`, with no privileged arbiter, achieving O(1) latency regardless of participant count; (3) **External Truth Anchoring** — a tool registry (Firecrawl search + DeepSeek verification) produces `GroundedFact` objects that cannot be overturned by internal consensus alone, requiring counter-evidence from external sources; (4) **Composable Consensus Persistence** — verified claims (α ≥ 0.65) are stored with decay functions, cross-session referencing, and dependency propagation, enabling incremental knowledge accumulation; (5) **Information-Routing Strategy** — UCB1-based attack scheduling controls what each agent *sees* rather than what it *does*, preserving emergent behavior while maximizing verification coverage.

Building on our prior work (Aletheia, which demonstrated structured belief revision with Evidence-Adjusted Entropy), NCP evolves the architecture from a centralized coordinator model to a fully decentralized protocol suitable for open networks. We formalize the protocol as 8 message types, 4 state transition rules, and 3 convergence conditions, independent of any specific transport layer.

Experimental evaluation across three problem classes (factual misconceptions, fabricated scientific claims, and complex causal reasoning) demonstrates: (a) 7 heterogeneous models (GPT-5.4, MiMo V2.5, DeepSeek V4, Qwen-Max, Qwen3.6, Kimi-K2.6, Doubao) successfully engage in structured adversarial verification with 40+ precision attacks per session targeting specific reasoning steps; (b) the Chain-Centric architecture produces 8 chain collapses and 7 chain repairs in a single session, with attack precision reaching individual step granularity; (c) external truth anchoring via Firecrawl correctly identifies fabricated papers that all 7 models initially accepted as plausible; (d) consensus persistence enables 3-4 round reduction in subsequent sessions addressing related questions.

We identify four fundamental limitations of the current approach — optimization of consensus rather than truth, game-theoretic incentive misalignment, training data homogeneity across models, and prompt-induced attack pressure — and propose mitigation strategies including external truth anchoring (implemented), decentralized voting (implemented), information-routing without prompt injection (implemented), and standardized benchmark evaluation (in progress).

The protocol specification (NCP v1.0) is published as a transport-layer-independent standard, compatible with HTTP, WebSocket, message queues, and peer-to-peer networks, positioning NCP as the cognitive trust layer complementing MCP (tool layer) and A2A (task layer) in the emerging multi-agent ecosystem.

**Keywords:** Multi-agent consensus, decentralized verification, adversarial reasoning, protocol design, chain-centric architecture, external truth anchoring, composable knowledge

---

## 摘要

本文提出神经共识协议（NCP），一种去中心化的多智能体推理共识协议，使异构 AI 智能体能够通过结构化对抗性验证就复杂推理任务达成可验证的一致。与现有多智能体框架（MCP 用于工具调用、A2A 用于任务委派）产出不可验证的对话不同，NCP 定义了一套用于*认知共识*的最小协议：独立推理引擎如何在保留完整审计链的前提下收敛到共享结论。

NCP 引入五项架构创新：（1）**可寻址断言链**——每个智能体维护一条版本化推理链，每个步骤拥有唯一地址（`agent.version.step`）并可被独立挑战；（2）**去中心化并行投票**——攻击有效性由所有非攻击方节点通过 `asyncio.gather` 同时投票决定，无特权仲裁者，实现 O(1) 延迟；（3）**外部真值锚定**——工具注册表（Firecrawl 搜索 + DeepSeek 验证）产出 `GroundedFact` 对象，不可被内部共识单独推翻，需要外部来源的反向证据；（4）**可组合共识持久化**——已验证断言（α ≥ 0.65）带衰减函数存储，支持跨会话引用和依赖传播，实现增量知识积累；（5）**信息路由策略**——基于 UCB1 的攻击调度控制每个智能体*看到什么*而非*做什么*，在最大化验证覆盖率的同时保持行为涌现性。

基于我们的前期工作（Aletheia，展示了基于证据调整熵的结构化信念修正），NCP 将架构从中心化协调器模型演进为适用于开放网络的完全去中心化协议。我们将协议形式化为 8 种消息类型、4 条状态转换规则和 3 个收敛条件，独立于任何特定传输层。

跨三类问题（事实性误解、虚构科学论文、复杂因果推理）的实验评估表明：（a）7 个异构模型成功进行结构化对抗性验证，每次会话产生 40+ 次精确攻击，目标精确到具体推理步骤；（b）Chain-Centric 架构在单次会话中产生 8 次链崩塌和 7 次链修正；（c）外部真值锚定通过 Firecrawl 正确识别了所有 7 个模型最初都认为合理的虚构论文；（d）共识持久化使后续相关问题的会话减少 3-4 轮。

我们识别了当前方法的四个根本局限——优化共识而非真理、博弈论激励失调、模型间训练数据同质性、以及 prompt 诱导的攻击压力——并提出缓解策略，包括外部真值锚定（已实现）、去中心化投票（已实现）、无 prompt 注入的信息路由（已实现）和标准化基准评估（进行中）。

协议规范（NCP v1.0）作为传输层无关的标准发布，兼容 HTTP、WebSocket、消息队列和点对点网络，将 NCP 定位为新兴多智能体生态中补充 MCP（工具层）和 A2A（任务层）的认知信任层。

**关键词：** 多智能体共识、去中心化验证、对抗性推理、协议设计、链中心架构、外部真值锚定、可组合知识
