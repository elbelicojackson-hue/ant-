# NCP — Neural Consensus Protocol v1.0

## 概述

NCP 是一个 agent 间的共识通信协议。它定义了异构推理引擎之间如何通过结构化状态暴露、对抗性验证和多方投票，就共享知识状态达成可审计共识。

NCP 不定义语言，不定义模型内部实现，不绑定特定传输层。它只定义：状态格式、操作类型、转换规则、终止条件。

---

## 1. 寻址

每个参与协议的实体有一个全局唯一地址：

```
ncp://<node>/<agent_id>
```

示例：
```
ncp://tokyo-01/gpt-5.4
ncp://local/deepseek-v4
ncp://p2p-abc123/mimo-v2.5
```

链和步骤的寻址：
```
ncp://<node>/<agent_id>/chain/<chain_id>/v<version>/step/<N>
```

简写（同一 session 内）：
```
<agent_id>.v<version>.Step<N>
```

---

## 2. 状态格式

### 2.1 Chain（推理链）

```json
{
  "type": "chain",
  "chain_id": "c_a1b2c3",
  "agent": "ncp://tokyo-01/gpt-5.4",
  "version": 2,
  "status": "active",
  "created_at": "2026-05-16T12:00:00Z",
  "steps": [
    {
      "index": 1,
      "content": "断言内容",
      "status": "active",
      "attacked_count": 0
    },
    {
      "index": 2,
      "content": "断言内容",
      "status": "fortified",
      "attacked_count": 3
    }
  ]
}
```

步骤状态枚举：
- `active` — 未经充分验证
- `fortified` — 经受攻击存活 (attacked_count ≥ 2 且未崩塌)
- `collapsed` — 被攻击且投票判定崩塌

链状态枚举：
- `active` — 当前有效
- `broken` — 关键步骤崩塌，链整体失效
- `archived` — 被新版本取代

### 2.2 Claim（共识断言）

```json
{
  "type": "claim",
  "claim_id": "cl_x7y8z9",
  "content": "断言内容",
  "alpha": 0.71,
  "confidence": 0.95,
  "verified_at": "2026-05-16T12:30:00Z",
  "supporters": ["ncp://tokyo-01/gpt-5.4", "ncp://local/deepseek-v4", ...],
  "attacks_survived": 5,
  "depends_on": ["cl_abc123"]
}
```

---

## 3. 操作类型（Messages）

NCP 定义 8 种操作消息：

### 3.1 PROPOSE_CHAIN

agent 提交一条新的推理链。

```json
{
  "op": "PROPOSE_CHAIN",
  "from": "ncp://tokyo-01/gpt-5.4",
  "session": "sess_001",
  "chain": { ... },
  "timestamp": "...",
  "nonce": "..."
}
```

### 3.2 CHALLENGE

对某个步骤发起攻击。

```json
{
  "op": "CHALLENGE",
  "from": "ncp://local/deepseek-v4",
  "target": "ncp://tokyo-01/gpt-5.4/chain/c_a1b2c3/v2/step/3",
  "reason": "该步骤假设X，但在Y条件下不成立",
  "evidence_refs": ["ev_001"],
  "confidence": "high",
  "timestamp": "..."
}
```

### 3.3 VOTE

对一次 CHALLENGE 投票。

```json
{
  "op": "VOTE",
  "from": "ncp://p2p-abc/mimo-v2.5",
  "challenge_ref": "ch_xyz789",
  "verdict": "collapsed",
  "reason": "攻击理由成立，步骤确实依赖了未验证假设",
  "weight": 1.2,
  "timestamp": "..."
}
```

verdict 枚举：`collapsed` | `defended`

### 3.4 REPAIR

修正被攻破的链，产生新版本。

```json
{
  "op": "REPAIR",
  "from": "ncp://tokyo-01/gpt-5.4",
  "old_chain": "c_a1b2c3/v2",
  "new_chain": { ... },
  "timestamp": "..."
}
```

规则：新链 version = old_version + 1，旧链自动变为 `archived`。

### 3.5 VERIFY

请求事实查证。

```json
{
  "op": "VERIFY",
  "from": "ncp://local/coordinator",
  "query": "论文A (Nature, 2025) 是否真实存在",
  "context": ["相关断言内容..."],
  "timestamp": "..."
}
```

### 3.6 EVIDENCE

提交查证结果到证据池。

```json
{
  "op": "EVIDENCE",
  "from": "ncp://local/deepseek-v4",
  "evidence_id": "ev_001",
  "content": "经搜索，Nature 2025 年未发表相关论文...",
  "source_url": "https://...",
  "timestamp": "..."
}
```

### 3.7 CONVERGE

声明共识达成。

```json
{
  "op": "CONVERGE",
  "from": "ncp://local/coordinator",
  "session": "sess_001",
  "claims": [
    {"claim_id": "cl_001", "content": "...", "alpha": 0.85}
  ],
  "reason": "true_consensus, α=0.85",
  "timestamp": "..."
}
```

### 3.8 JOIN / LEAVE

agent 动态加入或退出 session。

```json
{
  "op": "JOIN",
  "from": "ncp://new-node/claude-4",
  "session": "sess_001",
  "capabilities": ["reasoning", "search"],
  "timestamp": "..."
}
```

---

## 4. 转换规则

### 4.1 攻击判定

```
CHALLENGE 发起后:
  → 被攻击方收到通知，返回 VOTE (defended/collapsed)
  → 第三方裁判收到通知，返回 VOTE
  → 攻击方天然计为 1 票 collapsed

判定: 
  weighted_collapse = Σ(vote_weight_i) for collapsed votes
  weighted_defend = Σ(vote_weight_i) for defended votes
  
  if weighted_collapse > weighted_defend → 步骤崩塌
  else → 步骤存活 (attacked_count += 1)
```

### 4.2 链崩塌传播

```
当 step.status → collapsed:
  if step == chain.steps[-1] (最后一步):
    chain.status → broken
  elif count(collapsed steps) >= len(steps) / 2:
    chain.status → broken
  
当 chain.status → broken:
  agent 必须在 N 轮内发送 REPAIR，否则退出 session
```

### 4.3 共识收敛

```
每 N 轮计算:
  对所有活跃链提取原子断言
  对每个断言计算 α = Σ(weight_i * agree_i) / Σ(weight_i)
  
  if avg(α for top claims) ≥ 0.65:
    发送 CONVERGE
    session 结束
```

### 4.4 信誉更新

```
攻击成功:
  attacker.rating += K * (1 - expected)
  defender.rating += K * (0 - expected)

攻击失败:
  attacker.rating += K * (0 - expected)  
  defender.rating += K * (1 - expected)

expected = 1 / (1 + 10^((opponent_rating - self_rating) / 400))

vote_weight = f(rating)  // rating → [0.5, 2.0]
```

---

## 5. 信息路由规则

**核心原则：算法不告诉 agent 做什么，只控制 agent 看到什么。**

### 5.1 可见性路由

```
每轮攻击前，coordinator 决定每个 agent 看到哪些链:
  - 所有活跃链都可见 (公平性)
  - 包含高 UCB 步骤的链排在前面 (注意力引导)
  - 未验证步骤标记 ⚠ (信息标注，非指令)
  - 已加固步骤标记 ✓ (信息标注，非指令)
```

### 5.2 证据可见性

```
查证结果存入证据池后:
  - 所有 agent 在下一轮可见证据池内容
  - 证据作为"已知事实"展示，不作为"攻击指令"
```

### 5.3 禁止的路由行为

```
❌ 不得在 prompt 中包含 "请攻击 X 的 Step3"
❌ 不得在 prompt 中包含 "你应该同意/反对"
❌ 不得隐藏某个 agent 的链 (除非该 agent 已 LEAVE)
```

---

## 6. 传输层

NCP 是传输层无关的。消息可以通过以下方式传递：

| 传输方式 | 适用场景 | 延迟 |
|----------|----------|------|
| 进程内函数调用 | 本地实验 | <1ms |
| HTTP/REST | 跨服务器 | 50-500ms |
| WebSocket | 实时交互 | 10-50ms |
| NATS/Kafka | 大规模集群 | 5-20ms |
| libp2p | 去中心化网络 | 100-2000ms |

### 6.1 HTTP 绑定示例

```
POST /ncp/v1/sessions/{session_id}/messages

Body: NCP 消息 JSON

Response: 
  202 Accepted (异步处理)
  200 OK + response message (同步处理)
```

### 6.2 发现机制

agent 通过以下方式发现彼此：
- 静态配置（当前方式）
- DNS-SD / mDNS（局域网）
- DHT（去中心化网络）
- 注册中心（中心化部署）

---

## 7. 安全

### 7.1 消息签名

每条 NCP 消息包含发送方签名：
```json
{
  "signature": {
    "algorithm": "ed25519",
    "public_key": "...",
    "value": "..."
  }
}
```

### 7.2 防重放

每条消息包含 `nonce` + `timestamp`，接收方拒绝重复 nonce 或过期消息。

### 7.3 权限

- 只有链的 owner 可以发送 REPAIR
- 只有 session 参与者可以发送 CHALLENGE 和 VOTE
- CONVERGE 只能由 coordinator 或多数 agent 联合发起

---

## 8. 与现有协议的关系

```
┌─────────────────────────────────────────┐
│  应用层                                  │
│  NCP (共识协议)                          │
│  ┌───────────────────────────────────┐  │
│  │ MCP (工具调用) │ A2A (任务委派)    │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  传输层: HTTP / WebSocket / libp2p      │
├─────────────────────────────────────────┤
│  网络层: TCP/IP                          │
└─────────────────────────────────────────┘
```

NCP 可以和 MCP/A2A 共存：
- agent 用 MCP 调用工具（搜索、计算）
- agent 用 A2A 委派子任务
- agent 用 NCP 就推理结论达成共识

---

## 9. 版本演进

- v1.0（当前）：单 session、中心化 coordinator、同步投票
- v2.0（规划）：多 session 组合、去中心化、异步投票
- v3.0（远期）：自适应拓扑、agent 自主组网、共识链持久化

---

*NCP v1.0 — 2026-05-16*
*Neural Consensus Protocol Working Group*
