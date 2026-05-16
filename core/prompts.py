"""
prompts.py — 纯算法驱动的 prompt 模板

核心原则: 不给模型任何身份定义。
差异化完全来自输入结构的不同——问什么、给什么上下文、期望什么格式。
模型只是一个"给输入返回输出"的函数。
"""


# === Step 1: 生成初始论述 ===
# 不说"你是分析师"，只说"分析这个问题"

INITIAL_ANALYSIS = """分析以下问题，给出你的完整论述。

问题: {question}

要求:
- 给出明确的结论
- 列出支撑结论的关键论据
- 标注哪些是事实、哪些是假设、哪些是推理
- 如果存在不确定性，明确说明"""


# === Step 2: 分解论证结构 ===
# 不说"你是论证分析专家"，只说"分解这段论述"

DECOMPOSE_ARGUMENT = """将以下论述分解为因果论证结构。

论述:
---
{answer}
---

将论述拆分为独立节点，标注节点类型和支撑关系。

输出严格 JSON:
{{
  "nodes": [
    {{"id": "N1", "content": "陈述内容", "type": "fact/assumption/inference/conclusion", "supports": ["被支撑的节点id"]}},
    ...
  ],
  "root": "最终结论节点id"
}}

类型说明:
- fact: 可查证的客观事实
- assumption: 未经验证的前提或假设
- inference: 从其他节点推导出的中间结论
- conclusion: 最终结论

规则:
- 5-10 个节点
- supports 表示"这个节点支撑了哪些上游节点"
- 只输出 JSON"""


# === Step 3: 寻找崩塌条件 ===
# 不说"你是对抗性思考者"，只问"什么条件下不成立"

FIND_COLLAPSE = """以下是一个论证中的节点:

节点内容: {content}
节点类型: {node_type}
它支撑的结论: {supported_conclusions}

问题: 在什么具体、现实的条件下，这个节点会不成立？

要求:
- 给出 1-3 个具体的崩塌条件
- 每个条件必须是可验证的（不是抽象的）
- 评估每个条件在现实中发生的可能性

输出格式:
CONDITION_1: [具体条件] | LIKELIHOOD: [高/中/低]
CONDITION_2: [具体条件] | LIKELIHOOD: [高/中/低]
CONDITION_3: [具体条件] | LIKELIHOOD: [高/中/低]"""


# === Step 4: 验证崩塌条件 ===
# 不说"你是事实核查员"，只问"这个条件在现实中成立吗"

VERIFY_CONDITION = """以下是一个论证节点和它的崩塌条件:

论证节点: {node_content}
崩塌条件: {condition}

问题: 这个崩塌条件在当前现实中是否成立？

要求:
- 基于你所知的事实判断
- 如果成立，给出具体证据
- 如果不成立，说明为什么
- 如果无法确定，说明缺少什么信息

输出:
VERIFIED: [YES/NO/UNCERTAIN]
EVIDENCE: [具体证据或理由]"""


# === Step 5: 传播影响分析 ===

PROPAGATE_IMPACT = """一个论证图中的底层节点已经崩塌:

崩塌的节点: {collapsed_node}
崩塌原因: {collapse_reason}

依赖这个节点的上游结论:
{dependent_nodes}

问题:
1. 哪些上游结论因此崩塌？
2. 哪些只是被削弱？
3. 最终结论需要如何修正？

输出:
COLLAPSED: [节点id列表]
WEAKENED: [节点id列表]
CONCLUSION_IMPACT: [无影响/需修正/完全推翻]
REVISED_CONCLUSION: [修正后的结论]"""


# === Step 6: 综合最终结论 ===

SYNTHESIZE_FINAL = """经过系统性推敲，以下是结果:

原始问题: {question}
总质疑次数: {total_challenges}
修正次数: {revisions}

经受住质疑的核心论据:
{survived_claims}

被推翻或存在不确定性的论据:
{uncertain_claims}

基于以上推敲结果，给出最终结论。
明确标注:
1. 高度确定的部分
2. 存在不确定性的部分
3. 结论成立的前提条件"""


# === 知识源查询 ===
# 不说"你是知识源"，只问"提供相关事实"

ORACLE_QUERY = """以下论证中存在分歧:

断言: {claim}
依据: {basis}
质疑: {challenge}

请提供与此相关的客观事实和信息:
1. 这个领域的主流认知是什么
2. 有没有相关的数据或研究
3. 是否存在被忽略的重要背景

只提供信息，不做判断。"""
