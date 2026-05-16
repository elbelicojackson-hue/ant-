"""
example.py — 使用示例

演示如何构建一个神经回路:
- 感知层: 本地小模型 (免费，快速)
- 推理层: 中等模型 (便宜)
- 抑制层: 本地小模型 (免费，专注找错)
- 元认知层: 大模型 (贵，只在需要时激活)
- 执行层: 不需要模型，直接输出
"""

import asyncio

from core.signal import (
    NeuralSignal, Intent, Certainty,
    TraceStep, Uncertainty, Expectation,
)
from core.neuron import Neuron, NeuronConfig, NeuronRole
from core.circuit import NeuralCircuit
from core.adapter import MockAdapter


async def main():
    """构建并运行一个神经回路"""

    # === 1. 创建神经元 ===

    # 感知层 — 快速反应
    perception = Neuron(
        config=NeuronConfig(
            model_id="ollama-llama3-8b",
            role=NeuronRole.PERCEPTION,
            activation_threshold=0.2,
            max_response_time_ms=2000,
            cost_per_call=0.0,
            capabilities=["pattern_recognition", "quick_assessment"],
        ),
        llm_call=MockAdapter([
            "INTENT: propose\n"
            "CERTAINTY: moderate\n"
            "CONTENT: 检测到进程 nginx (PID 2847) 占用8080端口，"
            "该进程已运行72小时，内存占用正常(128MB)。"
            "初步判断: 这是一个正常的web服务进程。"
        ]).call,
    )

    # 推理层 — 深度分析
    reasoning = Neuron(
        config=NeuronConfig(
            model_id="gpt-4o-mini",
            role=NeuronRole.REASONING,
            activation_threshold=0.3,
            max_response_time_ms=5000,
            cost_per_call=0.001,
            capabilities=["logical_reasoning", "multi_step_planning"],
        ),
        llm_call=MockAdapter([
            "INTENT: propose\n"
            "CERTAINTY: strong\n"
            "CONTENT: 分析完成。nginx PID 2847 是用户主动部署的web服务。\n"
            "方案: 不应终止该进程。如果用户需要释放8080端口，"
            "建议将nginx配置改为其他端口，或确认是否有其他服务需要8080。"
        ]).call,
    )

    # 抑制层 — 找错
    inhibition = Neuron(
        config=NeuronConfig(
            model_id="ollama-llama3-8b-critic",
            role=NeuronRole.INHIBITION,
            activation_threshold=0.2,
            max_response_time_ms=3000,
            cost_per_call=0.0,
            capabilities=["error_detection", "logic_validation"],
        ),
        llm_call=MockAdapter([
            "INTENT: confirm\n"
            "CERTAINTY: strong\n"
            "CONTENT: PASS — 推理逻辑无明显漏洞。"
            "nginx作为web服务占用8080是合理的，不应盲目终止。"
        ]).call,
    )

    # 元认知层 — 最终审视 (只在抑制层不确定时激活)
    metacognition = Neuron(
        config=NeuronConfig(
            model_id="claude-sonnet",
            role=NeuronRole.METACOGNITION,
            activation_threshold=0.7,  # 高阈值，不轻易激活
            max_response_time_ms=10000,
            cost_per_call=0.01,
            capabilities=["meta_reasoning", "quality_assessment"],
        ),
        llm_call=MockAdapter([
            "INTENT: confirm\n"
            "CERTAINTY: verified\n"
            "CONTENT: 推理链完整，结论合理。"
        ]).call,
    )

    # === 2. 构建回路 ===

    circuit = NeuralCircuit(max_iterations=5)
    circuit.add_neuron(perception)
    circuit.add_neuron(reasoning)
    circuit.add_neuron(inhibition)
    circuit.add_neuron(metacognition)

    # 自动布线
    circuit.auto_wire()

    # === 3. 发送输入信号 ===

    input_signal = NeuralSignal(
        source_model="user",
        target_model="*",
        content="帮我找出占用8080端口的进程并杀掉",
        intent=Intent.QUERY,
        certainty=Certainty.VERIFIED,  # 用户的需求是确定的
        expectation=Expectation(
            action="execute",
            focus="port 8080",
            urgency="normal",
        ),
    )

    # === 4. 执行 ===

    print("=" * 60)
    print("NeuralComm Protocol — 神经回路执行演示")
    print("=" * 60)
    print(f"\n[输入] {input_signal.content}")
    print(f"[意图] {input_signal.intent.value}")
    print()

    result = await circuit.propagate(input_signal)

    # === 5. 输出结果 ===

    print(f"[结果] 收敛: {result.converged}")
    print(f"[迭代] {result.iterations} 轮")
    print(f"[耗时] {result.total_time_ms:.1f}ms")
    print(f"[激活] {result.neurons_activated}")
    print()

    print("[信号链路]")
    for i, signal in enumerate(result.signal_chain):
        print(f"  {i}. [{signal.source_model}] ({signal.intent.value}) → {signal.content[:80]}")

    print()
    if result.final_signal:
        print(f"[最终输出] {result.final_signal.content}")
        print(f"[置信度] {result.final_signal.certainty.value}")

    print()
    print("[回路拓扑]")
    topology = circuit.get_topology()
    for nid, info in topology["neurons"].items():
        print(f"  {nid}: role={info['role']}, activations={info['stats']['activation_count']}")
    print()
    for syn in topology["synapses"]:
        print(f"  {syn['source']} --({syn['type']}, w={syn['weight']})--> {syn['target']}")


if __name__ == "__main__":
    asyncio.run(main())
