"""
run.py — 真实运行

使用 yunwu.ai 中转的三个模型构建神经回路:
- 感知层: GPT-5.4 (快速初步判断)
- 推理层: GPT-5.4 (深度推理)
- 抑制层: Gemini 3.1 Pro (独立视角找错)
- 元认知层: Claude Opus 4.6 (最终审视，稀疏激活)
"""

import asyncio
import sys

from core.signal import NeuralSignal, Intent, Certainty, Expectation
from core.neuron import Neuron, NeuronConfig, NeuronRole
from core.circuit import NeuralCircuit
from core.adapter import OpenAIAdapter, GeminiAdapter, AnthropicAdapter


async def main():
    # 从用户输入获取任务，或使用默认
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("输入任务 > ")

    print()
    print("=" * 60)
    print("NeuralComm Protocol — 神经回路实时运行")
    print("=" * 60)
    print(f"\n[任务] {task}")
    print()

    # === 构建神经元 ===

    # 感知层 — GPT-5.4 (快速)
    perception = Neuron(
        config=NeuronConfig(
            model_id="gpt-5.4-perception",
            role=NeuronRole.PERCEPTION,
            activation_threshold=0.2,
            max_response_time_ms=5000,
            cost_per_call=0.002,
        ),
        llm_call=OpenAIAdapter().call,
    )

    # 推理层 — GPT-5.4 (深度)
    reasoning = Neuron(
        config=NeuronConfig(
            model_id="gpt-5.4-reasoning",
            role=NeuronRole.REASONING,
            activation_threshold=0.3,
            max_response_time_ms=10000,
            cost_per_call=0.002,
        ),
        llm_call=OpenAIAdapter().call,
    )

    # 抑制层 — Gemini 3.1 Pro (独立视角)
    inhibition = Neuron(
        config=NeuronConfig(
            model_id="gemini-3.1-inhibition",
            role=NeuronRole.INHIBITION,
            activation_threshold=0.2,
            max_response_time_ms=8000,
            cost_per_call=0.001,
        ),
        llm_call=GeminiAdapter().call,
    )

    # 元认知层 — Claude Opus 4.6 (权威验证，稀疏激活)
    metacognition = Neuron(
        config=NeuronConfig(
            model_id="claude-opus-metacognition",
            role=NeuronRole.METACOGNITION,
            activation_threshold=0.3,  # 降低阈值确保参与
            max_response_time_ms=15000,
            cost_per_call=0.03,
        ),
        llm_call=AnthropicAdapter().call,
    )

    # === 构建回路 ===

    circuit = NeuralCircuit(max_iterations=5)
    circuit.add_neuron(perception)
    circuit.add_neuron(reasoning)
    circuit.add_neuron(inhibition)
    circuit.add_neuron(metacognition)
    circuit.auto_wire()

    # === 发送信号 ===

    input_signal = NeuralSignal(
        source_model="user",
        target_model="*",
        content=task,
        intent=Intent.QUERY,
        certainty=Certainty.VERIFIED,
        expectation=Expectation(action="execute", urgency="normal"),
    )

    # === 执行 ===

    print("[开始传播...]")
    print()

    result = await circuit.propagate(input_signal)

    # === 输出 ===

    print("-" * 60)
    print(f"[收敛] {'是' if result.converged else '否'}")
    print(f"[迭代] {result.iterations} 轮")
    print(f"[耗时] {result.total_time_ms:.0f}ms")
    print(f"[激活神经元] {', '.join(result.neurons_activated)}")
    print()

    print("[信号链路]")
    for i, signal in enumerate(result.signal_chain):
        role_tag = signal.source_model
        intent_tag = signal.intent.value
        content_preview = signal.content[:120].replace("\n", " ")
        print(f"  {i}. [{role_tag}] ({intent_tag})")
        print(f"     {content_preview}")
        print()

    print("-" * 60)
    if result.final_signal:
        print(f"[最终输出]")
        print(result.final_signal.content)
        print()
        print(f"[置信度] {result.final_signal.certainty.value}")


if __name__ == "__main__":
    asyncio.run(main())
