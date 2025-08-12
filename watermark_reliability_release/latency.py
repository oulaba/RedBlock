import time
import torch
from transformers import AutoTokenizer
import numpy as np
from mb_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# 初始化 tokenizer 和水印处理器
def setup_watermark(vocab_size=50257, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # 使用 GPT-2 tokenizer 作为示例
    vocab = list(range(vocab_size))  # 假设词汇表为 0 到 vocab_size-1
    watermark_processor = WatermarkLogitsProcessor(
        vocab=vocab,
        gamma=0.25,
        delta=1.175,
        seeding_scheme="lefthash",
        message_length=8,
        code_length=4,
        base=4,
        device=device
    )
    watermark_detector = WatermarkDetector(
        vocab=vocab,
        gamma=0.25,
        delta=1.175,
        seeding_scheme="lefthash",
        message_length=8,
        code_length=4,
        base=4,
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0
    )
    return tokenizer, watermark_processor, watermark_detector

# 生成模拟输入序列
def generate_input_ids(tokenizer, length, device="cuda"):
    # 生成随机 token 序列
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=device)
    return input_ids

# 测试嵌入延迟
def test_embedding_latency(watermark_processor, input_ids, scores, num_trials=10):
    latencies = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        watermark_processor(input_ids, scores)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
    return np.mean(latencies), np.std(latencies)

# 测试检测延迟
def test_detection_latency(watermark_detector, text, num_trials=10):
    latencies = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        watermark_detector.detect(text=text)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
    return np.mean(latencies), np.std(latencies)

# 主测试函数
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, watermark_processor, watermark_detector = setup_watermark(device=device)

    # 测试不同输入长度的延迟
    sequence_lengths = [10, 50, 100, 500, 1000]
    num_trials = 10
    batch_size = 1  # 单批次测试

    print("=== 测试水印嵌入延迟 ===")
    for length in sequence_lengths:
        input_ids = generate_input_ids(tokenizer, length, device)
        scores = torch.randn(batch_size, tokenizer.vocab_size, device=device)  # 模拟 logits
        watermark_processor.set_message("1010")  # 设置水印消息
        mean_latency, std_latency = test_embedding_latency(watermark_processor, input_ids, scores, num_trials)
        print(f"序列长度: {length}, 平均延迟: {mean_latency:.6f}秒, 标准差: {std_latency:.6f}秒")

    print("\n=== 测试水印检测延迟 ===")
    for length in sequence_lengths:
        # 生成模拟文本（使用 tokenizer 解码随机 token）
        input_ids = generate_input_ids(tokenizer, length, device)
        text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        mean_latency, std_latency = test_detection_latency(watermark_detector, text, num_trials)
        print(f"文本长度: {length}, 平均延迟: {mean_latency:.6f}秒, 标准差: {std_latency:.6f}秒")

if __name__ == "__main__":
    main()