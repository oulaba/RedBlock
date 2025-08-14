import os
import random
import time
import torch

# HF classes

from datasets import load_dataset, IterableDataset

from torch import Tensor
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    LlamaTokenizer
)
from accelerate import Accelerator
from .data.lfqa import load_lfqa
from .data.essays import load_essays
from .data.wikitext import load_wikitext

MAX_GENERATIONS = int(10000)  # Hardcoded max length to avoid infinite loop
HF_TOKEN = os.environ['HF_ACCESS_TOKEN']

import torch
from peft import PeftModel

# def load_model(args):
#     """Load and return the model and tokenizer"""
#     accelerator = Accelerator()
#
#     # 确定模型类型
#     if "guanaco" in args.model_name_or_path.lower():
#         model_type = "guanaco"
#         base_model_name = "huggyllama/llama-7b"
#         adapters_name = "timdettmers/guanaco-7b"
#         args.is_decoder_only_model = True
#     elif "falcon" in args.model_name_or_path.lower():
#         model_type = "falcon"
#         model_name = "tiiuae/falcon-7b"
#         adapters_name = None
#         args.is_decoder_only_model = True
#     else:
#         # 现有模型类型检测逻辑
#         args.is_seq2seq_model = any(
#             [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
#         )
#         args.is_decoder_only_model = any(
#             [(model_type in args.model_name_or_path.lower()) for model_type in ["gpt", "opt", "bloom", "llama", "mistral", "gemma"]]
#         )
#         model_name = args.model_name_or_path
#         model_type = model_name
#         adapters_name = None
#
#     # 加载模型
#     if model_type == "guanaco":
#         dtype = torch.float16 if args.load_fp16 else torch.float32
#         model = AutoModelForCausalLM.from_pretrained(
#             base_model_name, torch_dtype=dtype, device_map="auto"
#         )
#         model = PeftModel.from_pretrained(model, adapters_name)
#     elif model_type == "falcon":
#         dtype = torch.float16 if args.load_fp16 else torch.float32
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, torch_dtype=dtype, device_map="auto"
#         )
#     elif args.is_seq2seq_model:
#         model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
#     elif args.is_decoder_only_model:
#         dtype = torch.float16 if args.load_fp16 else torch.float32
#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_name_or_path, torch_dtype=dtype, device_map="auto"
#         )
#     else:
#         raise ValueError(f"Unknown model type: {args.model_name_or_path}")
#
#     # 设置设备
#     if args.use_gpu:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         if not args.load_fp16:  # 如果不是 fp16，则手动移动到设备
#             model = model.to(device)
#     else:
#         device = "cpu"
#     model.eval()
#
#     # 设置 tokenizer
#     if args.is_decoder_only_model:
#         padding_side = "left"
#     else:
#         raise NotImplementedError(
#             "Need to check how to handle padding for seq2seq models when calling generate"
#         )
#
#     if "llama" in args.model_name_or_path.lower():
#         tokenizer = LlamaTokenizer.from_pretrained(
#             args.model_name_or_path, padding_side=padding_side
#         )
#         model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
#         model.config.bos_token_id = 1
#         model.config.eos_token_id = 2
#     elif "guanaco" in args.model_name_or_path.lower():
#         tokenizer = LlamaTokenizer.from_pretrained(
#             base_model_name, padding_side=padding_side
#         )
#         model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
#         model.config.bos_token_id = 1
#         model.config.eos_token_id = 2
#     elif "mistral" in args.model_name_or_path.lower():
#         tokenizer = AutoTokenizer.from_pretrained(
#             args.model_name_or_path, padding_side=padding_side, use_fast=True
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#     elif "falcon" in args.model_name_or_path.lower():
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, padding_side=padding_side, use_fast=True
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, padding_side=padding_side, use_fast=True
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#
#     args.model_max_length = model.config.max_position_embeddings
#
#     # 使用 Accelerator 准备模型和 tokenizer
#     model, tokenizer = accelerator.prepare(model, tokenizer)
#     device = accelerator.device
#
#     return model, tokenizer, device

def load_model(args):
    """Load and return the model and tokenizer"""

    accelerator = Accelerator()

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path.lower()) for model_type in ["gpt", "opt", "bloom", "llama", "mistral", "gemma", "guanaco", "falcon"]]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        dtype = torch.float16 if args.load_fp16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        raise NotImplementedError(
            "Need to check how to handle padding for seq2seq models when calling generate"
        )

    if "llama" in args.model_name_or_path.lower() or "guanaco" in args.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    elif "mistral" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif "falcon" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side, use_fast=True
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    args.model_max_length = model.config.max_position_embeddings

    model, tokenizer = accelerator.prepare(model, tokenizer)
    device = accelerator.device

    return model, tokenizer, device


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "lfqa":
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./utils/data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)
    if isinstance(indexed_dataset, IterableDataset):
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.take(args.limit_indices)
            if args.limit_indices is not None
            else shuffled_dataset
        )
    else:
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.select(range(args.limit_indices))
            if args.limit_indices is not None
            else shuffled_dataset
        )

    if args.limit_indices is None:
        try:
            args.limit_indices = len(limited_dataset)
        except Exception as e:
            # can't infer length of dataset, probably because it's an IterableDataset
            pass
    return limited_dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    input_ids = tokenizer(
        example[input_col_name], return_tensors="pt", truncation=True, max_length=model_max_length
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:, : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"
    # example['instructions'] = "<s>[INST] Write a complete essay with an introduction, main body, and conclusion following the below instructions.[/INST]" \
    #                   + example['instructions']
    # example['text'] = "<s>[INST] Complete the following news article: [/INST]" + example['text']
    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
        )
        # Parse the results of tokenization. Simple, since
        # the prompt and baseline completion are from the raw text
        re_decoded_input = example[args.input_col_name]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_inputs")
        full_sample_len = untruncated_inputs.shape[1]
        # decode the preprocessed input to store for audit
        re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        # also decode the original suffix of the input for audit as the baseline
        baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
        decoded_baseline_completion = tokenizer.batch_decode(
            baseline_completion_tokens, skip_special_tokens=True
        )[0]
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (
        input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
    ), "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    return collator({"input_ids": input_ids})["input_ids"]


def generate(
        examples,
        data_collator=None,
        generate_without_watermark=None,
        generate_with_watermark=None,
        watermark_processor=None,
        tokenizer=None,
        device=None,
        args=None,
        segment_length=100,
        num_segments=2,
):
    input_ids = collate_batch(input_ids=examples["input_ids"], collator=data_collator).to(device)
    batch_size = len(examples["input_ids"])

    # 设置水印消息
    msg_length = args.message_length
    if args.zero_bit:
        msg_binary = "0"
        msg_encoded = "0"
    else:
        use_ecc = False
        msg_binary, msg_encoded = sample_message(msg_length, use_ecc)

    watermark_processor.set_message(msg_encoded)
    print(f"Binary msg:\n{msg_binary}")
    print(f"Binary encoded msg:\n{msg_encoded}")
    print(f"Converted msg:\n{watermark_processor.converted_message}")
    messages = [msg_binary] * batch_size

    # final_output_without_watermark = torch.tensor([], device=device)
    # final_output_with_watermark = torch.tensor([], device=device)

    tokens_without_wm_list = []
    tokens_with_wm_list = []

    # 为每个样本初始化 sampled_positions_list
    sampled_positions_per_sample = ["" for _ in range(batch_size)]
    current_input_ids = input_ids
    wm_time_total = 0
    non_wm_time_total = 0

    for segment_idx in range(num_segments):
        print(f"Generating segment {segment_idx + 1}/{num_segments}")
        with torch.no_grad():
            if args.generation_seed is not None:
                torch.manual_seed(args.generation_seed)
            s_time = time.time()
            output_without_watermark = generate_without_watermark(
                input_ids=current_input_ids, max_new_tokens=segment_length
            )

            # print(f"Type: {type(output_without_watermark)}")
            # print(f"Dtype: {output_without_watermark.dtype}")
            # print(f"Shape: {output_without_watermark.shape}")
            # print(f"Sample: {output_without_watermark[0, :5]}")

            non_wm_time_total += time.time() - s_time

            if args.generation_seed is not None:
                torch.manual_seed(args.generation_seed)
            s_time = time.time()
            output_with_watermark = generate_with_watermark(
                input_ids=current_input_ids, max_new_tokens=segment_length
            )

            # print(f"Type: {type(output_with_watermark)}")
            # print(f"Dtype: {output_with_watermark.dtype}")
            # print(f"Shape: {output_with_watermark.shape}")
            # print(f"Sample: {output_with_watermark[0, :5]}")

            wm_time_total += time.time() - s_time

        new_tokens_without_wm = output_without_watermark[:, current_input_ids.shape[-1]:]
        new_tokens_with_wm = output_with_watermark[:, current_input_ids.shape[-1]:]
        tokens_without_wm_list.append(new_tokens_without_wm)
        tokens_with_wm_list.append(new_tokens_with_wm)

        # final_output_without_watermark = torch.cat(
        #     (final_output_without_watermark, new_tokens_without_wm), dim=1
        # )
        # final_output_with_watermark = torch.cat(
        #     (final_output_with_watermark, new_tokens_with_wm), dim=1
        # )
        current_input_ids = output_with_watermark

        # 获取采样位置并分配到每个样本
        sampled_positions = watermark_processor.flush_position()
        # 按样本累积位置
        for i in range(batch_size):
            sampled_positions_per_sample[i] += sampled_positions[i]  # 直接拼接字符串

        print(f"sampled positions per sample: {sampled_positions_per_sample}")

        # for i in range(batch_size):
        #     # 假设 sampled_positions 是每个样本的采样位置列表
        #     # 如果 watermark_processor.flush_position() 返回的是批次级别的列表，需要调整其实现
        #     sampled_positions_per_sample[i].extend(
        #         sampled_positions[i] if len(sampled_positions) > i else sampled_positions[0])
        #
        # print(final_output_without_watermark.dtype)
        # print(final_output_with_watermark.dtype)

    final_output_without_watermark = torch.cat(tokens_without_wm_list, dim=1)
    final_output_with_watermark = torch.cat(tokens_with_wm_list, dim=1)

    print(final_output_without_watermark.dtype)
    print(final_output_with_watermark.dtype)

    decoded_output_without_watermark = tokenizer.batch_decode(
        final_output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        final_output_with_watermark, skip_special_tokens=True
    )

    examples.update({
        "no_wm_output": decoded_output_without_watermark,
        "w_wm_output": decoded_output_with_watermark,
        # "sampled_positions": sampled_positions,  # 每个样本一个列表
        "sampled_positions": sampled_positions_per_sample,  # 每个样本一个列表
        "message": messages,
        "no_wm_output_length": (final_output_without_watermark != tokenizer.pad_token_id).sum(dim=-1).tolist(),
        "w_wm_output_length": (final_output_with_watermark != tokenizer.pad_token_id).sum(dim=-1).tolist(),
        "wm_encoding_time": [wm_time_total] * batch_size,
        "non_wm_encoding_time": [non_wm_time_total] * batch_size
    })

    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    return examples


# def generate(
#     examples,
#     data_collator=None,
#     generate_without_watermark=None,
#     generate_with_watermark=None,
#     watermark_processor=None,
#     tokenizer=None,
#     device=None,
#     args=None,
#     segment_length=50,
#     watermark_bits=4,  # 假设水印比特数为 8，可根据实际调整
# ):
#     input_ids = collate_batch(input_ids=examples["input_ids"], collator=data_collator).to(device)
#     batch_size = len(examples["input_ids"])
#     num_segments = args.max_new_tokens // segment_length
#     remaining_tokens = args.max_new_tokens % segment_length
#
#     # 设置水印消息
#     msg_length = args.message_length
#     if args.zero_bit:
#         msg_binary = "0"
#         msg_encoded = "0"
#     else:
#         use_ecc = False
#         msg_binary, msg_encoded = sample_message(msg_length, use_ecc)
#     full_watermark = msg_encoded
#     messages = [msg_binary] * batch_size
#
#     final_output_without_watermark = torch.tensor([], device=device)
#     final_output_with_watermark = torch.tensor([], device=device)
#     current_input_ids = input_ids
#     wm_time_total = 0
#     non_wm_time_total = 0
#
#     # 完整段生成
#     for segment_idx in range(num_segments):
#         print(f"Generating segment {segment_idx + 1}/{num_segments}")
#         watermark_processor.set_message(full_watermark)  # 每次设置完整水印
#         with torch.no_grad():
#             if args.generation_seed is not None:
#                 torch.manual_seed(args.generation_seed)
#             s_time = time.time()
#             output_without_watermark = generate_without_watermark(
#                 input_ids=current_input_ids, max_new_tokens=segment_length
#             )
#             non_wm_time_total += time.time() - s_time
#
#             if args.generation_seed is not None:
#                 torch.manual_seed(args.generation_seed)
#             s_time = time.time()
#             output_with_watermark = generate_with_watermark(
#                 input_ids=current_input_ids, max_new_tokens=segment_length
#             )
#             wm_time_total += time.time() - s_time
#
#         new_tokens_without_wm = output_without_watermark[:, current_input_ids.shape[-1]:]
#         new_tokens_with_wm = output_with_watermark[:, current_input_ids.shape[-1]:]
#         final_output_without_watermark = torch.cat(
#             (final_output_without_watermark, new_tokens_without_wm), dim=1
#         )
#         final_output_with_watermark = torch.cat(
#             (final_output_with_watermark, new_tokens_with_wm), dim=1
#         )
#         current_input_ids = output_with_watermark
#         sample_position = watermark_processor.flush_position()  # 记录采样位置
#
#     # 剩余 token 处理
#     if remaining_tokens > 0:
#         print(f"Generating remaining tokens: {remaining_tokens} tokens")
#         # 设置部分水印（可选：若坚持完整水印则保留 full_watermark）
#         tokens_per_bit = segment_length / watermark_bits
#         embeddable_bits = min(watermark_bits, int(remaining_tokens / tokens_per_bit))
#         # partial_watermark = full_watermark[:embeddable_bits] if embeddable_bits > 0 else ""
#         # watermark_processor.set_message(partial_watermark)  # 调整为部分水印
#         watermark_processor.set_message(full_watermark)
#
#         with torch.no_grad():
#             if args.generation_seed is not None:
#                 torch.manual_seed(args.generation_seed)
#             s_time = time.time()
#             output_without_watermark = generate_without_watermark(
#                 input_ids=current_input_ids, max_new_tokens=remaining_tokens
#             )
#             non_wm_time_total += time.time() - s_time
#
#             if args.generation_seed is not None:
#                 torch.manual_seed(args.generation_seed)
#             s_time = time.time()
#             output_with_watermark = generate_with_watermark(
#                 input_ids=current_input_ids, max_new_tokens=remaining_tokens
#             )
#             wm_time_total += time.time() - s_time
#
#         new_tokens_without_wm = output_without_watermark[:, current_input_ids.shape[-1]:]
#         new_tokens_with_wm = output_with_watermark[:, current_input_ids.shape[-1]:]
#         final_output_without_watermark = torch.cat(
#             (final_output_without_watermark, new_tokens_without_wm), dim=1
#         )
#         final_output_with_watermark = torch.cat(
#             (final_output_with_watermark, new_tokens_with_wm), dim=1
#         )
#         watermark_processor.flush_position()  # 记录采样位置
#
#     # 解码输出
#     decoded_output_without_watermark = tokenizer.batch_decode(
#         final_output_without_watermark, skip_special_tokens=True
#     )
#     decoded_output_with_watermark = tokenizer.batch_decode(
#         final_output_with_watermark, skip_special_tokens=True
#     )
#
#     examples.update({
#         "no_wm_output": decoded_output_without_watermark,
#         "w_wm_output": decoded_output_with_watermark,
#         "sampled_positions": watermark_processor.flush_position(),  # 调整为合适格式
#         "message": messages,
#         "no_wm_output_length": (final_output_without_watermark != tokenizer.pad_token_id).sum(dim=-1).tolist(),
#         "w_wm_output_length": (final_output_with_watermark != tokenizer.pad_token_id).sum(dim=-1).tolist(),
#         "wm_encoding_time": [wm_time_total] * batch_size,
#         "non_wm_encoding_time": [non_wm_time_total] * batch_size
#     })
#
#     if watermark_processor.spike_entropies is not None:
#         examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
#         examples["spike_entropies"] = [
#             ents[:num_toks]
#             for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
#         ]
#
#     return examples


try:
    from reedmuller import reedmuller
except:
    print("Error loading error correcting code module")


# 使用RM纠错码方式会导致最终要嵌入的水印长度翻倍，导致嵌入水印质量下降？
def sample_message(msg_length, use_ecc, ecc_params=None):
    msg_decimal = random.getrandbits(msg_length)
    msg_binary = format(msg_decimal, f"0{msg_length}b")
    if use_ecc:
        rm = reedmuller.ReedMuller(2, 5)
        msg_encoded = ''.join(map(str, rm.encode(list(map(int, msg_binary)))))
    else:
        msg_encoded = msg_binary

    return msg_binary, msg_encoded

# from pyfinite import ffield, rs_code, genericmatrix
# import random
# def sample_message(msg_length, use_ecc, ecc_params=None):
#     msg_decimal = random.getrandbits(msg_length)
#     msg_binary = format(msg_decimal, f"0{msg_length}b")
#     if use_ecc:
#         # RS码参数
#         n = 4   # 码字长度
#         k = 2   # 消息长度（符号数）
#         t = 1   # 纠错能力
#         F = ffield.FField(4)  # GF(16)，每个符号4位
#
#         # 自定义 RSCode 类来修复参数传递
#         class FixedRSCode(rs_code.RSCode):
#             def CreateEncoderMatrix(self):
#                 self.encoderMatrix = genericmatrix.GenericMatrix(
#                     self.k,  # 行数
#                     self.n,  # 列数
#                     zeroElement=0,
#                     identityElement=1,
#                     add=self.field.Add,
#                     sub=self.field.Subtract,
#                     mult=self.field.Multiply,
#                     div=self.field.Divide
#                 )
#                 # 设置生成矩阵（范德蒙德矩阵）
#                 for i in range(self.k):
#                     for j in range(self.n):
#                         self.encoderMatrix.Set(i, j, self.field.Power(self.generator, i * j))
#
#         rs = FixedRSCode(F, n, k)
#         # 将二进制消息转换为4位符号
#         msg_bytes = [int(msg_binary[i:i+4], 2) for i in range(0, len(msg_binary), 4)]
#         encoded = rs.Encode(msg_bytes)
#         msg_encoded = ''.join(format(x, '04b') for x in encoded)
#     else:
#         msg_encoded = msg_binary
#     return msg_binary, msg_encoded