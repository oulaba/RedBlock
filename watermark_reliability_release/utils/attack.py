import openai
import random

from utils.dipper_attack_pipeline import generate_dipper_paraphrases

from utils.evaluation import OUTPUT_TEXT_COLUMN_NAMES
from utils.copy_paste_attack import single_insertion, triple_insertion_single_len, k_insertion_t_len

SUPPORTED_ATTACK_METHODS = ["gpt", "dipper", "copy-paste", "scramble","delete","insert","synonym"]

import nltk
from nltk.corpus import wordnet

def get_synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    synset = random.choice(synsets)
    lemmas = [lemma.name() for lemma in synset.lemmas() if lemma.name() != word]
    if not lemmas:
        return word
    return random.choice(lemmas)

# Word Deletion Attack: Deletes 20% of words randomly
def word_deletion_attack(example, tokenizer=None, args=None):
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            text = example[column]
            words = text.split(" ")
            n = len(words)
            k = int(0.1 * n)  # Delete 20% of words
            if k > 0 and n > 0:
                indices_to_delete = random.sample(range(n), k)
                words = [word for i, word in enumerate(words) if i not in indices_to_delete]
            attacked_text = " ".join(words)
            example[f"{column}_attacked"] = attacked_text
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"])
    return example

# Word Insertion Attack: Inserts 20% additional words randomly from the text
def word_insertion_attack(example, tokenizer=None, args=None):
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            text = example[column]
            words = text.split(" ")
            n = len(words)
            k = int(0.2 * n)  # Insert 20% more words
            for _ in range(k):
                if words:  # Ensure there are words to choose from
                    word_to_insert = random.choice(words)
                    insert_index = random.randint(0, len(words))
                    words.insert(insert_index, word_to_insert)
            attacked_text = " ".join(words)
            example[f"{column}_attacked"] = attacked_text
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"])
    return example

# Synonym Replacement Attack: Replaces words with synonyms with 20% probability
def synonym_replacement_attack(example, tokenizer=None, args=None):
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            text = example[column]
            words = text.split(" ")
            new_words = []
            for word in words:
                if random.random() < 0.2:  # 20% chance to replace
                    synonym = get_synonym(word)
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            attacked_text = " ".join(new_words)
            example[f"{column}_attacked"] = attacked_text
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"])
    return example

def scramble_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            # # if not, copy the orig w_wm_output to w_wm_output_attacked
            # NOTE changing this to return "" so that those fail/we can filter out these examples
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            sentences = example[column].split(".")
            random.shuffle(sentences)
            example[f"{column}_attacked"] = ".".join(sentences)
            example[f"{column}_attacked_length"] = len(
                tokenizer(example[f"{column}_attacked"])["input_ids"]
            )
    return example


def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"

    gen_row = example

    if args.no_wm_attack:
        original_text = gen_row["no_wm_output"]
    else:
        original_text = gen_row["w_wm_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}

    from tenacity import retry, stop_after_attempt, wait_random_exponential
    from openai import OpenAI

    client = OpenAI(base_url = "https://chatapi.littlewheat.com/v1",
        api_key = "sk-KApaiTee3p62YAkvFr1Yn9zB566MbUyfR3RHmrXaRADukkSw")

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    try:
        outputs = completion_with_backoff(
            model=args.attack_model_name,
            messages=[query_msg],
            temperature=args.attack_temperature,
            max_tokens=args.attack_max_tokens,
        )
        # outputs = client.chat.completions.create(
        #     model='gpt-3.5-turbo',
        #     messages=[query_msg],
        #     temperature=0.7,
        #     max_tokens=args.max_new_tokens
        # )

    except Exception as e:
        print('Error:', e)
        print('Number of', total_attack_reviews_num, 'not generated')


    attacked_text = outputs.choices[0].message.content
    assert (
        len(outputs.choices) == 1
    ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
    example["w_wm_output_attacked"] = attacked_text
    if args.verbose:
        print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
        print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    return example


def dipper_attack(dataset, lex=None, order=None, args=None):
    dataset = generate_dipper_paraphrases(dataset, lex=lex, order=order, args=args)
    return dataset


def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds


def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            tokenized = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            # empty tensors are float type by default
            # this leads to an error when constructing pyarrow table
            if not str(tokenized.dtype) == "torch.int64":
                tokenized = tokenized.long()
            example[f"{text_col}_tokd"] = tokenized
    return example


def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))
    # input ids might have been converted to float if empty rows exist
    for key in example.keys():
        if "tokd" in key:
            example[key] = list(map(int, example[key]))

    if args.cp_attack_type == "single-single":  # 1-t
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "triple-single":  # 3-t
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    # error occurred during attacking
    if tokenized_attacked_output is None:
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    tokenized_attacked_output = list(map(int, tokenized_attacked_output))

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example