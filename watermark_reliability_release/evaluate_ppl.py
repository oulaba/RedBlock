import os
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HF_ACCESS_TOKEN'] = ''
import sys
import json
import torch
import argparse
import statistics
import pandas as pd
import wandb
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# from human_eval.data import read_problems


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='0,1,2',
    )
    parser.add_argument(
        "--sample_threshold",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1648,
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=True,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama_N500_T200",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="yahoho-hyt-257",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="block50-uni-mul-run",
        help="The unique name for the run.",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    sample_threshold = args.sample_threshold
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    #problems = read_problems()
    #prompts = [problems[key]['prompt'] for key in problems.keys()]
    #batch_size = len(prompts)

    parent_folder_path = Path(f'./experiments/block-bh-mul-delta-run/')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_name = "8b-200T-4R-lefthash-1.225"
    print(folder_name)
    folder_path = os.path.join(parent_folder_path, folder_name)
    df_watermark = pd.read_json(os.path.join(folder_path, f'gen_table.jsonl'), lines=True)
    #df_watermark = df_watermark.head(batch_size*2)
    completions_watermark = df_watermark[f'w_wm_output'].to_list()
    completions_wo_watermark = df_watermark[f'no_wm_output'].to_list()
    total_rows = len(df_watermark)
    # assert total_rows % batch_size == 0

    model_name, avg_ppl = 'no', None
    gen_model_name = 'meta-llama/Llama-2-7b-hf'
    if gen_model_name == 'codellama/CodeLlama-7b-Instruct-hf':
        model_name = 'codellama/CodeLlama-13b-Instruct-hf'
    elif gen_model_name == 'facebook/opt-1.3b':
        model_name = 'facebook/opt-2.7b'
    elif gen_model_name == 'meta-llama/Llama-2-7b-hf':
        model_name = 'meta-llama/Llama-2-13b-hf'

    print("model name", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
    model.eval()
    # model.to(device)

    # assert
    prompts = df_watermark['truncated_input'].to_list()
    batch_size = len(prompts)
    all_ppl = []
    for i in tqdm(range(len(completions_wo_watermark))):
        prompt = prompts[i]
        output_text = completions_wo_watermark[i]

        # 对prompt进行tokenization
        tokd_prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, max_length=args.max_length,
                                truncation=True).to(device)

        # 对output_text进行tokenization
        tokd_suffix = tokenizer(output_text, return_tensors="pt", add_special_tokens=False, max_length=args.max_length,
                                truncation=True).to(device)

        # 组合输入
        input_ids = torch.cat([tokd_prefix['input_ids'], tokd_suffix['input_ids']], dim=1)

        # 创建标签
        labels = input_ids.clone()
        # 将prompt部分设置为-100
        labels[:, :tokd_prefix['input_ids'].shape[1]] = -100

        # 确保labels和input_ids长度一致
        if labels.shape[1] > input_ids.shape[1]:
            labels = labels[:, :input_ids.shape[1]]
        elif labels.shape[1] < input_ids.shape[1]:
            pad_length = input_ids.shape[1] - labels.shape[1]
            labels = torch.cat([labels, torch.full((labels.shape[0], pad_length), -100, device=device)], dim=1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        if args.wandb:
            run.log({'ppl': ppl}, step=i)
        all_ppl.append(ppl)

    # assert len(all_ppl) // batch_size == args.pass_k
    all_avg_ppl = statistics.mean(all_ppl)
    all_ppl_variance = statistics.variance(all_ppl) if len(all_ppl) >= 2 else 0.0  # 添加方差计算


    all_batch_mean_ppl = []
    all_batch_ppl_variances = []
    for i in range(0, len(all_ppl), batch_size):
        batch_ppl = all_ppl[i:i+batch_size]
        print(len(batch_ppl))
        all_batch_mean_ppl.append(statistics.mean(batch_ppl))
        batch_variance = statistics.variance(batch_ppl) if len(batch_ppl) >= 2 else 0.0
        all_batch_ppl_variances.append(batch_variance)


    term_width = 80
    print('-'*term_width)
    results = {
               'all average ppl': all_avg_ppl,
               'all ppl values': all_ppl,
                'all ppl variance': all_ppl_variance,
               'all batch mean ppl values': all_batch_mean_ppl,
                'all batch ppl variances': all_batch_ppl_variances  # 记录每个batch的方差
               }
    with open(os.path.join(folder_path, f'eval_ppl.json'), 'w') as file:
        json.dump(results, file, indent=4)
    if args.wandb:
        run.finish()