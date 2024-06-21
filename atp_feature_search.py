from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nnsight import LanguageModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "reciprocate/dahoas-gptj-rm-static"
# model_name = "gpt2"
from transformers import AutoConfig
config =  AutoConfig.from_pretrained(model_name)
torch.jit.is_tracing = lambda : True

model = LanguageModel(
    model_name,
    device_map = device,
    automodel = AutoModelForSequenceClassification,
    dispatch = True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

import argparse
from huggingface_hub import hf_hub_download
from dictionary import GatedAutoEncoder

parser = argparse.ArgumentParser(description="Run script with specified parameters")
parser.add_argument('--layer', type=int, default=12, help='Layer number to use (default: 8)')
parser.add_argument('--total_num_of_datapoints', type=int, default=10, help='Total number of datapoints (default: 10)')
parser.add_argument('--batch_size', type=int, default=3, help='Batch size (default: 3)')
parser.add_argument('--token_length_cutoff', type=int, default=500, help='Token length cutoff (default: 300)')
args = parser.parse_args()

# Use the parsed values directly
layer = args.layer
total_num_of_datapoints = args.total_num_of_datapoints
batch_size = args.batch_size
token_length_cutoff = args.token_length_cutoff


activation_name = f"transformer.h.{layer}"
model_id = "Elriggs/rm"
sae_file_save_name = f"ae_layer{layer}"
sae_file_dir = f"sae_results/{sae_file_save_name}"
sae_filename = sae_file_save_name + ".pt"
ae_download_location = hf_hub_download(repo_id=model_id, filename=sae_filename)

sae = GatedAutoEncoder.from_pretrained(ae_download_location).to(device)

# Get module information for path-patching's idiosyncratic requirements
module_name = f"transformer.h.{layer}"
# Get module by it's name
attributes = module_name.split('.')
module = model
for attr in attributes:
    module = getattr(module, attr)

dictionaries = {}
submodule_names = {}
submodule_names[module] = module_name
dictionaries[module] = sae.to(device)
submodules = [module]

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

hh = load_dataset("Anthropic/hh-rlhf", split="train")
hh_ind_short_enough_filepath = f"rm_save_files/hh_ind_lower_than_{token_length_cutoff}_tokens.pt"
# Remove datapoints longer than a specific token_length
# Check if file exists
dataset_size = hh.num_rows
if os.path.exists(hh_ind_short_enough_filepath):
    index_small_enough = torch.load(hh_ind_short_enough_filepath)
else:
    index_small_enough = torch.ones(dataset_size, dtype=torch.bool)
    for ind, text in enumerate(tqdm(hh)):
        chosen_text = text["chosen"]
        rejected_text = text["rejected"]
        #convert to tokens
        length_chosen = len(tokenizer(chosen_text)["input_ids"])
        length_rejected = len(tokenizer(rejected_text)["input_ids"])
        if length_chosen > token_length_cutoff or length_rejected > token_length_cutoff:
            index_small_enough[ind] = False
    # Save the indices
    torch.save(index_small_enough, hh_ind_short_enough_filepath)

# Of those, find the largest-reward subset up to a certain size
top_reward_filename = f"rm_save_files/token_len_{token_length_cutoff}_top_{total_num_of_datapoints}_reward_diff_indices.pt"

if(os.path.exists(top_reward_filename)):
    top_reward_diff_ind = torch.load(top_reward_filename)
else:
    # But first, our cached reward diff is indexed by the 871 token cutoff
    eight_seventy_index = torch.load("rm_save_files/index_small_enough.pt")
    reward_diff = torch.load("/root/sae-rm/rm_save_files/rejected_chosen_reward_diff.pt")
    full_reward_diff = torch.zeros(dataset_size)
    full_reward_diff[eight_seventy_index] = reward_diff
    reward_diff = full_reward_diff[index_small_enough]

    # Get the indices of the top 1000
    top_reward_diff_ind = reward_diff.abs().topk(total_num_of_datapoints).indices
    torch.save(top_reward_diff_ind, top_reward_filename)

# Index the dataset into those
hh = hh.select(index_small_enough.nonzero()[:, 0])
hh = hh.select(top_reward_diff_ind)
hh_dl = DataLoader(hh, batch_size=batch_size, shuffle=False)

import torch
from tqdm import tqdm

num_datapoints = len(hh)
index_of_chosen_rejection_difference = torch.zeros(num_datapoints, dtype=torch.int16)

# Assuming hh_dl is a DataLoader that returns batches of data
subsets = 0
for i, batch in enumerate(tqdm(hh)):
    chosen_texts = batch["chosen"]
    rejected_texts = batch["rejected"]

    # Tokenize texts in batches
    chosen_tokens = tokenizer(chosen_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=token_length_cutoff)["input_ids"]
    rejected_tokens = tokenizer(rejected_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=token_length_cutoff)["input_ids"]

    chosen_token_original_length = tokenizer(chosen_texts, return_tensors="pt")["input_ids"].shape[1]
    rejected_token_original_length = tokenizer(rejected_texts, return_tensors="pt")["input_ids"].shape[1]
    min_length = min(chosen_token_original_length, rejected_token_original_length)

    # Compare tokens and find divergence points
    divergence_matrix = (chosen_tokens != rejected_tokens).to(torch.int)  # Matrix of 1s where tokens differ

    # Find the first divergence index for each pair of texts
    divergence_indices = divergence_matrix.argmax(dim=1)
    if divergence_indices == min_length:
        subsets += 1
        divergence_indices -= 1

    index_of_chosen_rejection_difference[i] = divergence_indices
torch.save(index_of_chosen_rejection_difference, f"rm_save_files/index_of_chosen_rejection_difference_{token_length_cutoff}.pt")

from interp_utils import patching_effect_two
import gc
# import torch
gc.collect()
tracer_kwargs = {'validate' : False, 'scan' : False}
def get_reward(model):
    return model.output.logits[:, 0]
torch.cuda.empty_cache()
print('Original Memory Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

num_features = sae.encoder.weight.shape[0]
num_datapoints = len(hh)*2
all_effects_per_feature = torch.zeros((num_datapoints, num_features))

batch_size = hh_dl.batch_size
for batch_ind, batch in enumerate(tqdm(hh_dl)):
    batch_loc = batch_ind * batch_size
    pos = [p_ind.item() for p_ind in index_of_chosen_rejection_difference[batch_loc:batch_loc+batch_size]]
    # pos = [0 for _ in range(batch_size)] # Just collect all of them and filter out later

    for text_ind, text_key in enumerate(["chosen", "rejected"]):
        tokens = tokenizer(batch[text_key], padding=True, truncation=True, return_tensors="pt")["input_ids"]
        # set tokens to a variable length token
        # length = token_length_cutoff
        # tokens = tokenizer(batch[text_key], padding="max_length", truncation=True, max_length=length, return_tensors="pt")["input_ids"]
        effects = patching_effect_two(
            tokens.to(device),
            None,
            model,
            submodules = submodules,
            dictionaries = dictionaries,
            tracer_kwargs=tracer_kwargs,
            positions = pos,
            metric_fn = get_reward,
            steps = 4,
        )

        # set the values before the divergence point to 0
        # Compute the starting index for the current batch and text
        start_index = batch_ind * batch_size * 2 + text_ind * batch_size
        end_index = start_index + batch_size
        
        all_effects_per_feature[start_index:end_index] = effects.sum(1)

        gc.collect()
        torch.cuda.empty_cache()

# make sae_file_dir if it doesn't exist
if not os.path.exists(sae_file_dir):
    os.makedirs(sae_file_dir)
torch.save(all_effects_per_feature, f"{sae_file_dir}/all_effects_per_feature_token_{token_length_cutoff}_top_{total_num_of_datapoints}.pt")