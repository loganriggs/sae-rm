# Create 
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from functools import partial
from baukit import Trace

from collections import defaultdict
import matplotlib.pyplot as plt
from einops import rearrange

from IPython.display import display, HTML
# import imgkit
# from PIL import Image, ImageDraw

# def combine_images(feature, save_path = "features/", setting="both"):
#     if(setting =="input_only"):
#         img1 = Image.open(f"features/feature_{feature}_input_combined.png")
#         img3 = Image.open(f"features/uniform_{feature}.png")
#         original_width = img3.width

#         # Resize images if you need to
#         image_scalar = original_width/img1.width 
#         img1 = img1.resize((int(img1.width*image_scalar), int(img1.height*image_scalar)))

#         # Determine dimensions for the new concatenated image
#         new_width = max(img1.width, img3.width)
#         new_height = img1.height + img3.height

#         # Create a new image with a white background
#         new_img = Image.new("RGB", (new_width, new_height), "white")

#         # Paste the images
#         new_img.paste(img1, (0, 0))
#         new_img.paste(img3, (0, img1.height))
#         # Now delete the old images
#         os.remove(f"{save_path}feature_{feature}_input_combined.png")
#         os.remove(f"{save_path}uniform_{feature}.png")
#     else: # Both
#         # Load the images
#         # feature = 1
#         img1 = Image.open(f"features/feature_{feature}_input_combined.png")
#         img2 = Image.open(f"features/feature_{feature}_logit_diff_combined.png")
#         img3 = Image.open(f"features/uniform_{feature}.png")

#         original_width = img3.width

#         # Resize images if you need to
#         image_scalar = original_width/img1.width 
#         img1 = img1.resize((int(img1.width*image_scalar), int(img1.height*image_scalar)))
#         img2 = img2.resize((int(img2.width*image_scalar), int(img2.height*image_scalar)))

#         # Determine dimensions for the new concatenated image
#         new_width = max(img1.width, img3.width)
#         new_height = img1.height + img2.height + img3.height

#         # Create a new image with a white background
#         new_img = Image.new("RGB", (new_width, new_height), "white")

#         # Paste the images
#         new_img.paste(img1, (0, 0))
#         new_img.paste(img2, (0, img1.height))
#         new_img.paste(img3, (0, img1.height + img2.height))
#         # draw = ImageDraw.Draw(new_img)

#         # Now delete the old images
#         os.remove(f"{save_path}feature_{feature}_input_combined.png")
#         os.remove(f"{save_path}feature_{feature}_logit_diff_combined.png")
#         os.remove(f"{save_path}uniform_{feature}.png")
#     # Make directory if it doesn't exist
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     new_img.save(f"{save_path}{feature}_{setting}_concatenated_image.png")
#     return new_img.show()

def get_dictionary_activations(model, dataset, cache_name, max_seq_length, autoencoder, batch_size=32):
    device = model.device
    num_features, d_model = autoencoder.encoder.shape
    datapoints = dataset.num_rows
    dictionary_activations = torch.zeros((datapoints*max_seq_length, num_features))
    token_list = torch.zeros((datapoints*max_seq_length), dtype=torch.int64)
    with torch.no_grad(), dataset.formatted_as("pt"):
        dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
        for i, batch in enumerate(tqdm(dl)):
            batch = batch.to(device)
            token_list[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length] = rearrange(batch, "b s -> (b s)")
            with Trace(model, cache_name) as ret:
                _ = model(batch).logits
                internal_activations = ret.output
                # check if instance tuple
                if(isinstance(internal_activations, tuple)):
                    internal_activations = internal_activations[0]
            batched_neuron_activations = rearrange(internal_activations, "b s n -> (b s) n" )
            batched_dictionary_activations = autoencoder.encode(batched_neuron_activations)
            dictionary_activations[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length,:] = batched_dictionary_activations.cpu()
    return dictionary_activations, token_list

def download_dataset(dataset_name, tokenizer, max_length=256, num_datapoints=None):
    if(num_datapoints):
        split_text = f"train[:{num_datapoints}]"
    else:
        split_text = "train"
    dataset = load_dataset(dataset_name, split=split_text).map(
        lambda x: tokenizer(x['text']),
        batched=True,
    ).filter(
        lambda x: len(x['input_ids']) > max_length
    ).map(
        lambda x: {'input_ids': x['input_ids'][:max_length]}
    )
    return dataset


def ablate_feature_direction(model, dataset, cache_name, max_seq_length, autoencoder, feature, batch_size=32, setting="full_dataset", model_type="causal"):
    device = model.device
    # def less_than_rank_1_ablate(value):
    #     if(isinstance(value, tuple)):
    #         second_value = value[1]
    #         internal_activation = value[0]
    #     else:
    #         internal_activation = value
    #     # Only ablate the feature direction up to the negative bias
    #     # ie Only subtract when it activates above that negative bias.

    #     # Rearrange to fit autoencoder
    #     int_val = rearrange(internal_activation, 'b s h -> (b s) h')
    #     # Run through the autoencoder
    #     act = autoencoder.encode(int_val)
    #     dictionary_for_this_autoencoder = autoencoder.get_learned_dict()
    #     feature_direction = torch.outer(act[:, feature].squeeze(), dictionary_for_this_autoencoder[feature].squeeze())
    #     batch, seq_len, hidden_size = internal_activation.shape
    #     feature_direction = rearrange(feature_direction, '(b s) h -> b s h', b=batch, s=seq_len)
    #     internal_activation -= feature_direction
    #     if(isinstance(value, tuple)):
    #         return_value = (internal_activation, second_value)
    #     else:
    #         return_value = internal_activation
    #     return return_value
    
    def sae_ablation(x, features, sae):
        # affine ablate all sae features up to their bias-component
        # This is iterative, so we remove each feature component one by one
        # This is to avoid double-subtracting directions that features have in common

        # baukit nonsense to handle both residual stream & mlp/attn_output
        if(isinstance(x, tuple)):
            second_value = x[1]
            internal_activation = x[0]
        else:
            internal_activation = x
        batch, seq_len, hidden_size = internal_activation.shape
        int_val = rearrange(internal_activation, "b seq d_model -> (b seq) d_model")
        
        # Encode in features, then remove all features
        f = sae.encode(int_val)
        residual = int_val - sae.decode(f)
        # set f of ablation to zero tensor
        f[..., features] = 0

        x_hat = sae.decode(f)
        x_recon = residual + x_hat

        # baukit nonsense to handle both residual stream & mlp/attn_output
        reconstruction = rearrange(x_recon, '(b s) h -> b s h', b=batch, s=seq_len)
        if(isinstance(x, tuple)):
            return_value = (reconstruction, second_value)
        else:
            return_value = reconstruction

        return return_value


    if(setting == "sentences"):
        # dataset = torch.stack(dataset)
        logit_diffs = torch.zeros_like(dataset)
        with torch.no_grad():
            dataset = dataset.to(device)
            original_logits = model(dataset).logits
            hook_function = partial(sae_ablation, features=[feature], sae=autoencoder)
            with Trace(model, cache_name, edit_output=hook_function) as ret:
                ablated_logits = model(dataset).logits
            if(model_type=="causal"):
                diff_logits = ablated_logits.log_softmax(dim=-1)  - original_logits.log_softmax(dim=-1)  # ablated > original -> negative diff
                gather_tokens = rearrange(dataset[:,1:].to(device), "b s -> b s 1")
                gathered = diff_logits[:, :-1].gather(-1,gather_tokens)
                # append all 0's to the beggining of gathered
                gathered = torch.cat([torch.zeros((gathered.shape[0],1,1)).to(device), gathered], dim=1)
                diff = rearrange(gathered, "b s 1 -> b s")
                logit_diffs =  diff.cpu().tolist()
            else: # reward model/ Sequence classification
                diff_logits = ablated_logits - original_logits
                logit_diffs = diff_logits.cpu()
    else: # full dataset (expensive)
        # ETA: does not support reward model
        assert model_type=="causal", "full dataset only supports causal models"
        datapoints = dataset.num_rows
        logit_diffs = torch.zeros((datapoints*max_seq_length))
        with torch.no_grad(), dataset.formatted_as("pt"):
            dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
            for i, batch in enumerate(tqdm(dl)):
                batch = batch.to(device)
                original_logits = model(batch).logits.log_softmax(dim=-1)
                with Trace(model, cache_name, edit_output=less_than_rank_1_ablate) as ret:
                    ablated_logits = model(batch).logits.log_softmax(dim=-1)
                diff_logits = ablated_logits  - original_logits# ablated > original -> negative diff
                gather_tokens = rearrange(batch[:,1:].to(device), "b s -> b s 1")
                gathered = diff_logits[:, :-1].gather(-1,gather_tokens)
                # append all 0's to the beggining of gathered
                gathered = torch.cat([torch.zeros((gathered.shape[0],1,1)).to(device), gathered], dim=1)
                diff = rearrange(gathered, "b s n -> (b s n)")
                # Add one to the first position of logit diff, so we're always skipping over the first token (since it's not predicted)
                logit_diffs[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length] = diff.cpu()
    return logit_diffs


def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal"):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text

def get_autoencoder_activation(model, cache_name, tokens, autoencoder):
    device = model.device
    with Trace(model, cache_name) as ret, torch.no_grad():
        _ = model(tokens.to(device))
        internal_activations = ret.output
        # check if instance tuple
        if(isinstance(internal_activations, tuple)):
            internal_activations = internal_activations[0]
    internal_activations = rearrange(internal_activations, "b s n -> (b s) n" )
    autoencoder_activations = autoencoder.encode(internal_activations)
    return autoencoder_activations

def ablate_context_one_token_at_a_time(model, dataset, cache_name, autoencoder, feature, max_ablation_length=20):
    all_changed_activations = []
    for token_ind, token_l in enumerate(dataset):
    # for token_ind, token_l in enumerate(full_token_list):
        seq_size = len(token_l)
        original_activation = get_autoencoder_activation(model, cache_name, torch.tensor(token_l).unsqueeze(0), autoencoder)
        original_activation = original_activation[-1,feature].item()
        # Run through the model for each seq length
        if(seq_size==1):
            continue # Size 1 sequences don't have any context to ablate
        # changed_activations = torch.zeros(seq_size).cpu() + original_activation
        changed_activations = torch.zeros(seq_size).cpu()
        minimum_seq_length = max(1, seq_size-max_ablation_length)
        for i in range(minimum_seq_length, seq_size-1):
            ablated_tokens = token_l[:i] + token_l[i+1:]
            # ablated_tokens = token_l
            ablated_tokens = torch.tensor(ablated_tokens).unsqueeze(0)
            with torch.no_grad():
                dictionary_activations = get_autoencoder_activation(model, cache_name, ablated_tokens, autoencoder)
                changed_activations[i] = dictionary_activations[-1,feature].item()
        # changed_activations -= original_activation
        changed_activations[minimum_seq_length:] -= original_activation
        all_changed_activations.append(changed_activations.tolist())
    return all_changed_activations
# Deprecated
# def display_tokens(tokens, activations, tokenizer, logit_diffs=None):
#     return display(HTML(tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs)))

def save_token_display(tokens, activations, tokenizer, path, save=True, logit_diffs=None, show=False, model_type="causal"):
    html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs, model_type=model_type)
    # if(save):
    #     imgkit.from_string(html, path)
    # if(show):
    return display(HTML(html))
    return

def get_feature_indices(feature_index, dictionary_activations, k=10, setting="max"):
    best_feature_activations = dictionary_activations[:, feature_index]
    # Sort the features by activation, get the indices
    if setting=="max":
        found_indices = torch.argsort(best_feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(best_feature_activations)
        min_value = torch.min(best_feature_activations)
        max_value = torch.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    return found_indices

def get_feature_datapoints(found_indices, best_feature_activations, tokenizer, max_seq_length, dataset):
    num_datapoints = dataset.num_rows
    datapoint_indices =[np.unravel_index(i, (num_datapoints, max_seq_length)) for i in found_indices]
    all_activations = best_feature_activations.reshape(num_datapoints, max_seq_length).tolist()
    full_activations = []
    partial_activations = []
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for i, (md, s_ind) in enumerate(datapoint_indices):
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md]["input_ids"])
        full_text.append(tokenizer.decode(full_tok))
        tok = dataset[md]["input_ids"][:s_ind+1]
        full_activations.append(all_activations[md])
        partial_activations.append(all_activations[md][:s_ind+1])
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, partial_activations, full_activations


def get_token_statistics(feature, feature_activation, dataset, tokenizer, max_seq_length, tokens_for_each_datapoint, save_location="", num_unique_tokens=10, setting="input", negative_threshold=-0.01):
    if(setting=="input"):
        nonzero_indices = feature_activation.nonzero()[:, 0]  # Get the nonzero indices
    else:
        nonzero_indices = (feature_activation < negative_threshold).nonzero()[:, 0]
    nonzero_values = feature_activation[nonzero_indices].abs()  # Get the nonzero values

    # Unravel the indices to get the token IDs
    datapoint_indices = [np.unravel_index(i, (dataset.num_rows, max_seq_length)) for i in nonzero_indices]
    all_tokens = [dataset[int(md)]["input_ids"][int(s_ind)] for md, s_ind in datapoint_indices]

    # Find the max value for each unique token
    token_value_dict = defaultdict(int)
    for token, value in zip(all_tokens, nonzero_values):
        token_value_dict[token] = max(token_value_dict[token], value)
    # if(setting=="input"):
    sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: -token_value_dict[x])
    # else:
    #     sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: token_value_dict[x])
    # Take the top 10 (or fewer if there aren't 10)
    max_tokens = sorted_tokens[:min(num_unique_tokens, len(sorted_tokens))]
    total_sums = nonzero_values.abs().sum()
    max_token_sums = []
    token_activations = []
    assert len(max_tokens) > 0, "No tokens found for this feature"
    for max_token in max_tokens:
        # Find ind of max token
        max_token_indices = tokens_for_each_datapoint[nonzero_indices] == max_token
        # Grab the values for those indices
        max_token_values = nonzero_values[max_token_indices]
        max_token_sum = max_token_values.abs().sum()
        max_token_sums.append(max_token_sum)
        token_activations.append(max_token_values)

    if(setting=="input"):
        title_text = "Input Token Activations"
        save_name = "input"
        y_label = "Feature Activation"
    else:
        title_text = "Output Logit-Difference"
        save_name = "logit_diff"
        y_label = "Logit Difference"

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Add a supreme title for the entire figure
    if(setting=="input"):
        fig.suptitle(f"Feature: {feature}", fontsize=32)

    # Boxplot on the left
    ax = axs[0]
    ax.set_title(f'{title_text}')
    max_text = [tokenizer.decode([t]).replace("\n", "\\n").replace(" ", "_") for t in max_tokens]
    ax.set_ylabel(y_label)
    plt.sca(ax)
    plt.xticks(rotation=35)
    ax.title.set_size(24)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.boxplot(token_activations[::-1], labels=max_text[::-1])

    # Bar graph on the right
    ax = axs[1]
    ax.set_title(f'Weighted % of {title_text}')
    plt.sca(ax)
    plt.xticks(rotation=35)
    ax.title.set_size(24)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(f'Weighted Percentage of Total {y_label}')
    ax.bar(max_text[::-1], [t/total_sums*100 for t in max_token_sums[::-1]])

    # Save the figure
    plt.savefig(f'{save_location}feature_{feature}_{save_name}_combined.png', bbox_inches='tight')

    return

from task_patching_utils import SparseAct
import torch as t
from collections import namedtuple
EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])
index_of_chosen_rejection_difference = torch.load("rm_save_files/index_of_chosen_rejection_difference.pt")

# from torchtyping import TensorType
def patching_effect_two(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        tracer_kwargs,
        positions,
        steps=10,
        metric_kwargs=dict(),
):

    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
        # metric_clean = model.output.logits[:, 0].save()
        # metric_clean = model.score.output.save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    # print("metric clean reward: -5.6 or 0.6: ",metric_clean)

    if patch is None:
        # print(f"hidden state clean: {hidden_states_clean[submodule]}")
        v = hidden_states_clean[submodule]
        # print(v)
        v_act = v.act.clone()
        v_res = v.res.clone()
        # print(f"v_act shape: {v_act}")
        # print(f"v_res shape: {v_res.shape}")
        for pos_ind, pos in enumerate(positions):
            v_act[pos_ind, pos:] = 0
            v_res[pos_ind, pos:] = 0
        hidden_states_patch = {
            # k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
            k : SparseAct(act=v_act, res=v_res) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    # output_t = metric_fn(model, **metric_kwargs).save()
                    metrics.append(metric_fn(model, **metric_kwargs))

            metric = sum([m for m in metrics])
            mm = [m.detach().cpu().save() for m in metrics]
            metric.sum().backward(retain_graph=True)


        # print("Metrics", output_t)
        # print("metric ", mm)
        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        print('Out-Out next loop Memory Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        # return grad, delta
        effect = grad @ delta

        # effects[submodule] = effect.detach().cpu()
        # deltas[submodule] = delta.detach().cpu()
        # grads[submodule] = grad.detach().cpu()
        # effects[submodule] = effect.act.detach().cpu()
        # deltas[submodule] = delta.act.detach().cpu()
        # grads[submodule] = grad.act.detach().cpu()
    return effect.act.detach().cpu()