from __future__ import annotations
import random

from torchtyping import TensorType
from collections import namedtuple

from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List, Tuple, TypeVar, Union
from transformers import PreTrainedTokenizerBase
import multiprocessing as mp
import math
T = TypeVar("T", bound=Union[Dataset, DatasetDict])
from torch.utils.data import DataLoader


from torch.nn.functional import kl_div, cross_entropy
from functools import partial
import torch as t
from tqdm import tqdm

from baukit import Trace
from einops import rearrange


class SparseAct():
    def __init__(
            self,
            act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None,
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            resc: TensorType["batch_size", "n_ctx"] = None, # contracted residual
            ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'SparseAct':
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseAct(**kwargs)

    def __mul__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            # Handle SparseAct * SparseAct
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return SparseAct(**kwargs)

    def __rmul__(self, other) -> 'SparseAct':
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)

    def __matmul__(self, other: SparseAct) -> SparseAct:
        # dot product between two SparseActs, except only the residual is contracted
        return SparseAct(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))

    def __add__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) + getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return SparseAct(**kwargs)

    def __radd__(self, other: SparseAct) -> SparseAct:
        return self.__add__(other)

    def __sub__(self, other: SparseAct) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return SparseAct(**kwargs)

    def __truediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseAct(**kwargs)

    def __rtruediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseAct(**kwargs)

    def __neg__(self) -> SparseAct:
        sparse_result = -self.act
        res_result = -self.res
        return SparseAct(act=sparse_result, res=res_result)

    def __invert__(self) -> SparseAct:
            return self._map(lambda x, _: ~x)


    def __gt__(self, other) -> SparseAct:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) > other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")

    def __lt__(self, other) -> SparseAct:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) < other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")

    def __getitem__(self, index: int):
        return self.act[index]

    def __repr__(self):
        if self.res is None:
            return f"SparseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")

    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseAct(**kwargs)

    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseAct(**kwargs)

    def nonzero(self):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return SparseAct(**kwargs)

    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return SparseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseAct(**kwargs)

    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseAct(**kwargs)

    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return SparseAct(**kwargs)

    def save(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self

    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseAct(act=self.act, res=self.res)

    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)

    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)

    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())

    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))

    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)

    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))

    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))

    def abs(self):
        return self._map(lambda x, _: x.abs())


EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        tracer_kwargs,
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
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
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
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
        
    return EffectOut(effects, deltas, grads, total_effect)


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

def task_kl(model, task_dataset, target_token_position, ae, features, activation_name):
    task_kl_losses = t.zeros(len(features))
    with t.no_grad():
        for batch_ind, batch in enumerate(task_dataset):
            batch = batch.to(model.device)
            batch_size = batch.shape[0]
            batch_global_ind = batch_ind * batch_size
            # original_task_logits = model(batch).logits.log_softmax(dim=-1)[t.arange(target_token_position.shape[0]), target_token_position]
            original_task_logits = model(batch).logits.log_softmax(dim=-1)[t.arange(batch_size), target_token_position[batch_global_ind:batch_global_ind + batch_size]]
            for feature_ind, feature in enumerate(tqdm(features)):
                hook_function = partial(sae_ablation, features=[feature.item()], sae=ae)
                with Trace(model, activation_name, edit_output = hook_function) as ret:
                    # Only do KL on the target token pos 
                    # logits = model(batch).logits.log_softmax(dim=-1)[t.arange(target_token_position.shape[0]), target_token_position]
                    logits = model(batch).logits.log_softmax(dim=-1)[t.arange(batch_size), target_token_position[batch_global_ind:batch_global_ind + batch_size]]
                task_kl_losses[feature_ind] = kl_div(original_task_logits, logits, log_target=True, reduction="batchmean").item()
        task_kl_losses /= len(task_dataset)
    return task_kl_losses

def overall_kl(model, kl_data_dataloader, ae, features, activation_name, total_batches=10):
    kl_losses = t.zeros(len(features))
    with t.no_grad():
        for feature_ind, feature in enumerate(tqdm(features)):
            hook_function = partial(sae_ablation, features=[feature.item()], sae=ae)
            for batch_ind, batch in enumerate(kl_data_dataloader):
                if(batch_ind >= total_batches):
                    break
                with Trace(model, activation_name, edit_output=hook_function) as ret:
                    edited_logits = model(batch.to(model.device)).logits.log_softmax(dim=-1)
                original_logits = model(batch.to(model.device)).logits.log_softmax(dim=-1)
                kl_div_value = kl_div(original_logits, edited_logits, log_target=True, reduction="batchmean")
                kl_losses[feature_ind] += kl_div_value.item()
        kl_losses /= total_batches
    return kl_losses


def logit_diff_metric(model, clean_answers, patch_answers):
        return t.mean(
            model.embed_out.output[:,-1, patch_answers] - model.embed_out.output[:,-1, clean_answers],
            dim = -1
        )



def load_overall_dataset(dataset_name, tokenizer, ctx_length, batch_size, shuffle=False):
    dataset = load_dataset(dataset_name, split="train")
    token_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=ctx_length)

    return DataLoader(
        token_dataset["input_ids"], 
        batch_size=batch_size, 
        shuffle=shuffle,
    )

def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = min(mp.cpu_count() // 2, 8),
    text_key: str = "text",
    max_length: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
    add_bos_token: bool = False,
) -> Tuple[T, float]:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_length: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.
        add_bos_token: Whether to prepend a BOS token before each sample. 
        

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_fn(x: Dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
        if add_bos_token:
            # this is already sufficient, as the tokenizer does left-sided padding with 0, which is the EOS token. 
            # BUT: needs to be checked for non-pythia models. 
            chunk_size -= 1 
        sep = tokenizer.eos_token or "<|endoftext|>"
        sep_token = tokenizer.encode(sep)[0]
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )
        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}
            
        if add_bos_token:
            output["input_ids"] = [[sep_token] + c for c in output["input_ids"]] 

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names

def get_task_function(task):
    task_dict = {
        "gender": gender_dataset,
    }
    return task_dict[task]

def tokenize_task_data(text, labels, tokenizer):
    tokenized_text = tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
    target_token_position = (tokenized_text != tokenizer.pad_token_id).sum(dim=1) - 1
    token_labels = tokenizer(labels, return_tensors="pt")["input_ids"][:, 0].tolist()
    return tokenized_text, target_token_position, token_labels

def gender_dataset(num_datapoints):
    masc_names = [" Alex", " Lee", " John", " Will", " Erik", " Adam", " Juan", " Henry", " Richard", " Mike" , " Ken", " Carlos", " Noah", " Lucas", " Jimmy"]
    fem_names = [" Jenny", " Lucy", " Emma", " Daisy", " Chloe", " Betty", " Erin", " Rachel", " Angela", " Maria", " Violet", " Grace", " Ivy", " Anne", " Mary"]
    patterns = [
        "{} fixed the bike and then",
        "{} booked the tickets, and immediately after that",
        "{} left the documents on the table before",
        "{} finished the presentation, and everyone applauded as",
        "{} cooked dinner for the whole family, while",
        "{} received the package, and then",
        "{} found the keys on the counter before",
        "{} wrote a new blog post, and right after publishing it",
        "{} played the guitar, and everyone listened until",
        "{} called their friends, and soon",
        " After the lecture,{} quickly left the room as",
        " During the party,{} sang beautifully until",
        " When the alarm rang,{} was already prepared because",
        " In the meeting,{} gave an opinion and then",
        " On the trip,{} took many photos before",
        " Under the tree,{} found a quiet spot where",
        " Next to the window,{} set up the workspace and soon",
        " Before going to bed,{} checked all the doors and then",
        " After finishing the book,{} felt quite emotional as",
        " Once the game was over,{} shook hands and then"
    ]

    masc_sentences = []
    fem_sentences = []
    masc_labels = " he"
    fem_labels = " she"
    for _ in range(num_datapoints): 
        masc_name = random.choice(masc_names)
        fem_name = random.choice(fem_names)
        pattern = random.choice(patterns)
        masc_sentences.append(pattern.format(masc_name))
        fem_sentences.append(pattern.format(fem_name))

    return masc_sentences, fem_sentences, masc_labels, fem_labels