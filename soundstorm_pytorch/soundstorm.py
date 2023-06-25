import math
from random import random
from functools import wraps
from contextlib import nullcontext
from collections import namedtuple

import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, unpack, pack
from einops.layers.torch import Rearrange, EinMix

from beartype import beartype
from beartype.typing import Union, Dict, Optional

from soundstorm_pytorch.attend import Attend

from spear_tts_pytorch import TextToSemantic

from audiolm_pytorch import SoundStream

from tqdm import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def eval_decorator(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# prob helpers

def sample_prob(prob):
    return random() < prob

def coin_flip():
    return sample_prob(0.5)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# schedules

def linear_schedule(t):
    return 1 - t

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# conformer

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None
    ):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask, rotary_emb = rotary_emb) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim_head)

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal,
                attn_flash = attn_flash

            ))

    def forward(self, x):

        rotary_emb = self.rotary_emb(x.shape[-2])

        for block in self.layers:
            x = block(x, rotary_emb = rotary_emb)

        return x

# conformer with sum reduction across quantized tokens at the beginning, along with heads

class ConformerWrapper(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_quantizers,
        conformer: Union[Conformer, Dict[str, any]],
        grouped_quantizers = 1
    ):
        super().__init__()
        self.conformer = conformer

        if isinstance(conformer, dict):
            self.conformer = Conformer(**self.conformer)

        dim = self.conformer.dim

        self.embedding_proj = nn.Sequential(
            nn.Linear(dim * grouped_quantizers, dim),
            nn.LayerNorm(dim)
        ) if grouped_quantizers > 1 else nn.Identity()

        num_codes_with_mask = codebook_size + 1
        num_effective_quantizers = num_quantizers * grouped_quantizers

        self.code_embeds = nn.Embedding(num_codes_with_mask * num_effective_quantizers, dim)

        self.register_buffer('quantizer_offsets', torch.arange(num_effective_quantizers) * num_codes_with_mask, persistent = False)
        self.register_buffer('mask_tokens', self.quantizer_offsets + num_codes_with_mask, persistent = False)

        self.dim = dim
        self.codebook_size = codebook_size

        self.num_codes_with_mask = num_codes_with_mask
        self.num_quantizers = num_quantizers
        self.grouped_quantizers = grouped_quantizers

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * num_effective_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h = num_effective_quantizers)
        )

        # each quantizer codebook would require its own logits weight and bias matrices
        # the amazing einops makes this easy with 'EinMix'

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (n gq) d -> b n gq d', gq = num_effective_quantizers),
            EinMix(
                'b n gq d -> b n gq l',
                weight_shape = 'gq d l',
                bias_shape = 'gq l',
                gq = num_effective_quantizers,
                l = codebook_size,
                d = dim
            ),
            Rearrange('b ... d -> b (...) d')
        )

    def forward(
        self,
        x,
        cond = None,
        sum_embeds = None,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        """
        einops notation:
        b - batch
        n - sequence
        g - groups
        q - quantizers
        d - feature dimension
        """

        n, q, g = x.shape[-1], self.num_quantizers, self.grouped_quantizers
        assert divisible_by(n, g * q), 'sequence must be divisible by number of quantizers'

        x = rearrange(x, 'b (n gq) -> b n gq', gq = g * q)
        x = x + self.quantizer_offsets

        x = self.code_embeds(x)

        x = reduce(x, 'b n (g q) d -> b n (g d)', 'sum', g = g)

        x = self.embedding_proj(x)

        if exists(sum_embeds):
            x = x + sum_embeds

        if exists(cond):
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            x = x + cond

        x = self.conformer(x)
        embeds = self.heads(x)

        if return_embeddings or not exists(self.to_logits):
            return embeds

        logits = self.to_logits(embeds)

        if return_logits_and_embeddings:
            return logits, embeds

        return logits

# for main logits as well as self token critic

class LogitHead(nn.Module):
    def __init__(
        self,
        net: ConformerWrapper,
        logit_dim
    ):
        super().__init__()
        self.net = net
        dim = net.dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

# main soundstorm class, which is just a maskgit

LossBreakdown = namedtuple('LossBreakdown', ['generator_loss', 'critic_loss'])

class SoundStorm(nn.Module):

    @beartype
    def __init__(
        self,
        net: ConformerWrapper,
        *,
        soundstream: Optional[SoundStream] = None,
        spear_tts_text_to_semantic: Optional[TextToSemantic] = None,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # which percentage of the tokens masked will stay the same, done in original MLM paper
        random_token_prob = 0.1,         # which percentage of tokens to be replaced with random token, done in original MLM paper
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # when unmasking, whether it can remask previously unmasked        
        self_token_critic = False,       # https://aclanthology.org/2021.naacl-main.409/
        critic_loss_weight = 1.,
        num_semantic_token_ids = None,
        semantic_pad_id = -1,
        wav2vec_target_sample_hz = None,
        wav2vec_downsample_factor = None,
        codec_target_sample_hz = None,
        codec_downsample_factor = None,
    ):
        super().__init__()

        # conformer settings

        self.net = net
        dim = net.dim
        self.dim = dim
        self.num_tokens = net.codebook_size

        # set soundstream

        self.soundstream = soundstream

        if exists(soundstream):
            self.codec_target_sample_hz = soundstream.target_sample_hz
            self.codec_downsample_factor = soundstream.downsample_factor
        else:
            self.codec_target_sample_hz = codec_target_sample_hz
            self.codec_downsample_factor = codec_downsample_factor

        if exists(self.soundstream):
            assert net.grouped_quantizers == soundstream.rq_groups
            assert net.codebook_size == soundstream.codebook_size
            assert net.num_quantizers == soundstream.num_quantizers

        # set text-to-semantic

        self.text_to_semantic = spear_tts_text_to_semantic

        if exists(spear_tts_text_to_semantic) and exists(spear_tts_text_to_semantic.wav2vec):
            assert not (exists(wav2vec_downsample_factor) or exists(wav2vec_target_sample_hz)), 'wav2vec downsample factor and sampling freq being auto-set from the text-to-semantic module passed in, as it contains the wav2vec instance'
            self.wav2vec = spear_tts_text_to_semantic.wav2vec
            self.wav2vec_target_sample_hz = maybe_wav2vec.target_sample_hz
            self.wav2vec_downsample_factor = maybe_wav2vec.downsample_factor
        else:
            self.wav2vec = None
            self.wav2vec_target_sample_hz = wav2vec_target_sample_hz
            self.wav2vec_downsample_factor = wav2vec_downsample_factor

        # whether to text condition on audio generation is dependent on whether hyperparameters are supplied

        self.should_condition = exists(self.wav2vec_downsample_factor) and exists(self.wav2vec_target_sample_hz)

        # in the case that text-to-semantic module passed in

        if self.should_condition:
            assert exists(self.codec_target_sample_hz) and exists(self.codec_downsample_factor)

            if exists(spear_tts_text_to_semantic):
                self.semantic_token_emb = spear_tts_text_to_semantic.semantic_token_emb
                self.semantic_cond_to_model_dim = nn.Linear(spear_tts_text_to_semantic, net.dim)
                self.semantic_pad_id = spear_tts_text_to_semantic.semantic_pad_id
            else:
                assert exists(num_semantic_token_ids), 'if you are conditioning, you must pass in the number of semantic token ids'
                self.semantic_token_emb = nn.Embedding(num_semantic_token_ids, dim)
                self.semantic_cond_to_model_dim = nn.Identity()
                self.semantic_pad_id = semantic_pad_id

        # detect token critic settings

        assert not (self_token_critic and exists(token_critic))

        self.num_quantizers = net.num_quantizers
        self.grouped_quantizers = net.grouped_quantizers

        self.mask_id = net.codebook_size

        # afaict, maskgit paper did not do this
        # but may help for self conditioning, as used successfully in original BERT

        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob

        self.steps = steps

        if callable(schedule):
            self.schedule_fn = schedule
        if schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')

        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # self conditioning

        self.self_cond = self_cond

        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(dim))
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob

        # token critic

        self.token_critic = None
        if self_token_critic:
            self.token_critic = LogitHead(net, 1)

        self.critic_loss_weight = critic_loss_weight

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        num_latents = None,
        *,
        cond_semantic_token_ids = None,
        seconds = None,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
        **kwargs
    ):
        assert not (exists(cond_semantic_token_ids) ^ self.should_condition), 'you either have text-conditioning turned on and have not passed in any conditioning semantic token ids, or vice versa'

        assert exists(num_latents) ^ exists(seconds)

        if not exists(num_latents):
            assert exists(self.soundstream), 'soundstream must be passed in to generate in seconds'
            num_latents = (seconds * self.soundstream.target_sample_hz) //  self.soundstream.seq_len_multiple_of

       # maybe condition

        cond_tokens = self.maybe_get_condition(cond_semantic_token_ids)

        # determine batch size and sequence length, which depends whether it is conditioning

        if exists(cond_tokens):
            batch_size, num_latents = cond_tokens.shape[:2]
            sample_one = batch_size == 1
        else:
            sample_one = not exists(batch_size)
            batch_size = default(batch_size, 1)

        seq_len = num_latents * self.grouped_quantizers * self.num_quantizers

        # device and time

        device = self.device

        times = torch.linspace(0., 1., self.steps + 1)

        # sequence starts off as all masked

        shape = (batch_size, seq_len)

        seq = torch.full(shape, self.mask_id, device = device)
        mask = torch.full(shape, True, device = device)

        # slowly demask

        all_mask_num_tokens = (self.schedule_fn(times[1:]) * seq_len).long()

        # self conditioning

        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None

        for mask_num_tokens, steps_until_x0 in tqdm(zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))), total = self.steps):

            self_cond = self.to_self_cond(last_embed) if has_self_cond else None

            logits, embeds = self.net(
                seq,
                cond = cond_tokens,
                sum_embeds = self_cond,
                return_logits_and_embeddings = True,
                **kwargs
            )

            if has_self_cond:
                last_embed = embeds

            if exists(filter_thres):
                logits = top_k(logits, filter_thres)

            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale

            probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

            sampled_ids = gumbel_sample(logits, temperature = max(temperature, 1e-3))

            seq = torch.where(mask, sampled_ids, seq)

            if exists(self.token_critic):
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                scores = 1 - logits.softmax(dim = -1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')

            if mask_num_tokens == 0:
                pass

            if not self.can_mask_prev_unmasked:
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

            mask_indices = scores.topk(mask_num_tokens, dim = -1).indices
            mask = torch.zeros_like(scores, dtype = torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, self.mask_id)

        out = seq

        if exists(self.soundstream):
            seq = rearrange(seq, 'b (n q) -> b n q', q = self.num_quantizers)

            with torch.no_grad():
                self.soundstream.eval()
                out = self.soundstream.decode_from_codebook_indices(seq)
                out = rearrange(out, 'b 1 ... -> b ...')

        if sample_one:
            out = rearrange(out, '1 ... -> ...')

        return out

    def maybe_get_condition(self, token_ids = None, length = None):
        assert not (exists(token_ids) ^ self.should_condition), 'you either have text-conditioning turned on and have not passed in any conditioning semantic token ids, or vice versa'

        if not exists(token_ids):
            return None

        context = torch.no_grad if exists(self.text_to_semantic) else nullcontext

        with context():
            mask = token_ids != self.semantic_pad_id
            token_ids = token_ids.masked_fill(~mask, 0)

            semantic_tokens = self.semantic_token_emb(token_ids)
            cond_tokens = self.semantic_cond_to_model_dim(semantic_tokens)

            # just mask out the padding to 0s and let the network learn that for now
            # eventually should add self attention masking to conformer, and calculate the correct number of masked tokens per variable lengthed batch row

            cond_tokens = cond_tokens.masked_fill(~rearrange(mask, '... -> ... 1'), 0.)


        # now need to interpolate the conditioning tokens
        # to align semantic and vector quantized tokens, time-wise

        cond_length = cond_tokens.shape[-2]

        target_cond_length = math.ceil(cond_length * (self.wav2vec_downsample_factor / self.wav2vec_target_sample_hz) / (self.codec_downsample_factor / self.codec_target_sample_hz))

        # pytorch does not interpolate 1d, so hack by convert to 2d

        cond_tokens = rearrange(cond_tokens, 'b n d -> b d n 1')
        cond_tokens = F.interpolate(cond_tokens, (target_cond_length, 1), mode = 'bilinear')
        cond_tokens = rearrange(cond_tokens, 'b d n 1 -> b n d')

        # whether to curtail or pad to length

        cond_length = cond_tokens.shape[-2]

        if exists(length):
            if cond_length < length:
                cond_tokens = F.pad(cond_tokens, (0, 0, 0, length - cond_length), value = 0.)
            elif cond_length > length:
                cond_tokens = cond_tokens[:, :length]

        return cond_tokens

    def forward(
        self,
        x,
        *,
        cond_semantic_token_ids = None,
        only_train_generator = False,
        only_train_critic = False,
        generator_sample_temperature = None,
        **kwargs
    ):
        # if raw audio passed in, convert to residual quantized vectors

        is_raw_audio = x.dtype == torch.float

        if is_raw_audio:
            assert exists(self.soundstream)
            with torch.no_grad():
                self.soundstream.eval()
                _, x, _ = self.soundstream(x, return_encoded = True)

        # shape

        b, n, gq, device = *x.shape, x.device

        # maybe condition

        cond_tokens = self.maybe_get_condition(cond_semantic_token_ids, length = x.shape[-2])

        # prepare masking

        orig_seq = rearrange(x.clone(), 'b n q -> b (n q)')

        t = torch.randint(0, n, (1,)).item()
        q = torch.randint(0, gq, (1,)).item()

        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        batched_randperm = torch.rand((b, n - t), device = device).argsort(dim = -1).float()

        rand_probs = self.schedule_fn(rand_times)
        num_tokens_mask = (rand_probs * (n - t)).clamp(min = 1.)
        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        # to ensure all tokens produce embeddings, instead of just the ones with [mask] input, as done in seminal BERT MLM paper
        # potentially needed for self-conditioning (on embedding) to work well

        replace_mask_id_mask = mask.clone()
        frac_seq_left = 1.

        if self.no_replace_prob > 0. and coin_flip():
            frac_seq_left -= self.no_replace_prob

            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            replace_mask_id_mask &= ~no_replace_prob_mask

        if self.random_token_prob > 0. and coin_flip():
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
            random_tokens = torch.randint(0, self.num_tokens, (b, n - t), device = device)

            x[:, t:, q] = torch.where(random_token_prob_mask, random_tokens, x[:, t:, q])
            replace_mask_id_mask &= ~random_token_prob_mask

        masked = torch.where(replace_mask_id_mask, self.mask_id, x[:, t:, q])
        masked = rearrange(torch.cat((x[:, :t, q], masked), dim=1), 'b n -> b n 1')
        masked = torch.cat((x[:, :, :q], masked, x[:, :, q+1:]), dim=2)
        masked = rearrange(masked, 'b n q -> b (n q)')

        prompt_mask = torch.full((b, t), False, device=device)
        lower_quantizers_mask = torch.full((b, n, q), False, device=device)
        upper_quantizers_mask = torch.full((b, n, (gq - q - 1)), True, device=device)
        mask = rearrange(torch.cat((prompt_mask, mask), dim=1), 'b n -> b n 1')
        mask = torch.cat((lower_quantizers_mask, mask, upper_quantizers_mask), dim=2)
        mask = rearrange(mask, 'b n q -> b (n q)')

        # self conditioning

        if self.self_cond:
            self_cond = self.null_embed

            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    self_cond = self.net(masked, cond = cond_tokens, return_embeddings = True, **kwargs).detach()

            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # logits

        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            logits = self.net(masked, cond = cond_tokens, **kwargs)

        # cross entropy loss

        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )

        if not exists(self.token_critic) or only_train_generator:
            return loss, LossBreakdown(loss, None)

        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        generated = torch.where(mask, sampled_ids, orig_seq)

        critic_logits = self.token_critic(generated)
        critic_labels = (sampled_ids != orig_seq).float()

        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )

        # determine losses to be returned based on what researcher wants to train

        if only_train_critic:
            total_loss = critic_loss
            loss = None
        else:
            total_loss = loss + critic_loss * self.critic_loss_weight

        return total_loss, LossBreakdown(loss,  critic_loss)
