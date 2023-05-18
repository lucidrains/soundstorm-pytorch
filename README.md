<img src="./soundstorm.png" width="450px"></img>

## Soundstorm - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2305.09636">SoundStorm</a>, Efficient Parallel Audio Generation from Google Deepmind, in Pytorch.

They basically applied <a href="https://arxiv.org/abs/2202.04200">MaskGiT</a> to the residual vector quantized latents from <a href="https://github.com/lucidrains/audiolm-pytorch#soundstream--encodec">Soundstream</a>. The transformer architecture they chose to use is one that fits well with the audio domain, named <a href="https://arxiv.org/abs/2005.08100">Conformer</a>

<a href="https://google-research.github.io/seanet/soundstorm/examples/">Project Page</a>

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/docs/accelerate/index">🤗 Accelerate</a> for providing a simple and powerful solution for training

## Install

```bash
$ pip install soundstorm-pytorch
```

## Usage

```python
import torch
from soundstorm_pytorch import SoundStorm, ConformerWrapper

conformer = ConformerWrapper(
    codebook_size = 1024,
    num_quantizers = 4,
    conformer = dict(
        dim = 512,
        depth = 2
    ),
)

model = SoundStorm(
    conformer,
    steps = 18,          # 18 steps, as in original maskgit paper
    schedule = 'cosine'  # currently the best schedule is cosine
)

# get your codes from the soundstream

codes = torch.randint(0, 1024, (2, 1024))

# learn to de-mask

loss, _ = model(codes)
loss.backward()

# generate by de-masking iteratively

generated = model.generate(1024, batch_size = 2) # (2, 1024)
```

## Todo

- [ ] integrate soundstream
- [ ] turn it into a command line tool
- [ ] add cross attention and adaptive layernorm conditioning (just copy paste in the entire conformer repository, if conditioning adds too much cruft to the other repo)

## Citations

```bibtex
@misc{borsos2023soundstorm,
    title   = {SoundStorm: Efficient Parallel Audio Generation}, 
    author  = {Zalán Borsos and Matt Sharifi and Damien Vincent and Eugene Kharitonov and Neil Zeghidour and Marco Tagliasacchi},
    year    = {2023},
    eprint  = {2305.09636},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Chang2022MaskGITMG,
    title   = {MaskGIT: Masked Generative Image Transformer},
    author  = {Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman},
    journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    pages   = {11305-11315}
}
```

```bibtex
@article{Lezama2022ImprovedMI,
    title   = {Improved Masked Image Generation with Token-Critic},
    author  = {Jos{\'e} Lezama and Huiwen Chang and Lu Jiang and Irfan Essa},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2209.04439}
}
```

```bibtex
@inproceedings{Nijkamp2021SCRIPTSP,
    title   = {SCRIPT: Self-Critic PreTraining of Transformers},
    author  = {Erik Nijkamp and Bo Pang and Ying Nian Wu and Caiming Xiong},
    booktitle = {North American Chapter of the Association for Computational Linguistics},
    year    = {2021}
}
```
