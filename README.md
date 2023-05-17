<img src="./soundstorm.png" width="450px"></img>

## Soundstorm - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2305.09636">SoundStorm</a>, Efficient Parallel Audio Generation from Google Deepmind, in Pytorch.

They basically applied <a href="https://arxiv.org/abs/2202.04200">MaskGiT</a> to the residual vector quantized latents from <a href="https://github.com/lucidrains/audiolm-pytorch#soundstream--encodec">Soundstream</a>. The transformer architecture they chose to use is one that fits well with the audio domain, named <a href="https://arxiv.org/abs/2005.08100">Conformer</a>

<a href="https://google-research.github.io/seanet/soundstorm/examples/">Project Page</a>

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

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
