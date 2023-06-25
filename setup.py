from setuptools import setup, find_packages

setup(
  name = 'soundstorm-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.16',
  license='MIT',
  description = 'SoundStorm - Efficient Parallel Audio Generation from Google Deepmind, in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/soundstorm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation'
  ],
  install_requires=[
    'accelerate',
    'audiolm-pytorch>=1.2.8',
    'beartype',
    'classifier-free-guidance-pytorch>=0.1.5',
    'einops>=0.6.1',
    'spear-tts-pytorch',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
