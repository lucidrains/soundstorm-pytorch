from pathlib import Path
import re
from shutil import rmtree

from beartype import beartype
from beartype.typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split

from audiolm_pytorch.data import get_dataloader
from audiolm_pytorch.optimizer import get_optimizer

from soundstorm_pytorch.soundstorm import SoundStorm

from accelerate import Accelerator, DistributedType


# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


class SoundStormTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        model: SoundStorm,
        *,
        num_train_steps,
        num_warmup_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        only_train_generator = False,
        only_train_critic = False,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            **accelerate_kwargs
        )

        self.model = model

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        
        self.only_train_generator = only_train_generator
        self.only_train_critic = only_train_critic

        # optimizer

        self.optim = get_optimizer(
            model.parameters(),
            lr = lr,
            wd = wd
        )

        self.lr = lr
        self.initial_lr = initial_lr
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr}
        self.accelerator.init_trackers("soundstorm", config=hps)

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path, restore_optimizer = True):
        model = self.accelerator.unwrap_model(self.model)
        pkg = model.load(path)

        if restore_optimizer:
            self.optim.load_state_dict(pkg['optim'])
            self.scheduler.load_state_dict(pkg['scheduler'])

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
    
    def train_step(self):
        steps = int(self.steps.item())

        self.model.train()
        
        # adjust the lr according to the schedule
        
        if steps < self.num_warmup_steps:
            # Apply warmup
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # After warmup period, start to apply CosineAnnealingLR
            self.scheduler.step()

        # logs

        logs = {}

        # update generator

        for _ in range(self.grad_accum_every):
            semantic_token_ids, acoustic_token_ids = next(self.dl_iter)

            loss, loss_breakdown = self.model(
                acoustic_token_ids,
                cond_semantic_token_ids = semantic_token_ids,
                only_train_generator = self.only_train_generator,
                only_train_critic = self.only_train_critic
            )

            generator_loss, critic_loss = loss_breakdown
            generator_loss = 0. if generator_loss is None else generator_loss
            critic_loss = 0. if critic_loss is None else critic_loss
            
            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every, 'generator_loss': generator_loss / self.grad_accum_every, 'critic_loss': critic_loss / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']:0.3f}, generator loss: {logs['generator_loss']:0.3f}, critic loss: {logs['critic_loss']:0.3f}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            semantic_token_ids, acoustic_token_ids = next(self.valid_dl_iter)

            with torch.inference_mode():
                self.model.eval()
                valid_loss, valid_loss_breakdown = self.model(acoustic_token_ids, cond_semantic_token_ids = semantic_token_ids)
                
                valid_generator_loss, valid_critic_loss = valid_loss_breakdown
                valid_generator_loss = 0. if valid_generator_loss is None else valid_generator_loss
                valid_critic_loss = 0. if valid_critic_loss is None else valid_critic_loss

            self.print(f'{steps}: valid loss {valid_loss:0.3f}, valid generator loss {valid_generator_loss:0.3f}, valid critic loss {valid_critic_loss:0.3f}')
            self.accelerator.log({"valid_loss": valid_loss, "valid_generator_loss": valid_generator_loss, "valid_critic_loss": valid_critic_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'soundstorm.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
