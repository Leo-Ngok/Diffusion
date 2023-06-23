import jittor as jt
from jittor import nn
import logging
from tqdm import tqdm
from typing import Tuple

class Diffusion(nn.Module):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha:jt.Var = 1. - self.beta
        self.alpha_hat = jt.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self) -> jt.Var:
        return jt.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x:jt.Var, t:jt.Var) -> Tuple[jt.Var, jt.Var]:
        sqrt_alpha_hat:jt.Var = jt.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat:jt.Var = jt.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = jt.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample_timesteps(self, n: int) -> jt.Var:
        return jt.randint(low=1, high=self.noise_steps, shape=(n,))
    
    def sample(self, model:nn.Module, n: int) -> jt.Var:
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with jt.no_grad():
            x = jt.randn((n, 3, self.img_size, self.img_size))
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                eyes:jt.Var = jt.ones(n) * i
                t:jt.Var = eyes.int64() #.long()
                predicted_noise:jt.Var = model(x, t)
                alpha:jt.Var = self.alpha[t][:, None, None, None]
                alpha_hat:jt.Var = self.alpha_hat[t][:, None, None, None]
                beta:jt.Var = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = jt.randn_like(x)
                else:
                    noise = jt.zeros_like(x)
                x:jt.Var = 1 / jt.sqrt(alpha) * \
                    (x - ((1 - alpha) / (jt.sqrt(1 - alpha_hat))) * predicted_noise) \
                    + jt.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = x * 255
        x = x.uint8() #.type(jt.uint8)
        return x
    

    # def execute(self, *args, **kwargs):
    #     pass
