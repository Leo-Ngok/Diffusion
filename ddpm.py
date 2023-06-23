from diffusion import Diffusion
from unet import UNet
from jittor import nn
from utilities import setup_logging, get_data, save_images
from argparse import Namespace
import logging
from tqdm import tqdm
import jittor as jt
import os

image = None
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

jt.flags.use_cuda=1
def train(args: Namespace):
    setup_logging(args.run_name)
    device:str = args.device
    model = UNet()
    optimizer = nn.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device = device)
    #logger = SummaryWriter(os.path.)
    dataloader = get_data(args)
    l = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for i, (img, _) in enumerate(pbar):
            global image
            img:jt.Var = img
            t = diffusion.sample_timesteps(img.shape[0])
            x_t, noise = diffusion.noise_images(img, t)
            pred_noise:jt.Var = model(x_t, t)
            loss:jt.Var = mse(noise, pred_noise)
            optimizer.step(loss)
            image = img
            pbar.set_postfix(MSE=loss.item())
            #logger.add_scalar("MSE",)
            pass
        sampled_img = diffusion.sample(model, image.shape[0])
        save_images(sampled_img, os.path.join('results', args.run_name, f"{epoch}.jpg"))
        jt.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

        jt.save_image(image, 't3est')
