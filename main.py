import ddpm
from ddpm import train

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 50 #500
    args.batch_size = 2 # 12
    args.image_size = 64 #64
    args.dataset_path = r"../Diffusion-Models-pytorch/dataset" #"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
if __name__ == '__main__':
    launch()