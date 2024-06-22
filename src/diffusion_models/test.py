from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import torch

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
).cuda()

trainer = Trainer(
    diffusion,
    'data',
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 20000,         # total training steps
    # gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    calculate_fid = False,
)

trainer.load(20)

outs = diffusion.sample(batch_size=16, return_all_timesteps=True)

import torchvision
from PIL import Image, ImageSequence

# Convert the tensor to the range [0, 1]
outs = (outs + 1) / 2

# Create a list to store all frames
frames = []

for i in range(0, outs.shape[1], 10):  # Change here to process every 50th timestep
    # Select the i-th timestep for all batch samples
    timestep_outs = outs[:, i, :, :, :]

    # Make a grid
    grid = torchvision.utils.make_grid(timestep_outs, nrow=4)

    # Convert the tensor to a PIL image
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(grid)

    # Append the image to the frames list
    frames.append(img)

# Save the frames as a gif
frames[0].save('output.gif', save_all=True, append_images=frames[1:], loop=0, duration=10)
