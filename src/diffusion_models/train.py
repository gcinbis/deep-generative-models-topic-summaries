from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

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

trainer.train()