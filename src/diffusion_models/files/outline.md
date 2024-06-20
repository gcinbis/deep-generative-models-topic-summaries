# Denoising Diffusion Models & Score Matching Models

## 1. High-level explanation of mechanisms
- A high level explanation of mechanisms:
     - Forward diffusion
     - Reverse diffusion

### 1.1. Forward diffusion
- Explanation of probabilistic foundations of forward diffusion.
- Maybe explain what happens to the data during this process (should be
useful later) (Might need a simple explanation of Fourier)

### 1.2. Reverse diffusion
- Model reverse diffusion and discuss it's effect on the data.
- Maybe offer some intuition and discuss what might be different then a
mixture of VAEs and mixtures of gaussian model?

## 2. Learning a diffusion process

### 2.1 Tractability of the reverse diffusion.
Discuss the tractability of the reverse diffusion.
What should you assume to make it tractable?
### 2.2 Markov processes
A very brief talk about markov processes
### 2.3 Gaussian diffusion
What else could be used for this? (Maybe throw a Cold Diffusion paper
discussion here (or later))
### 2.4 Creating a Denoising Model
- Variational Upper bound
- Parameterizing the denoising model
- Training objective weighting
- Explain the training and inference algorithm
- Time representations for the model
### 2.5 Discussions
- Discuss the connection with the VAEs
- Provoke some thinking about the intuition of underlying mechanisms
- Point out the points to improve, like noise scheduling

## 3. Conditioning (This might or might not belong here)

### 3.1 Modals to condition
Mainly talk about image, as there is a whole different topic called
'text-to-image models'

### 3.2 Intuitions about conditioning diffusion models
Why it's not so straightforward? Use some previous intuitions inferred.

### 3.3 Different architectures to condition
- ControlNet
- Ip Adapters
- CLIP
any other?

### 3.4 Known Diffusion Based Models

* Imagen
* Stable Diffusion
