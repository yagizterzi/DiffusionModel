import torch
import numpy as np

class DDPMSampler:
    #SCALED LINEAR SCHEDULE
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, 
                 beta_end: float = 0.0120):
        # Initializes self.betas as a tensor of noise variances for each training step, increasing quadratically from beta_start to beta_end.
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        
        # Initialize alpha = 1-betas
        self.alphas = 1.0-self.betas
        # Get the cummulative product of alphas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps=50):
        # Sets the timesteps for the inference process.
        # It calculates a subset of the training timesteps to be used during inference.
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
    def _get_previous_timestep(self,timestep)->int:
        # Calculates the previous timestep based on the current timestep.
        # This is used to step backwards in the diffusion process during inference.
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self,timestep)->int:
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod[prev_t] if prev_t >= 0 else self.one
        current_beta = 1-alpha_prod_t/alpha_prod_t_prev
        # Variance  = (1-at).bt/(1-at-1).bt 
        variance =(1-alpha_prod_t) *current_beta /(1-alpha_prod_t_prev)*current_beta
        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength=1):
        # Noise level is adjusted by strength and this strength(0-1) adjust the skipped noisifaction step ratio
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        
        # Adjust the steps for skkiping
        start_step = self.timesteps[start_step:]
        self.start_step = start_step
        
    def step(self,timestep: int ,latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        #Get alphas and betas
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # Compute the predicted noise 
        # Formula for it x_0 ≈ (x_t - sqrt(1 - α̅_t) * model_output) / sqrt(α̅_t)
        # x0: predicted image ,xt: latent 
        pred_original_sample = (latents - beta_prod_t**(0,5)*model_output)/alpha_prod_t**(0,5)
        
        # Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        
        #  Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance 
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(self,original_samples: torch.FloatTensor,timesteps: torch.IntTensor,) -> torch.FloatTensor:
        
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        
        # To have operations with latent we need to add dimensions to sqrt_alpha_prod
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        # Calculate it for formula
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        # Same dimension issue
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
          
        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod  
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
        
        
        
        
    
        
        