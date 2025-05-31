import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# Main pipeline method
def generate(prompt: str,
             uncon_prompt: str,# Negative Prompt(Unconditional Output)
             input_img=None,
             strength=0.8,  # How much attention payed to the starting image
             do_cfg=True,  # Classifier free guidance
             cfg_scale=7.5,  # How much attention to our prompt(1 - 14)
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},  # Pre trained models
             seed=None,  # How we want to initialze our random number generator
             device=None,  # Where we want to create our tensor
             idle_device=None,
             tokenizer=None,
             ):
    with torch.no_grad():
        
        # First strength needs to be between 0-1
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        # If we want to move things to CPU we create this lambda fucntion
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

       # Now we will create a random number generator to generate the noise
        generator = torch.Generator(device=device)
        if seed is None:
           generator.seed()
        else:
           generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Turn the prompt into tokens with tokenizer 
            # First we have to encode the prompt then we want append the padding to max length(77) and we take the input ids 
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            
            # We will turn this ids into tensor (Batch_Size, Seq_Len)(Seq_Len=77)
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            # Now run it in clip to turn them into embeddings
            cond_context = clip(cond_tokens) # (Batch_Size,Seq_Len,Dim)(Dim=768)
            
            # Now same procedure for negative prompt
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncon_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            
            # Combine them into single context 
            context = torch.cat([cond_context, uncond_context])# (2 * Batch_Size, Seq_Len, Dim) = (2,77,768)
        #If we dont use CFG we only have to use prompt
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)# (Batch_Size, Seq_Len)
            context = clip(tokens)# (Batch_Size, Seq_Len, Dim)
            
        #When we finished using CLIP we can move it to idle device for stopping gpu overload 
        to_idle(clip)
        
        #Now load and use the sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)# Tells the sampler how many steps for inferencing
        else:
            raise ValueError("Sampler %s")

        latent_shape = (1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
        
        # IMAGE TO IMAGE 
        if input_img:
            #Run the image through encoder
            # Move encoder to device 
            encoder=models["encoder"]
            encoder.to(device)
            
            input_img_tensor = input_img.resize((WIDTH,HEIGHT))
            # We will transform input image into a tensor
            input_img_tensor = np.array(input_img_tensor)
            input_img_tensor = torch.tensor(input_img_tensor,device=device,dtype=torch.float32)# (512,512,3(channel))
            
            # Now rescale the image pixels between -1 and 1 to make them ready for unet input
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Now add the batch dimension
            input_img_tensor = input_img_tensor.unsqueeze(0)# (1,512,512,3)
            
            # Now change the order of tensor so it can fit the encoder
            input_img_tensor = input_img_tensor.permute(0,3,1,2)# (Batch_Size, Channel, Height, Width)
            
            #Sample some noise
            encoder_noise = torch.randn(latent_shape,generator=generator,device=device)# (Batch_Size, 4, Latents_Height, Latents_Width)
            
            # Run the image through decoder 
            # First create latents executing the encoder
            latents = encoder(input_image_tensor,encoder_noise)# (Batch_Size, 4, Latents_Height, Latents_Width)
            
            # Then add noise using scheduler and use strength parameter to adjust the amount of noise added 
            # Define strength
            sampler.set_strength(strength=strength)
            
            # Add noise
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
            
        # TEXT TO IMAGE    
        else:
            # First start with random noise 
            latents = torch.randn(latent_shape, generator=generator, device=device)
            
        # Load the diffusion model
        diffusion =models["diffusion"]
        diffusion.to(device)
        
        # Define timesteps
        # We define timesteps so scheduler can remove noise according to them and it defines by the number of inference steps
        timesteps = tqdm(sampler.timesteps)
        
        for i,timesteps in enumerate(timesteps):
            # Calculate time embedding
            time_embedding = get_time_embedding(timesteps).to(device)
            
            model_input = latents
            
            if do_cfg:
                model_input = model_input.repeat(2,1,1,1)# (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                
            # Predicted noise by UNET
            model_output = diffusion(model_input, context, time_embedding)# (Batch_Size, 4, Latents_Height, Latents_Width)
            
            # Enter another if block so we can split the output into 2 seperate batches 
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # Combine them according to formula
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Start to remove the noise thats predicted by UNET
            latents = sampler.step(timesteps, latents, model_output)
        to_idle(diffusion)
           
        # Load the decoder
        decoder = models["decoder"]
        decoder.to(device)
        #Run the latents through decoder 
        images = decoder(latents)
        to_idle(decoder)
        
        #Reverse the scaling we done before
        images = rescale(images, (-1, 1), (0, 255),clamp = True)
        #Reverse the order change to save it in cpu 
        images  =  images.permutate(0,2,3,1)# (Batch_Size, Height, Width, Channel)
        # Move the image to the CPU
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x
    
# Get the timestep and turn into a vector with size 320 using transformer function     
def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
           
            
                
            
        
            
        
            
            
            
            
            
            
        
    
        
        
       
        
 