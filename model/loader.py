from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import converter

def preload_models_from_standard_weights(ckpt_path,device):
    # Load the weights
    state_dict = converter.load_from_standard_weights(ckpt_path,device)
    
    # Load the encoder
    encoder = VAE_Encoder().to(device)
    # This makes it sure that all the names map 
    encoder.load_state_dict(state_dict["encoder"], strict=True)
    
    # Load the decoder
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"],strict=True)
    
    # Load the diffusion model 
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"],strict=True)
    
    # Load the CLIP 
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"],strict=True)
    
    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
    
