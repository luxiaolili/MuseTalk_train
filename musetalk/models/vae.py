from diffusers import AutoencoderKL
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
from musetalk.utils.lmk_pic import DrawLandmark

class VAE():
    """
    VAE (Variational Autoencoder) class for image processing.
    """

    def __init__(self, model_path="./models/sd-vae-ft-mse/", resized_img=256, use_float16=False):
        """
        Initialize the VAE instance.

        :param model_path: Path to the trained model.
        :param resized_img: The size to which images are resized.
        :param use_float16: Whether to use float16 precision.
        """
        self.model_path = model_path
        self.vae = AutoencoderKL.from_pretrained(self.model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)
        self.draw = DrawLandmark(is_test=True)
        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False

        self.scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )   
        self._resized_img = resized_img
        self._mask_tensor = self.get_mask_tensor()
        
    def get_mask_tensor(self):
        """
        Creates a mask tensor for image processing.
        :return: A mask tensor.
        """
        mask_tensor = torch.zeros((self._resized_img,self._resized_img))
        mask_tensor[:self._resized_img//2,:] = 1
        mask_tensor[mask_tensor< 0.5] = 0
        mask_tensor[mask_tensor>= 0.5] = 1
        return mask_tensor
            

    def preprocess_img(self,img_name,half_mask=False):
        """
        Preprocess an image for the VAE.

        :param img_name: The image file path or a list of image file paths.
        :param half_mask: Whether to apply a half mask to the image.
        :return: A preprocessed image tensor.
        """
        window = []
        
        if isinstance(img_name, str):
            window_fnames = [img_name]
            for fname in window_fnames:
                img = self.transform(Image.open(fname).convert('RGB').resize((self._resized_img, self._resized_img)))
                if half_mask:
                    mask = torch.zeros(img.shape[1], img.shape[2])
                    mask[:img.shape[2]//2,:] = 1
                    mask[mask < 0.5] = 0
                    mask[mask >= 0.5] = 1
                    img = img * (mask > 0.5)
                window.append(img)
        else:
            img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img.resize((self._resized_img, self._resized_img)))
            if half_mask:
                mask = torch.zeros(img.shape[1], img.shape[2])
                mask[:img.shape[2]//2,:] = 1
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                img = img * (mask > 0.5)
            window.append(img)
        imgs = torch.stack([img for img in window])
        imgs = imgs.to(self.vae.device) 

        return imgs

    def encode_latents(self,image):
        """
        Encode an image into latent variables.

        :param image: The image tensor to encode.
        :return: The encoded latent variables.
        """
        with torch.no_grad():
            init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents
    
    def decode_latents(self, latents):
        """
        Decode latent variables back into an image.
        :param latents: The latent variables to decode.
        :return: A NumPy array representing the decoded image.
        """
        latents = (1/  self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[...,::-1] # RGB to BGR
        return image
 
    
    def get_latents_for_unet(self,img,lmk=False):
        """
        Prepare latent variables for a U-Net model.
        :param img: The image to process.
        :return: A concatenated tensor of latents for U-Net input.
        """
        
        ref_image = self.preprocess_img(img,half_mask=True) # [1, 3, 256, 256] RGB, torch tensor
        masked_latents = self.encode_latents(ref_image) # [1, 4, 32, 32], torch tensor
        ref_image = self.preprocess_img(img,half_mask=False) # [1, 3, 256, 256] RGB, torch tensor
        ref_latents = self.encode_latents(ref_image) # [1, 4, 32, 32], torch tensor
        mask = self._mask_tensor.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(ref_latents.shape[2], ref_latents.shape[3])).to(ref_latents.device)
       
        if lmk:
            lmk_img = self.draw.draw_lmk(img)
            lmk_img = self.preprocess_img(img,half_mask=False) 
            lmk_latents = self.encode_latents(lmk_img)
            latent_model_input = torch.cat([masked_latents,lmk_latents, ref_latents], dim=1)
        else:
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
        return latent_model_input

if __name__ == "__main__":
    vae_mode_path = "./models/sd-vae-ft-mse/"
    vae = VAE(model_path = vae_mode_path,use_float16=False)
    img_path = "./results/sun001_crop/00000.png"
    
    crop_imgs_path = "./results/sun001_crop/"
    latents_out_path = "./results/latents/"
    if not os.path.exists(latents_out_path):
        os.mkdir(latents_out_path)

    files = os.listdir(crop_imgs_path)
    files.sort()
    files = [file for file in files if file.split(".")[-1] == "png"]

    for file in files:
        index = file.split(".")[0]
        img_path = crop_imgs_path + file
        latents = vae.get_latents_for_unet(img_path)
        print(img_path,"latents",latents.size())
        #torch.save(latents,os.path.join(latents_out_path,index+".pt"))
        #reload_tensor = torch.load('tensor.pt')
        #print(reload_tensor.size())