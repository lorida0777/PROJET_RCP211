import torch
from diffusers import AutoencoderKL
from PIL import Image
import requests # Pour télécharger une image d'exemple

# --- ÉTAPE 1: CHARGER LE MODÈLE VAE PRÉ-ENTRAÎNÉ ---
# On utilise un VAE qui fait partie du célèbre modèle "Stable Diffusion".
# Il a été entraîné sur des millions d'images et est très performant.
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

print("Modèle VAE chargé avec succès sur le device:", device)

# --- ÉTAPE 2: PRÉPARER LES IMAGES ---
# Pour ce test, nous allons simuler les images.
# Dans votre projet, vous chargerez vos 3 images de CelebA ici.
# Par exemple:
# image_man_glasses = Image.open("path/to/man_with_glasses.jpg").convert("RGB")
# image_man_no_glasses = Image.open("path/to/man_without_glasses.jpg").convert("RGB")
# image_woman_no_glasses = Image.open("path/to/woman_without_glasses.jpg").convert("RGB")

# Pour que le code fonctionne directement, téléchargeons des exemples :
def download_image(url):
    return Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((512, 512))

# Remplacez ces URLs par les chemins de vos images CelebA
image_man_glasses = download_image("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/celeb_a_1.png")
image_man_no_glasses = download_image("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/celeb_a_2.png")
image_woman_no_glasses = download_image("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/celeb_a_3.png")

# Cette fonction va transformer une image en tenseur pour le modèle
def preprocess_image(image):
    image = torch.tensor(image.getdata()).float().reshape(image.size[0], image.size[1], 3)
    image = image.permute(2, 0, 1) / 255.0
    return image.unsqueeze(0).to(device) * 2.0 - 1.0 # Normaliser entre -1 et 1

# --- ÉTAPE 3: ENCODER LES IMAGES EN VECTEURS LATENTS ---
@torch.no_grad()
def encode_image_to_latents(img):
    img_tensor = preprocess_image(img)
    # L'encodeur retourne une distribution, on prend la moyenne (.mean) comme vecteur
    latent_distribution = vae.encode(img_tensor).latent_dist
    return latent_distribution.mean

# Obtenir les 3 vecteurs latents
z_man_glasses = encode_image_to_latents(image_man_glasses)
z_man_no_glasses = encode_image_to_latents(image_man_no_glasses)
z_woman_no_glasses = encode_image_to_latents(image_woman_no_glasses)

print("Forme d'un vecteur latent:", z_man_glasses.shape)

# --- ÉTAPE 4: FAIRE L'ARITHMÉTIQUE DANS L'ESPACE LATENT ---
# C'est le cœur du projet !
z_result = z_man_glasses - z_man_no_glasses + z_woman_no_glasses

# --- ÉTAPE 5: DÉCODER LE VECTEUR RÉSULTANT EN IMAGE ---
@torch.no_grad()
def decode_latents_to_image(latents):
    # Le VAE a un facteur de mise à l'échelle, on l'applique ici
    latents = latents / vae.config.scaling_factor
    image_tensor = vae.decode(latents).sample
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1) # Dénormaliser
    image_pil = Image.fromarray((image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    return image_pil

# Générer l'image finale
result_image = decode_latents_to_image(z_result)
result_image.save("resultat_femme_avec_lunettes_VAE.jpg")
print("L'image résultat a été sauvegardée !")

# Sauvegarder aussi les images de départ pour le rapport
image_man_glasses.save("input_1_man_glasses.jpg")
image_man_no_glasses.save("input_2_man_no_glasses.jpg")
image_woman_no_glasses.save("input_3_woman_no_glasses.jpg")