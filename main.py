from diffusers import DiffusionPipeline
import os
import time

# Create 'output' folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load the diffusion model pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("mps")

def generate_image(prompt: str):
    # Generate the image using the diffusion pipeline
    results = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    
    # Save the image and the prompt
    timestamp = time.strftime("%Y%m%d%H%M%S")
    image_path = f"output/image_{timestamp}.png"
    results.images[0].save(image_path)

    with open(f"output/image_{timestamp}_prompt.txt", "w") as f:
        f.write(prompt)
    
    # Return the generated image
    return results.images[0]

# Main loop to get user input and generate images
while True:
    prompt = input("Enter a prompt: ")
    if prompt.lower() == "exit":
        break
    image = generate_image(prompt)
    image.show()
