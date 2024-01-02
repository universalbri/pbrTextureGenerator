import sys
n = len(sys.argv)
if n <= 3: print("Usage: pbrMat.py <texture description> <texture filename> <number of generations>"), exit()

from diffusers import StableDiffusionPipeline
import torch

model_id = "dream-textures/texture-diffusion"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

iterations = int( sys.argv[3] )
print(f'Number of Images to Create={iterations}')
fileName = sys.argv[2]
prompt = sys.argv[1]
count = 0
while (count < iterations):
    image = pipe(prompt).images[0]
    saveFileNameFull = f'{fileName}_{count}.png'
    image.save( saveFileNameFull )
    print(f'Saved image {saveFileNameFull}')
    count = count + 1
