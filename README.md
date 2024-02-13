---
pipeline_tag: text-to-image
license: other
license_name: stable-cascade-nc-community
license_link: LICENSE
---

# Stable Cascade

<!-- Provide a quick summary of what the model is/does. -->
<img src="figures/collage_1.jpg" width="800">

This model is built upon the [Würstchen](https://openreview.net/forum?id=gU58d5QeGv) architecture and its main 
difference to other models like Stable Diffusion is that it is working at a much smaller latent space. Why is this 
important? The smaller the latent space, the **faster** you can run inference and the **cheaper** the training becomes. 
How small is the latent space? Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being 
encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a 
1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the 
highly compressed latent space. Previous versions of this architecture, achieved a 16x cost reduction over Stable 
Diffusion 1.5. <br> <br>
Therefore, this kind of model is well suited for usages where efficiency is important. Furthermore, all known extensions
like finetuning, LoRA, ControlNet, IP-Adapter, LCM etc. are possible with this method as well.

## Model Details

### Model Description

Stable Cascade is a diffusion model trained to generate images given a text prompt.

- **Developed by:** Stability AI
- **Funded by:** Stability AI
- **Model type:** Generative text-to-image model

### Model Sources

For research purposes, we recommend our `StableCascade` Github repository (https://github.com/Stability-AI/StableCascade).

- **Repository:** https://github.com/Stability-AI/StableCascade
- **Paper:** https://openreview.net/forum?id=gU58d5QeGv

### Model Overview
Stable Cascade consists of three models: Stage A, Stage B and Stage C, representing a cascade to generate images,
hence the name "Stable Cascade".
Stage A & B are used to compress images, similar to what the job of the VAE is in Stable Diffusion. 
However, with this setup, a much higher compression of images can be achieved. While the Stable Diffusion models use a 
spatial compression factor of 8, encoding an image with resolution of 1024 x 1024 to 128 x 128, Stable Cascade achieves 
a compression factor of 42. This encodes a 1024 x 1024 image to 24 x 24, while being able to accurately decode the 
image. This comes with the great benefit of cheaper training and inference. Furthermore, Stage C is responsible 
for generating the small 24 x 24 latents given a text prompt. The following picture shows this visually.

<img src="figures/model-overview.jpg" width="600">

For this release, we are providing two checkpoints for Stage C, two for Stage B and one for Stage A. Stage C comes with 
a 1 billion and 3.6 billion parameter version, but we highly recommend using the 3.6 billion version, as most work was 
put into its finetuning. The two versions for Stage B amount to 700 million and 1.5 billion parameters. Both achieve 
great results, however the 1.5 billion excels at reconstructing small and fine details. Therefore, you will achieve the 
best results if you use the larger variant of each. Lastly, Stage A contains 20 million parameters and is fixed due to 
its small size.

## Evaluation
<img height="300" src="figures/comparison.png"/>
According to our evaluation, Stable Cascade performs best in both prompt alignment and aesthetic quality in almost all 
comparisons. The above picture shows the results from a human evaluation using a mix of parti-prompts (link) and 
aesthetic prompts. Specifically, Stable Cascade (30 inference steps) was compared against Playground v2 (50 inference 
steps), SDXL (50 inference steps), SDXL Turbo (1 inference step) and Würstchen v2 (30 inference steps).

## Code Example

**⚠️ Important**: For the code below to work, you have to install `diffusers` from this branch while the PR is WIP.

```shell
pip install git+https://github.com/kashif/diffusers.git@wuerstchen-v3
```

```python
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

device = "cuda"
num_images_per_prompt = 2

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to(device)

prompt = "Anthropomorphic cat dressed as a pilot"
negative_prompt = ""

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20
)
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.half(),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images

#Now decoder_output is a list with your PIL images
```

## Uses

### Direct Use

The model is intended for research purposes for now. Possible research areas and tasks include

- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, 
and therefore using the model to generate such content is out-of-scope for the abilities of this model.
The model should not be used in any way that violates Stability AI's [Acceptable Use Policy](https://stability.ai/use-policy).

## Limitations and Bias

### Limitations
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.


### Recommendations

The model is intended for research purposes only.

## How to Get Started with the Model

Check out https://github.com/Stability-AI/StableCascade