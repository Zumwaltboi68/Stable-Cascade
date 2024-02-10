---
pipeline_tag: text-to-image
license: other
license_name: stable-cascade-nc-community
license_link: LICENSE
---

# Stable Cascade Text-to-Image Model Card

<!-- Provide a quick summary of what the model is/does. -->
![image]()

Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it. 

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


## Evaluation

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