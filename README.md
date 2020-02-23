# Computationally Efficient, Accurate and Real-Time Detection of Colon Polyps Using Deep Neural Networks
#Aalok Patwa
#Presented to Synopsys Science Fair, Silicon Valley, 2019

Colorectal cancer is the second leading cause of cancer-related death in the United States, and early detection of
precancerous polyps (adenomas) in the colon increases the chance of survival. However, the Adenoma Detection
Rate (ADR) during screening colonoscopies varies between 7-53%, and the likelihood of missed colorectal cancer
increases dramatically when ADR is < 20%. Computer vision models can aid in identification of both precancerous
and cancerous polyps, increase the ADR, and save lives!
I had three design criteria:
1) The model should detect polyps of all different shapes, sizes, and colors and increase ADR.
2) As publicly available colon polyp images are scarce, the model needs to be trainable on a limited dataset.
3) The model should have the least possible parameters and computations such that detection can occur in
real-time during colonoscopy.

As the base of my model, I modified UNet, a Fully-Convolutional neural network composed of a series of
contracting steps (encoder) followed by expansive steps (decoder) that creates a semantic segmentation map for
the input image. This architecture allowed for rapid training (< 4 hours) even on a small dataset; the
encoder-decoder system preserved spatial knowledge, skip (residual) connections prevented exploding and
vanishing gradients, and stacked convolutions reduced the number of parameters required.

I acquired datasets of colonoscopy images from GIANA and augmented the images by performing elastic
deformations and applying random rotations, flips, zooms, and translations in real time. This increased dataset size
and diversity, making the model robust to variations in polyp shape, size, and color.

I experimented with various model hyperparameters. I initialized parameters from the He Normal distribution, used
Nearest Neighbor interpolation for upsampling in the encoder, implemented a 0.375 dropout rate in two layers,
instituted a learning rate scheduler, chose binary cross-entropy as my loss function, and utilized the Adam
optimizer. I discovered that a batch size of 1 was required for effective training. The chosen hyperparameters
enhanced training time, accuracy, and prediction latency.

After a series of modifications, my final model, which featured an extra convolutional layer per stack in the encoder
and decoder and a 25% reduction in number of filters per layer, yielded a state-of-the-art Dice Score of 0.445 using
~23.5 million parameters. It achieved a latency of 60 ms and a frame rate of ~17 FPS.

I met all three of my design criteria. My model accurately detected polyps of all different shapes, sizes, and colors
and was resilient to variation in image quality. My model was trainable on a dataset of 315 images, a miniscule
amount compared to other object detection models. Finally, my model’s computational efficiency induced low
latency, allowing for real-time detection from colonoscopy video streams.

Thus I created a novel neural network architecture that accurately identifies colorectal polyps in colonoscopies. The
model achieves high Dice Scores on challenging datasets, all while being 84% smaller and using 96% less training
images than existing solutions. My model can increase ADR, identify polyps, and nip colorectal cancer in the bud.
Nobody likes preparing for a colonoscopy, but using this network, the procedure could improve vastly!
