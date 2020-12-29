# StarGAN
This is an independent project in which I attempt to reproduce the results from [1] on the celebA dataset. Similar to my Season Transfer project, this type of image
translation takes mutliple domains that share the same content, but differ in style. Using the annotations of the celebA dataset, I seek to be able to translate photos of people
between/amongst 5 binary attributes: Black hair, Blonde hair, Brown hair, Male/Female, Young/Old. Indeed this type of mapping is much more complex than the two way mapping
between summer and fall from season transfer. I am using a much bigger dataset, which also includes annotations for the identity of the subject of each photo.

In adhering the authors' methods, I am using the same resnet generator architecture used in [2]. The discriminator is rather deep, with the last layer containing 2048 feature
maps. All images were processed and produced at 128x128 resolution. Training required nearly 3 hours per epoch. I needed to slightly deviate from the authors' original
training method since their specified training protocol and architecture did not lead to stable losses, i.e. the discriminator runs away with the adversarial game and
classfication error continually rises on the generator. Thus I added R1 regularization [3], whose gamma parameter started at 0.5 and was decayed/multiplied by 0.9 every epoch.
The learning rate was 1e-4 (Adam) for both the generator and discriminator for the first 10 epochs, and then decayed/multiplied by 0.9 for the final 10 epochs. Also I used
binary cross entropy for the adversarial loss instead of WGAN-GP.

Results are decent, but could be improved. Hair color seems to be an easy translation, while gender and age tend to be more scattershot. One issue may have been that the
regularization parameter gamma was too high, i.e. hindered the discriminator too much such that the generator could get away with garbage. See results for example pictures
and a text file with the score evolution throughout training.

[1] Choi, Y., Choi, M., Kim, M., Ha, J., Kim, S., Choo, J. (2018) StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation
[2] Zhu, Y., Park, T., Isola, P., Efros A.A. (2018) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[3] Roth, K., Lucchi, A., Nowozin, S., Hofmann, T. (2017) Stabilizing Training of Generative Adversarial Networks through Regularization
