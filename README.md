# Prisma-esque
A Deep Learning Style transfer algorithm

## Algorithm:
We use the VGG16 pretrained architecture, and extract the information at "block2_conv2" and minimize the content loss. We iterate the evaluator a number of times in order to get the perfect style transfer image.

Input Image:

![input_image](https://i.imgur.com/zCKOYAh.jpg "input")


Style Image:

![style_image](https://i.imgur.com/hSBcIqd.jpg "style")


Output Image:

![output_image](https://i.imgur.com/WgQgVzj.jpg "output")

**TODO**: Experiment with extraction layers. 
