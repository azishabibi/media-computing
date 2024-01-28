import imageio

images = []
for i in range(0, 500):  
    filename = f"./generated_images/gen_{i}.png"
    images.append(imageio.imread(filename))

imageio.mimsave('mov.gif', images, fps=20)
