import imageio

images = []
for i in range(0, 200):  
    filename = f"./vae_img_42/2/batch_image_{i}.png"#change ito other path if you want, like vae_img_42/1 
    images.append(imageio.imread(filename))

imageio.mimsave('2.gif', images, fps=20) 