import numpy as np


def show_animation(images):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure(figsize=(6, 4))
    ims = []

    for t in range(len(images)):
        im = plt.imshow(images[t], animated=True, cmap='gray', vmin=np.percentile(
            images, 5), vmax=np.percentile(images, 95))
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims, interval=150, blit=True, repeat_delay=1500)

    return HTML(ani.to_html5_video())
