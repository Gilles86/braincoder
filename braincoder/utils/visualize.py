import numpy as np


def show_animation(images, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure(figsize=(6, 4))
    ims = []

    if vmin is None:
        vmin = np.percentile(images, 5)

    if vmax is None:
        vmax = np.percentile(images, 95)

    for t in range(len(images)):
        im = plt.imshow(images[t], animated=True, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims, interval=150, blit=True, repeat_delay=1500)

    return HTML(ani.to_html5_video())
