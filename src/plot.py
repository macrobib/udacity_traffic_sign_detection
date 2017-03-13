"""Plotting support APIs."""
import matplotlib as plt
import random

class plot:
    """Plot utility."""
    def __init__(self, data=None):
        self.data = data

    def plot_data(self):
        """plot data."""
        pass

    def plot_image(self, image):
        """Display given images."""
        plt.figure(figsize=(1, 1))
        plt.imshow(image)
        plt.show()

    def visualize_data(self, x_data, y_data):
        figure, ax = plt.subplots(figsize=(20, 40))
        ylen = np.arange(len(y_data))

        ax.barh(ylen, x_data, align='center', color='green')
        ax.yticklabels(y_data)
        ax.set_xlabel('No of Images')
        ax.set_title('Image distribution')
        figure.subplots_adjust(lef=0.12)
        plt.show()

    def sample_data(self,data, label, size):
        """Randomly render a set of images from given data."""
        val = random.randint(1, len(data))
        random.shuffle(val)
        val = val[:size]
        data_sample = data[val]
        label_sample = label[val]



