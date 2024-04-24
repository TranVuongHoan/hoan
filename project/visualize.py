from math import sqrt
import matplotlib.pyplot as plt

def visualize_digit(image, filename=None, reshape=True):
    if len(image.shape) == 1 and reshape:
        ori_shape = image.shape[0]
        new_shape = int(sqrt(ori_shape))
        image = image.reshape((new_shape, new_shape))
    plt.gray()
    plt.imshow(image)
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
    return None


def visualize_multiple_digits(digits, labels, filename=None):
    assert len(digits) == len(labels), \
        f"Assert length of digits equal to length of labels: {len(digits)} != {len(labels)}"
    figure, axes = plt.subplots(nrows=1, ncols=len(digits), figsize=(8, 8))
    for ax, image, label in zip(axes, digits, labels):
        if len(image.shape) == 1:
            ori_shape = image.shape[0]      
            new_shape = int(sqrt(ori_shape))
            image = image.reshape((new_shape, new_shape))
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Digit: %i" % label)
    if filename != None:
        figure.savefig(filename)
    else:
        figure.show()
    return None