
from sim_image_colorizer import Colorizer
from dataset import ImagesDataset
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_images_path = 'C:\\ws\\faces_sets\\faces_sets\\training_set'
    train_data = ImagesDataset(path=train_images_path)
    train_data.load(verbose=True)

    colorizer = Colorizer(train_data)
    colorizer.fit_data(gray_method='luminosity', verbose=True)

    test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
    test_data = ImagesDataset(path=test_images_path)
    test_data.load(verbose=True)

    sim_img, sim_gray_img, target_img = colorizer.colorize(test_data.images[0], verbose=True)

    plt.figure()
    plt.imshow(sim_img)
    
    plt.figure()
    plt.imshow(sim_gray_img, cmap='gray')

    plt.figure()
    plt.imshow(target_img)

    plt.show()