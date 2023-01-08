from utils import get_image_paths, read_img, gaussian_blurring, \
    gaussian_pixel_noise, image_contrast_change, image_brightness_change, occlusion, salt_and_pepper_noise, \
    save_img

# Create a folder named "perturbations" in data folder
# Save all perturbed images in that folder in sub folders
# each sub folder named from A to H represents a type of perturbation in the order written in Assignment
# Inside each sub folder there are sub sub folders from 0 to 9 representing the parameter used in each perturbation
# in the order written in Assignment.
# Each of these contain the 8 superhero folders with the corresponding perturbed images.

if __name__ == '__main__':

    DATA_PATH = 'data'
    IMAGE_CATEGORIES = [
        'black widow', 'captain america', 'doctor strange', 'hulk',
        'ironman', 'loki', 'spider-man', 'thanos'
    ]

    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, IMAGE_CATEGORIES)

    # Gaussian pixel noise
    par_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = gaussian_pixel_noise(img, i)
            save_img(img, "data/perturbations/A/" + str(index) + img_path[9:])
        index += 1

    # Gaussian blurring
    par_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = gaussian_blurring(img, i)
            save_img(img, "data/perturbations/B/" + str(index) + img_path[9:])
        index += 1

    # Image contrast increase
    par_list = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20, 1.25]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = image_contrast_change(img, i)
            save_img(img, "data/perturbations/C/" + str(index) + img_path[9:])
        index += 1

    # Image contrast decrease
    par_list = [1, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = image_contrast_change(img, i)
            save_img(img, "data/perturbations/D/" + str(index) + img_path[9:])
        index += 1

    # Image brightness increase
    par_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = image_brightness_change(img, i)
            save_img(img, "data/perturbations/E/" + str(index) + img_path[9:])
        index += 1

    # Image brightness decrease
    par_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = image_brightness_change(img, -i)
            save_img(img, "data/perturbations/F/" + str(index) + img_path[9:])
        index += 1

    # Occlusion of the image increase plot
    par_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = occlusion(img, i)
            save_img(img, "data/perturbations/G/" + str(index) + img_path[9:])
        index += 1

    # Salt and pepper noise
    par_list = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    index = 0
    for i in par_list:
        for img_path in test_image_paths:
            img = read_img(img_path)
            img = salt_and_pepper_noise(img, i)
            save_img(img, "data/perturbations/H/" + str(index) + img_path[9:])
        index += 1
