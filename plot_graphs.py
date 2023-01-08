import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import feature
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from utils import read_img, resize_img, get_test_image_paths

if __name__ == '__main__':
    # Setup for classical method
    IMAGE_CATEGORIES = [
        'black widow', 'captain america', 'doctor strange', 'hulk',
        'ironman', 'loki', 'spider-man', 'thanos'
    ]

    # Load model
    classical_model = joblib.load('svm_hog.joblib')

    ############################
    # Setup for ResNet 18 method
    # Enable GPU support if available and load data into iterable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}

    # Transform used in training as well with no pre-augmentation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load saved weights and params:
    best_params = torch.load("best-model-weights")

    # Load Resnet with best weights
    model = models.resnet18()

    # Freeze all but fully connected layer
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    NUM_CLASSES = 8

    # Adjust output to 8 classes
    feature_number = model.fc.in_features
    model.fc = nn.Linear(feature_number, NUM_CLASSES)

    model.load_state_dict(best_params["model_state_dict"])
    model = model.to(device)

    ##################
    # Plot the graphs
    # classical_model is the model using hog features
    # model is the model using Resnet18

    # In each for-loop "i" refers to the index of the parameter used in each perturbation
    # cm refers to classical method
    # nn refers to neural network method (RESNET 18)

    # Gaussian pixel noise plot (use sub folder "A")
    gaussian_pixel_noise_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    cm_gaussian_pixel_noise_accuracy_list = []
    nn_gaussian_pixel_noise_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/A/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_gaussian_pixel_noise_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_gaussian_pixel_noise_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(gaussian_pixel_noise_list, cm_gaussian_pixel_noise_accuracy_list, marker="o", label="Classical method")
    plt.plot(gaussian_pixel_noise_list, nn_gaussian_pixel_noise_accuracy_list, marker="o", label="Deep learning method")
    plt.legend()
    plt.xlabel("Standard deviation of Gaussian distribution")
    plt.ylabel("F1 score")
    plt.title("Gaussian pixel noise")
    plt.savefig("gaussian_pixel_noise.jpg")
    plt.show()

    # Gaussian blurring plot (use sub folder "B")
    gaussian_blurring_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cm_gaussian_blurring_accuracy_list = []
    nn_gaussian_blurring_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/B/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_gaussian_blurring_accuracy_list.append(accuracy)
        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_gaussian_blurring_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(gaussian_blurring_list, cm_gaussian_blurring_accuracy_list, marker="o", label="Classical method")
    plt.plot(gaussian_blurring_list, nn_gaussian_blurring_accuracy_list, marker="o", label="Deep learning method")
    plt.legend()
    plt.xlabel("Times image convolved with gaussian blurring mask")
    plt.ylabel("F1 score")
    plt.title("Gaussian blurring")
    plt.savefig("gaussian_blurring.jpg")
    plt.show()

    # Image contrast increase plot (use sub folder "C")
    image_contrast_increase_list = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20, 1.25]
    cm_image_contrast_increase_accuracy_list = []
    nn_image_contrast_increase_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/C/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_image_contrast_increase_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_image_contrast_increase_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(image_contrast_increase_list, cm_image_contrast_increase_accuracy_list, marker="o",
             label="Classical method")
    plt.plot(image_contrast_increase_list, nn_image_contrast_increase_accuracy_list, marker="o",
             label="Deep learning method")
    plt.legend()
    plt.xlabel("Scaling factor of each pixel")
    plt.ylabel("F1 score")
    plt.title("Image Contrast Increase")
    plt.savefig("image_contrast_increase.jpg")
    plt.show()

    # Image contrast decrease plot (use sub folder "D")
    image_contrast_decrease_list = [1, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    cm_image_contrast_decrease_accuracy_list = []
    nn_image_contrast_decrease_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/D/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_image_contrast_decrease_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_image_contrast_decrease_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(image_contrast_decrease_list, cm_image_contrast_decrease_accuracy_list, marker="o",
             label="Classical method")
    plt.plot(image_contrast_decrease_list, nn_image_contrast_decrease_accuracy_list, marker="o",
             label="Deep learning method")
    plt.legend()
    plt.xlabel("Scaling factor of each pixel")
    plt.ylabel("F1 score")
    plt.title("Image Contrast Decrease")
    plt.savefig("image_contrast_decrease.jpg")
    plt.show()

    # Image brightness increase plot (use sub folder "E")
    image_brightness_increase_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cm_image_brightness_increase_accuracy_list = []
    nn_image_brightness_increase_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/E/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_image_brightness_increase_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_image_brightness_increase_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(image_brightness_increase_list, cm_image_brightness_increase_accuracy_list, marker="o",
             label="Classical method")
    plt.plot(image_brightness_increase_list, nn_image_brightness_increase_accuracy_list, marker="o",
             label="Deep learning method")
    plt.legend()
    plt.xlabel("Magnitude of addition to each pixel")
    plt.ylabel("F1 score")
    plt.title("Image Brightness Increase")
    plt.savefig("image_brightness_increase.jpg")
    plt.show()

    # Image brightness decrease plot (use sub folder "F")
    image_brightness_decrease_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cm_image_brightness_decrease_accuracy_list = []
    nn_image_brightness_decrease_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/F/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_image_brightness_decrease_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_image_brightness_decrease_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(image_brightness_decrease_list, cm_image_brightness_decrease_accuracy_list, marker="o",
             label="Classical method")
    plt.plot(image_brightness_decrease_list, nn_image_brightness_decrease_accuracy_list, marker="o",
             label="Deep learning method")
    plt.legend()
    plt.xlabel("Magnitude of subtraction to each pixel")
    plt.ylabel("F1 score")
    plt.title("Image Brightness Decrease")
    plt.savefig("image_brightness_decrease.jpg")
    plt.show()

    # Occlusion of the image increase plot (use sub folder "G")
    occlusion_increase_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cm_occlusion_increase_accuracy_list = []
    nn_occlusion_increase_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/G/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_occlusion_increase_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_occlusion_increase_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(occlusion_increase_list, cm_occlusion_increase_accuracy_list, marker="o", label="Classical method")
    plt.plot(occlusion_increase_list, nn_occlusion_increase_accuracy_list, marker="o", label="Deep learning method")
    plt.legend()
    plt.xlabel("Square edge length")
    plt.ylabel("F1 score")
    plt.title("Occlusion of the Image Increase")
    plt.savefig("occlusion.jpg")
    plt.show()

    # Salt and pepper noise plot (use sub folder "H")
    salt_and_pepper_noise_list = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    cm_salt_and_pepper_noise_accuracy_list = []
    nn_salt_and_pepper_noise_accuracy_list = []
    for i in range(10):
        DATA_PATH = "data/perturbations/H/" + str(i)

        # Classical method
        test_image_paths, test_labels = \
            get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES)

        perturbed_test_images = []
        for img_path in test_image_paths:
            img = read_img(img_path, mono=True)
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data
            perturbed_test_images.append(hog_desc)

        perturbed_test_predictions = classical_model.predict(perturbed_test_images)
        accuracy = f1_score(test_labels, perturbed_test_predictions, average='weighted')
        cm_salt_and_pepper_noise_accuracy_list.append(accuracy)

        # Res-Net 18 method
        """ Test """
        model.eval()
        # test sets
        test_ds = datasets.ImageFolder(DATA_PATH, transform=test_transform)

        test_loader = DataLoader(
            test_ds,
            batch_size=test_ds.__len__(),
            shuffle=False,
            num_workers=8,
            **kwargs
        )

        with torch.no_grad():
            # Transfer data to GPU if available
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)

        f1_test = f1_score(labels.data, preds, average="weighted")
        nn_salt_and_pepper_noise_accuracy_list.append(f1_test)
        print(f1_test)

    plt.plot(salt_and_pepper_noise_list, cm_salt_and_pepper_noise_accuracy_list, marker="o",
             label="Classical method")
    plt.plot(salt_and_pepper_noise_list, nn_salt_and_pepper_noise_accuracy_list, marker="o",
             label="Deep learning method")
    plt.legend()
    plt.xlabel("Salt and pepper noise strength")
    plt.ylabel("F1 score")
    plt.title("Salt and Pepper Noise")
    plt.savefig("salt_and_pepper_noise.jpg")
    plt.show()
