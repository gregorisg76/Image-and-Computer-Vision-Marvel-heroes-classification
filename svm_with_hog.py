import os
import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from utils import read_img, get_image_paths, resize_img
from skimage import feature


if __name__ == '__main__':
    # with help from https://debuggercafe.com/image-recognition-using-histogram-of-oriented-gradients-hog-descriptor/
    DATA_PATH = 'data'
    IMAGE_CATEGORIES = [
        'black widow', 'captain america', 'doctor strange', 'hulk',
        'ironman', 'loki', 'spider-man', 'thanos'
    ]
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, IMAGE_CATEGORIES)

    # If the model is saved do not retrain
    if os.path.exists('svm_hog.joblib'):
        print('Loading existing linear SVM model...')
        svm = joblib.load('svm_hog.joblib')
    else:
        # Train the model
        train_images = []
        for image_path in train_image_paths:
            # read the image as grayscale
            img = read_img(image_path, mono=True)
            # resize the image
            img = resize_img(img, (112, 112))
            # get the HOG descriptor for the image
            hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
            # update the data
            train_images.append(hog_desc)
        print('Training on train images...')

        # #use grid search to tune the parameters.
        # with help from https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
        # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
        # grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
        # grid.fit(train_images, train_labels)
        # print(grid.best_estimator_)
        svm = SVC(C=10, gamma=0.1)
        svm.fit(train_images, train_labels)
        joblib.dump(svm, 'svm_hog.joblib')
        print('Done')

        # 5-fold cross validation
        cross_val_scores = cross_val_score(svm, train_images, train_labels, cv=5, scoring='f1_weighted')
        print(np.mean(cross_val_scores))

    # Test the model using testing data
    test_images = []
    for image_path in test_image_paths:
        # read the image as grayscale
        img = read_img(image_path, mono=True)
        # resize the image
        img = resize_img(img, (112, 112))
        # get the HOG descriptor for the image
        hog_desc = feature.hog(img, orientations=30, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

        # update the data
        test_images.append(hog_desc)

    # predict the labels of the set of images
    test_predictions = svm.predict(test_images)
    # calculate accuracy using f1_score
    accuracy = f1_score(test_labels, test_predictions, average='weighted')
    print('Classification accuracy of SVM with HOG features:', accuracy)
