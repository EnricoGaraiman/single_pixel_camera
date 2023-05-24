import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import datetime


# Generate the encoding matrix
def make_encode(height, width):
    while True:
        # Generate a random matrix in the range of 0 to 2, with dimensions (height^2, width^2) (36,36)
        encoded = 2 * np.random.rand(height ** 2, width ** 2)

        # Assign a specific value (white) if the pixel value is greater than a threshold, otherwise assign another value (black)
        ret, encoded = cv2.threshold(encoded, 0.5, 1, cv2.THRESH_BINARY)

        if cv2.determinant(encoded) != 0:
            break
    return encoded


# Root Mean Square Error
def RMSE(img1, img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif ** 2
    rmse = np.sqrt(np.sum(dif2) / n)
    return rmse


def main():
    start = datetime.datetime.now()

    # Read the image 
    path = r'E:\1. Documente Facultate\_Master SIVA Anul 1, SEM 2\ImCom\single_pixel_camera'
    img = cv2.imread(path + '/assets/' + 'strawberry.jpg', 0)
    img = cv2.resize(img, [8, 8])
    cv2.imwrite(path + "/results/results-random-mask/resize.jpg", img)

    height, width = img.shape[:2]

    # Reshape the image array from (height, width) to (height*width, 1) (36,1)
    img_array = img.reshape(height * width, 1)
    cv2.imwrite(path + "/results/results-random-mask/img_array.jpg", img_array)

    # Generate the encoding matrix (height, width) (36,36)
    encoded = make_encode(height, width)
    cv2.imwrite(path + "/results/results-random-mask/encoded.jpg", encoded)

    errors = []
    zero_error_iteration = None
    for i in tqdm(range(height * width)):
        # Select a subset of the encoding matrix, increasing the rows by 1 in each iteration
        tank = encoded[0:i + 1, :]

        # Calculate the pseudo-inverse of the selected encoding matrix
        mask_inv = np.linalg.pinv(tank)

        # Perform matrix multiplication between the selected encoding matrix and the input image array
        output_array = np.dot(tank, img_array)

        # Reconstruct the image by performing matrix multiplication between the pseudo-inverse and the output array
        reconstruct = np.dot(mask_inv, output_array)

        # Reshape the reconstructed array into the original image dimensions and convert the data type to "uint8"
        reimg = reconstruct.reshape(height, width).astype("uint8")

        if i % 10 == 0 or i + 1 == len(range(height * width)):
            cv2.imwrite(path + "/results/results-random-mask/reconstruct_{}.jpg".format(i), reimg)

        # Calculate the Root Mean Square Error between the input image array and the reconstructed array
        error = RMSE(img_array, reconstruct)

        # Store the error value in the list of errors for plotting later
        errors.append(error)

        if errors[i] < 0.0001 and zero_error_iteration is None:
            zero_error_iteration = i + 1

    print('Total time: ', (datetime.datetime.now() - start).total_seconds(), 's')

    print('Iteration number', len(errors))
    if zero_error_iteration is not None:
        print("Error 0 for", zero_error_iteration)
    else:
        print("Error is ", min(errors))
    plt.figure(figsize=(12, 8))
    plt.ylabel("RMSE", fontsize=25)
    plt.xlabel("iteration number", fontsize=25)
    plt.ylim(0, max(errors) + 10)
    plt.tick_params(labelsize=20)
    plt.grid(which='major', color='black', linestyle='-')
    plt.plot(np.arange(1, height * width + 1), errors)
    plt.savefig(path + "/results/results-random-mask/error_plot.png")


if __name__ == "__main__":
    main()
