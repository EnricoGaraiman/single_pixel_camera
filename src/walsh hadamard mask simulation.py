import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm.auto import tqdm
import datetime

path = r'E:\1. Documente Facultate\_Master SIVA Anul 1, SEM 2\ImCom\single_pixel_camera'


def img_write(input_img, name):
    # Convert the image array to a binary image (0 or 255)
    input_img = np.where(input_img == 1, 255, 0)
    input_img = input_img.astype("uint8")
    cv2.imwrite(path + "/results/results-walsh-hadamard/" + name + ".jpg", input_img)


def make_hadamard(shape):
    # Check if the shape is a power of 2
    n = np.log(shape) / np.log(2)
    if 2 ** n == shape:
        print("making hadamard")
    else:
        print("error")
        return None, None

    hadamard_shape = [2 ** i for i in range(int(n) + 1)]
    print("hadamard shape\n{}".format(hadamard_shape))

    hadamard = np.array([1])
    hadamard_matrix = {1: hadamard}
    iteration = len(hadamard_shape) - 1
    for i in range(iteration):
        hadamard = np.hstack((hadamard, hadamard))
        hadamard = np.vstack((hadamard, hadamard))
        # Invert the values in the last (2,2) block for a (4,4) hadamard matrix
        reverse_range = -hadamard_shape[i]
        hadamard[reverse_range:, reverse_range:] = hadamard[reverse_range:, reverse_range:] * -1
        hadamard_matrix[hadamard_shape[i + 1]] = hadamard

    return hadamard_matrix


def change_count(shape, hadamard_matrix):
    encode = hadamard_matrix[shape]
    zero_crossing = np.zeros(shape)
    for h in range(shape):
        for w in range(shape - 1):
            if encode[h][w] != encode[h][w + 1]:
                zero_crossing[h] += 1
    return zero_crossing


def make_walsh_hadamard(shape, hadamard_matrix, zero_crossing):
    walsh_hadamard = hadamard_matrix[shape].copy()
    encode = hadamard_matrix[shape]

    # Sort the indexes based on the number of zero crossings in ascending order
    indexes = np.argsort(zero_crossing)

    for i, index in enumerate(indexes):
        walsh_hadamard[i] = encode[index]

    return walsh_hadamard


def RMSE(img1, img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif ** 2
    rmse = np.sqrt(np.sum(dif2) / n)
    return rmse


def simulation(img_array, walsh_hadamard, height, width):
    errors = []
    re_imges = {}
    zero_error_iteration = None
    for i in tqdm(range(height * width)):
        tank = walsh_hadamard[0:i + 1, :]
        # print("tank", tank.shape)

        # Calculate the pseudoinverse of the tank matrix
        mask_inv = np.linalg.pinv(tank)
        # print("mask_inv", mask_inv.shape)

        # Perform matrix multiplication between tank and img_array
        output_array = np.dot(tank, img_array)
        # print("output_array", output_array.shape)

        # Reconstruct the image using the pseudoinverse
        reconstruct = np.dot(mask_inv, output_array)
        # print("reconstruct", reconstruct.shape)

        # Reshape the reconstructed image and convert it to uint8
        reimg = reconstruct.reshape(height, width).astype("uint8")

        # Save the reconstructed image to disk
        if i % 10 == 0 or i + 1 == len(range(height * width)):
            cv2.imwrite(path + "/results/results-walsh-hadamard/reconstruct_{}.jpg".format(i), reimg)

        # Store the reconstructed image in the dictionary
        re_imges[i + 1] = reimg

        # Calculate the error between the original image and the reconstructed image
        error = RMSE(img_array, reconstruct)
        errors.append(error)

        if errors[i] < 0.0001 and zero_error_iteration is None:
            zero_error_iteration = i + 1
        # print(i)

    print('Iteration number', len(errors))
    if zero_error_iteration is not None:
        print("Error 0 for", zero_error_iteration)
    else:
        print("Error is ", min(errors))

    return errors, re_imges


def error_plot(errors):
    plt.ylabel("RMSE", fontsize=15)
    plt.xlabel("iteration number", fontsize=15)
    plt.ylim(0, max(errors) + 10)
    plt.plot(np.arange(1, len(errors) + 1), errors)
    plt.savefig(path + '/results/results-walsh-hadamard/errors.jpg')


def main():
    start = datetime.datetime.now()

    img = cv2.imread(path + '/assets/cat.jpg', 0)
    img = cv2.resize(img, [8,8])
    cv2.imwrite(path + "/results/results-walsh-hadamard/resize.jpg", img)

    print(img.shape)
    height, width = img.shape[:2]
    print("height: {}\nwidth: {}".format(height, width))

    shape = height * width
    img_array = img.reshape(height * width, 1)
    img_write(img_array, 'img_array')
    print(img_array.shape)

    # Create Hadamard matrix
    hadamard_matrix = make_hadamard(shape)
    img_write(hadamard_matrix, 'hadamard_matrix')

    # Count zero crossings for each row
    zero_crossing = change_count(shape, hadamard_matrix)
    print(len(zero_crossing))

    # Create Walsh-Hadamard matrix
    walsh_hadamard = make_walsh_hadamard(shape, hadamard_matrix, zero_crossing)

    name = "walsh_hadamard" + str(shape)
    img_write(walsh_hadamard, name)

    errors, re_imges = simulation(img_array, walsh_hadamard, height, width)
    print('Total time: ', (datetime.datetime.now() - start).total_seconds(), 's')

    error_plot(errors)


if __name__ == "__main__":
    main()
