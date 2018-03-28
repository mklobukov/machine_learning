#Mark Klobukov
#CS 383
#HW 1 Q3
#Eigenfaces
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
from scipy.misc import imresize

ALPHA = 0.95
pathToImages = './yalefaces/'

def main():
    images, original = loadAndPrepareImages(pathToImages)

    one = np.copy(images[0])
    one_original = np.copy(one)
    one = np.reshape(one, (40,40))

    images_std = standardizeData(images)
    eigenval, eigenvec = findEigen(images_std)

    firstEigen = eigenvec[:,0]
    #find k
    eigenval = np.matrix(eigenval)
    #sort the eigenvalues
    k = findK(eigenval, ALPHA)
    # k = 1600
    print "The value of K to represent 95 percent of info: ", k

    proj_matrix = createProjectionMatrix(eigenval, eigenvec, k)
    firstPC = proj_matrix[:,0]

    one = images[0]

    ####Q1
    print "Displaying Visualization of First Principal Component"
    plotFirstPC(proj_matrix)
    plt.title("First Principal Component")
    plt.show()

    ###Q2
    print "Displaying Original Image #1"
    plt.imshow(np.reshape(one_original, (40,40)), cmap='gray')
    plt.title("Original Image #1")
    plt.show()

    ###Q3
    print "Displaying Reconstruction of the Image from Single PC"
    kpc = projectOntoKComponents(one, proj_matrix, 1)
    recon = reconstruct(kpc, proj_matrix, 1)
    plt.imshow(recon, cmap='gray')
    plt.title("Reconstruction from Single Principal Component")
    plt.show()

    ###Q4
    print "Displaying Reconstruction of the Image from k PCs"
    kpc = projectOntoKComponents(one, proj_matrix, k)
    reconK = reconstruct(kpc, proj_matrix, k)
    plt.imshow(reconK, cmap='gray')
    plt.title("Reconstruction from k=33 Principal Components")
    plt.show()

def projectOntoKComponents(image, proj_matrix, k):
    #grab k components of proj matrix
    comp = proj_matrix[:, 0:k]

    #resize the image to (40,40) and flatten to (1,1600)
    image = np.copy(image)
    # image = imresize(image, (40,40))
    # image = image.flatten()
    image = np.reshape(image, (1,1600))

    projected = np.dot(image, comp)
    return projected

def reconstruct(projected_image, proj_matrix, k):
    #grap k components
    comp = proj_matrix[:, 0:k]

    rec = np.dot(projected_image, np.transpose(comp))
    rec = np.reshape(rec, (40,40))
    return rec

def plotFirstPC(proj_matrix):
    pc = proj_matrix[:,0]
    pc = np.reshape(pc, (40,40))
    plt.imshow(pc, cmap='gray')

def projectOntoFirstPC(image, proj_matrix):
    #grab PC -- first col of proj matrix
    pc = proj_matrix[:,0]

    #resize the image to (40,40) and flatten to(1,1600)
    image = np.copy(image)
    # image = imresize(image, (40,40))
    # image = image.flatten()
    image = np.reshape(image, (1,1600))
    pc = np.reshape(pc, (1600,1))

    projected = np.dot(image, pc)

    #reconsruct: mult projected by transpose of pC
    rec = np.dot(projected, np.transpose(pc))
    rec = np.reshape(rec, (40,40))

    for row in rec:
        print row

    plt.imshow(rec)
    plt.show()


def createProjectionMatrix(eigenvals, eigenvecs, k):
    eigenvecs = np.real(eigenvecs)
    eigenvals = np.copy(eigenvals)
    eigenvals = eigenvals.flatten()

    eigenvals_k = findIdxOfNLargest(eigenvals, k)
    evec = np.zeros([eigenvals.shape[0],k])
    counter = 0
    for idx in eigenvals_k:
        evec[:,counter] = np.transpose(eigenvecs[:,idx])
        counter+=1

    # evec *= -1
    return evec #projection matrix

def projectImages(images, proj_matrix):
    projected_images = np.dot(images, proj_matrix)
    return projected_images

def findIdxOfNLargest(arr, N):
    return arr.argsort()[-N:][::-1]

def loadAndPrepareImages(path):
    pattern = path + "subject*"
    imageFns = sorted(glob.glob(pattern))
    images = [ndimage.imread(f) for f in imageFns]

    rescaledAndFlat = list()

    rescaledAndFlat = np.zeros([154, 1600])
    images = np.array(images)
    images = images.astype(np.float32)
    original = np.copy(images)

    for i in range(len(images)):
        rescaled = imresize(images[i], (40,40))
        rescaled.astype(np.float32)
        rescaledAndFlat[i,:] = rescaled.flatten()

    return np.array(rescaledAndFlat), np.array(original)

def standardizeData(matrix):
    #for each column, find std and mean
    #subtract mean from each entry and
    #divide by std
    for i in range(0, len(matrix[0])):
        std = myStd(matrix[:,i])
        mean = np.mean(matrix[:,i])
        for j in range(0, len(matrix[:,i])):
            matrix[j][i] -= mean
            matrix[j][i] = myDivide(matrix[j][i], std)
    #at this point, all data is normalized. Return matrix
    return matrix

def myDivide(num, denom):
    if denom == 0:
        return num
    return float(num)/float(denom)

def myStd(array):
    array = np.array(array)
    mean = np.mean(array)

    summ = 0
    for el in array:
        summ += pow(el-mean, 2)
    summ /= (len(array) -1)
    summ = pow(summ, 0.5)
    return summ


def findK(vals, alpha):
    reals = np.copy(vals) #remove reference to argument
    #eliminate complex component. Amounts to pretty much zero
    reals = np.real(reals)
    eigenvalSum = np.sum(reals)
    #sort eigenvalue array in descending order
    reals = np.array(reals)
    #reals[::-1].sort()

    #find the number of eigenvalues whose sum
    #is over 0.95 of total eigenvalues
    running_sum = 0
    k = 1

    for eig in reals[0]:
        running_sum += (eig)
        if running_sum / eigenvalSum > alpha:
            return k
        k += 1

def findEigen(matrix):
    #find covariance matrix
    #set rowvar argument to False because features are columns, not rows
    covmat = np.cov(matrix, rowvar=False) #8x8 matrix
    #find eigenvalues/eigenvectors
    w, v = np.linalg.eig(covmat)
    return w, v



if __name__ == "__main__":
    main()
