#include<stdio.h>
#include<mpi.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<fstream>
#include<chrono>
#include<cstdlib>
#include<omp.h>


using namespace cv;
using namespace std::chrono;

//function declaration
Mat applyGreyscale(Mat image);
Mat convertBGR2HSV(Mat image);
Mat myOwnEmboss(Mat image);
void imageProcessing(std::string* imageInputs, int* buffer, int bufferSize, int rank);

const int rounds = 15;
const int imageCount = 15;
const int filter_width = 3;
const int filter_height = 3;

double times[rounds];

int filter[filter_width][filter_height] =
{
    -1, -1, 0,
    -1, 0, 1,
    0, 1, 1
};

int main(int argc, char* argv[])
{
    //number of processes
    int size;
    int rank;

    std::string inputs[imageCount];//magic number should be removed in future

    int indices[imageCount];
    for (int i = 0; i < imageCount; i++)
    {
        indices[i] = i;
    }

    std::ifstream file("img/img_names.txt");
    if (file.is_open())
    {
        int index = 0;

        while (!file.eof() && index < imageCount) {
            getline(file, inputs[index]);
            ++index;
        }
        file.close();
    }
    else {
        std::cout << "Unable to open the file";
        return -1;
    }


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size > imageCount) {
        std::cout << "The count of starting processes can't be higher than the maximal number of images: " << imageCount << std::endl;
        MPI_Finalize();
        return -1;
    }

    const int partBufferSize = imageCount / size;
    const int restBufferSize = imageCount % size;

    int restBuffer[imageCount];

    if (imageCount % size != 0) {
        if (rank == 0) {
            for (int i = 0; i < restBufferSize; i++)
            {
                restBuffer[i] = imageCount - restBufferSize + i;
            }
        }
    }

    int partBuffer[imageCount];

    for (int round = 0; round < rounds; round++)
    {

        //start calculation for duration
        auto start = high_resolution_clock::now();

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter(&indices, partBufferSize, MPI_INT, &partBuffer, partBufferSize, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++)
        {
            if (rank == i) {
                for (int i = 0; i < partBufferSize; i++)
                {
                    std::cout << "Partbuffer Value:  " << partBuffer[i] << " at process number:  " << rank << std::endl;
                }
            }
        }

        imageProcessing(inputs, partBuffer, partBufferSize, rank);

        if (rank == 0) {
            imageProcessing(inputs, restBuffer, restBufferSize, rank);
        }

        //Following lines are for displaying images
        //namedWindow(inputs[i], WINDOW_AUTOSIZE );
        //imshow(inputs[i], img);
        //waitKey(0);

        //stop calculation for duration
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

        times[round] = duration.count();

    }

    MPI_Finalize();

    double time = 0.0;

    for (int i = 0; i < rounds; i++)
    {
        time += times[i];
    }

    std::cout << "DURATION WITH MPI PARALLEL: " << time / rounds << "    From process number:   " << rank << std::endl;

    return 0;
}

void imageProcessing(std::string* imageInputs, int* buffer, int bufferSize, int rank) {

    for (size_t i = 0; i < bufferSize; i++)
    {
        int index = buffer[i];

        Mat image;
        image = imread("img/" + imageInputs[index], IMREAD_COLOR);

        if (!image.data)
        {
            std::cout << "No Image Data from process NR: " << rank << std::endl;
            MPI_Finalize();
            return;
        }

        imwrite("out/greyscale/" + imageInputs[index], applyGreyscale(image));
        imwrite("out/hsv/" + imageInputs[index], convertBGR2HSV(image));
        imwrite("out/emboss/" + imageInputs[index], myOwnEmboss(image));


        std::cout << "File: " << imageInputs[index] << "saved" << " from process number: " << rank << std::endl;
    }

}

Mat applyGreyscale(Mat image)
{
    //initializing a zero Mat array from the input image
    Mat new_image = Mat::zeros(image.size(), image.type());


    for (int y = 0; y < image.rows; y++)
    {

        for (int x = 0; x < image.cols; x++)
        {

            for (int c = 0; c < image.channels(); c++)
            {
                Vec3b intensity = image.at<Vec3b>(y, x);
                new_image.at<Vec3b>(y, x)[c] = 0.07 * intensity.val[0] + 0.72 * intensity.val[1] + 0.21 * intensity.val[2];
            }
        }
    }

    return new_image;
}

Mat convertBGR2HSV(Mat image)
{
    //initializing a zero Mat array from the input image
    Mat new_image = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            Vec3b intensity = image.at<Vec3b>(y, x);
            int h;
            int s;
            int v = 0;

            float r = intensity.val[2] / (float)255;
            float g = intensity.val[1] / (float)255;
            float b = intensity.val[0] / (float)255;

            float cmax = max(max(r, g), b);
            float cmin = min(min(r, g), b);
            float diff = cmax - cmin;

            if (cmax == cmin)
            {
                h = 0;
            }
            else if (cmax == r)
            {
                h = ((int)(60 * ((g - b) / diff) + 360)) % 360;
            }
            else if (cmax == g)
            {
                h = ((int)(60 * ((b - r) / diff) + 120)) % 360;
            }
            else if (cmax == b)
            {
                h = ((int)(60 * ((r - g) / diff) + 240)) % 360;
            }

            if (cmax == 0)
            {
                s = 0;
            }
            else
            {
                s = (diff / (float)cmax) * 100;
            }

            v = cmax * 100;

            intensity.val[0] = h;
            intensity.val[1] = s;
            intensity.val[2] = v;

            new_image.at<Vec3b>(y, x) = intensity;
        }
    }

    return new_image;
}

Mat myOwnEmboss(Mat image) {
    //initializing a zero Mat array from the input image
    Mat new_image = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows - 1; y++)
    {
        for (int x = 0; x < image.cols - 1; x++)
        {
            Vec3b intensity = image.at<Vec3b>(y, x);

            double red = 0.0, green = 0.0, blue = 0.0, diffG, diffB, diffR, absDiff;

            if (y - 1 > 0 && y - 1 < image.rows && x - 1 > 0 && x - 1 < image.cols) {
                diffR = image.at<Vec3b>(y, x)[2] - image.at<Vec3b>(y - 1, x - 1)[2];
                diffG = image.at<Vec3b>(y, x)[1] - image.at<Vec3b>(y - 1, x - 1)[1];
                diffB = image.at<Vec3b>(y, x)[0] - image.at<Vec3b>(y - 1, x - 1)[0];

                absDiff = abs(diffR) < abs(diffB) ? (abs(diffB) < abs(diffG) ? diffG : diffB) : (abs(diffR) < abs(diffG) ? diffG : diffR);
            }
            else {
                absDiff = 0;
            }

            absDiff += 128;

            if (absDiff > 255) {
                absDiff = 255;
            }
            if (absDiff < 0) {
                absDiff = 0;
            }

            new_image.at<Vec3b>(y, x)[2] = absDiff;
            new_image.at<Vec3b>(y, x)[1] = absDiff;
            new_image.at<Vec3b>(y, x)[0] = absDiff;
        }
    }
    return new_image;
}