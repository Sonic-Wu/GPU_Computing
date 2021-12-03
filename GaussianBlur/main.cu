#include <string>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cuda.h>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


static void HandleError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}


#define HANDLE_ERROR(err)(HandleError(err,__FILE__,__LINE__))



//================== Definition =================================//
// Define Global variables
int width, height,temp_width, temp_height, conv_im_width,conv_im_height,kernel_size;

// Define struct to save RGB image
struct pixel{
    unsigned char red, green, blue;
};

//================== PPM file read function =====================//
pixel * read_ppmfile(const std::string filename){
    std::ifstream ppmFile;
    std::string buffer;
    std::string inputfilename = "../" + filename + ".ppm";

    //open file and save into stream
    ppmFile.open(inputfilename, std::ifstream::binary);
    if(ppmFile.fail()){
        std::cout << "Could not open the file!" << std::endl;
    }

    //check PPM header
    if(std::getline(ppmFile, buffer)){
        if(buffer != "P6"){
            std::cout << "invaild PPM file"<<std::endl;
        }
    }                          
    else{
        std::cout << "Cannot read the PPM type!" << std::endl;
    }

    //skip comments
    while(std::getline(ppmFile, buffer)){
        if(buffer[0] == '#')
            continue;
        else
            break;
    }

    //read image size
    std::stringstream iss;
    iss.str(buffer);
    iss>>width;
    iss>>height;

    //read maximum value in the PPM file
    int max_value;
    if(std::getline(ppmFile,buffer)){
        iss.clear();
        iss.str(buffer);
        iss>>max_value;
        if(max_value <= 0)
            std::cout << "Invaild maximum value!" << std::endl;
    }

    //read RGB of image
    pixel* image;

    //allocate space for RGB of image
    image = (pixel*)malloc(width * height * sizeof(pixel));

    //read RGB from stream into string

        //read from istream into string
        std::string ppm_rgb ((std::istreambuf_iterator<char>(ppmFile)), std::istreambuf_iterator<char>());

        //saving RGB value of pixels into RGB struct
        if(ppm_rgb.size() / 3 != height * width)
            std::cout << "Invaild pixel RGB number!" << std::endl;
        for(int i = 0; i < height * width; i ++){
            image[i].red = ppm_rgb[3*i];
            image[i].green = ppm_rgb[3*i + 1];
            image[i].blue = ppm_rgb[3*i + 2];
        }
    ppmFile.close();

    return image;
}

//================== PPM file output function =======================//
void write_ppmFile(pixel* image, std::string filename, int image_x, int image_y){
    std::ofstream PPMout;
    std::string outname = "../" + filename + ".ppm";
    PPMout.open(outname, std::ofstream::binary);
    PPMout << "P6" <<std::endl
           << "# File after convolution" << std::endl
           << image_x << " " << image_y << std::endl
           << "255" << std::endl;
    for(int i(0); i < image_x * image_y; i++){
        PPMout << image[i].red << image[i].green << image[i].blue;
    }
    PPMout.close();
}

//================= Guassian Kernel creating funciton ===============//
float * Gaussian_Kernel(int sigma){
    float const pi = 3.1415926;
    int k;
    k = 6 * sigma;
    if(k % 2 == 0) k++;
    kernel_size = k;
    float sum = 0;

    //allocate space for gaussian kernel
    float* K = (float*)malloc(k * sizeof(float));
    for(int i(0); i < k; i++){
        K[i] = exp(-(i - (k-1)/2)*(i - (k-1)/2)/(2 * sigma * sigma)) / sqrt(2 * sigma * sigma * pi);
        sum += K[i];
    }

    //nomalize kernel so that the brightness of image won't be changed
    for(int i(0); i < k; i++)
        K[i] = K[i] / sum;

    return K;
}

//=================== CPU - Applying Gaussian Kernel to image =======//
pixel * Gaussian_Blur(float * Kernel, pixel * image){
    temp_width = width - kernel_size + 1;
    temp_height = height;
    conv_im_width = temp_width;
    conv_im_height = height - kernel_size + 1;

    pixel* temp = (pixel*)malloc(temp_width * temp_height * sizeof(pixel));
    pixel* output = (pixel*)malloc(conv_im_width * conv_im_height * sizeof(pixel));
    // 2D Gaussian Blur can be separated into two 1D convolution horizontally and vertically

        // horizontal convolution
        // image height scan
        for(int i_height(0);i_height < temp_height; i_height++){
            //image width scan
            for(int i_width(0); i_width < temp_width; i_width++){
                float sum_red(0);
                float sum_green(0);
                float sum_blue(0);

                for(int k_width(0); k_width < kernel_size; k_width++){
                    sum_red += Kernel[k_width] * (int)image[i_height * width + i_width + k_width].red;
                    sum_green += Kernel[k_width] * (int)image[i_height * width + i_width + k_width].green;
                    sum_blue += Kernel[k_width] * (int)image[i_height * width + i_width + k_width].blue;
                }
                temp[i_height * temp_width + i_width].red = (unsigned char)(int)sum_red;
                temp[i_height * temp_width + i_width].green = (unsigned char)(int)sum_green;
                temp[i_height * temp_width + i_width].blue = (unsigned char)(int)sum_blue;

            }
        }
        //vertical convolution
        //image width scann
        for(int i_height(0); i_height < conv_im_height; i_height++){
            //image height scann
            for(int i_width(0); i_width < conv_im_width; i_width++){
                float sum_red(0);
                float sum_green(0);
                float sum_blue(0);
                for(int k_height(0); k_height < kernel_size; k_height ++){
                    sum_red += Kernel[k_height] * (int)temp[(i_height + k_height) * conv_im_width + i_width].red;
                    sum_green += Kernel[k_height] * (int)temp[(i_height + k_height) * conv_im_width + i_width].green;
                    sum_blue += Kernel[k_height] * (int)temp[(i_height + k_height) * conv_im_width + i_width].blue;
                }
                output[i_height * conv_im_width + i_width].red = (unsigned char)(int)sum_red;
                output[i_height * conv_im_width + i_width].green = (unsigned char)(int)sum_green;
                output[i_height * conv_im_width + i_width].blue = (unsigned char)(int)sum_blue;
            }
        }

    return output;
}

//============================= GPU Xdim Kernel =======================//
__global__
void Convolution_image_x(unsigned char * D_Input_Image, float * D_Kernel, unsigned char * D_Output_Image, size_t t_kernel_size, size_t t_width, size_t t_height){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= t_width || j >= t_height) return;
    float sum[3] = {0};
    int k;
    for(k = 0; k < t_kernel_size; k++){
        sum[0] += D_Kernel[k] * (int)D_Input_Image[(j * (t_width + t_kernel_size - 1) + i + k) * 3];
        sum[1] += D_Kernel[k] * (int)D_Input_Image[(j * (t_width + t_kernel_size - 1) + i + k) * 3 + 1];
        sum[2] += D_Kernel[k] * (int)D_Input_Image[(j * (t_width + t_kernel_size - 1) + i + k) * 3 + 2];
    }
    D_Output_Image[(j * t_width + i) * 3] = (unsigned char)(int)sum[0];
    D_Output_Image[(j * t_width + i) * 3 + 1] = (unsigned char)(int)sum[1];
    D_Output_Image[(j * t_width + i) * 3 + 2] = (unsigned char)(int)sum[2];
}
//============================= GPU Ydim Kenrel =======================//
__global__
void Convolution_image_y(unsigned char * D_Input_Image, float * D_Kernel, unsigned char * D_Output_Image, size_t i_kernel_size, size_t i_width, size_t i_height){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= i_width || j >= i_height) return;
    float sum[3] = {0};
    int k;
    for(k = 0; k < i_kernel_size; k++){
        sum[0] += D_Kernel[k] * (int)D_Input_Image[((j+k) * i_width  + i ) * 3];
        sum[1] += D_Kernel[k] * (int)D_Input_Image[((j+k) * i_width + i ) * 3 + 1];
        sum[2] += D_Kernel[k] * (int)D_Input_Image[((j+k) * i_width + i ) * 3 + 2];
    }
    D_Output_Image[(j * i_width + i) * 3] = (unsigned char)(int)sum[0];
    D_Output_Image[(j * i_width + i) * 3 + 1] = (unsigned char)(int)sum[1];
    D_Output_Image[(j * i_width + i) * 3 + 2] = (unsigned char)(int)sum[2];
}


//================= Main Fucntion ================================//
int main(int argc, char* argv[]){
    if(argc != 3)
        return 1;
    //std::string filename("..//hereford_512.ppm")
    std::string filename(argv[1]);
    int sigma = atoi(argv[2]);

    //CPU version
    //set begin timer
    auto t_start = std::chrono::high_resolution_clock::now();

    //begin compute
    pixel* input = read_ppmfile(filename);
    float* H_Kernel = Gaussian_Kernel(sigma);
    pixel* output = Gaussian_Blur(H_Kernel, input);
    write_ppmFile(output, filename + "_CPUOutput", conv_im_width, conv_im_height);

    //Set End Timer
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
              << std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000
              << "s\n";

    //GPU Version
    int dev = 0;
    cudaDeviceProp deviceProp;
    HANDLE_ERROR((cudaGetDeviceProperties(&deviceProp, dev)));
    //Set Begin Timer
    cudaEvent_t D_start, D_stop;
    cudaEventCreate(&D_start);
    cudaEventCreate(&D_stop);
    
    //Copy Data to a Device
    int Device_count(0);
    HANDLE_ERROR(cudaGetDeviceCount(&Device_count));
    if(Device_count < 1)
        std::cout<< "There is not enough device to run CUDA" << '\n';

    float* D_Kernel;
    unsigned char* D_Input_Image;
    unsigned char* D_Temp_Image;   
    unsigned char* D_Output_Image;
    unsigned char* H_Output_Image;
    unsigned char* H_Input_Image;
    
    //Copy Kernel from Host to Device
    HANDLE_ERROR(cudaMalloc(&D_Kernel, kernel_size * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(D_Kernel, H_Kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    //Copy Original Image from Host to Device
    H_Input_Image = (unsigned char *)malloc( 3 * width * height * sizeof(unsigned char));
   
    for(int i = 0; i < width * height; i++){
        H_Input_Image[3 * i] = input[i].red;
        H_Input_Image[3 * i + 1] = input[i].green;
        H_Input_Image[3 * i + 2] = input[i].blue;
    }
    HANDLE_ERROR(cudaMalloc(&D_Input_Image, 3 * width * height * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMemcpy(D_Input_Image, H_Input_Image, 3 * width * height, cudaMemcpyHostToDevice));
      
    //Allocate Space in Device for Images After X Convolution and Y Convolution 
    HANDLE_ERROR(cudaMalloc(&D_Temp_Image, 3 * temp_width * temp_height * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc(&D_Output_Image, 3 * conv_im_width * conv_im_height * sizeof(unsigned char)));

    //Run Kernel
    HANDLE_ERROR(cudaEventRecord(D_start));
    dim3 threads(sqrt(deviceProp.maxThreadsDim[0]), sqrt(deviceProp.maxThreadsDim[0]));
    dim3 blocks(width/threads.x + 1, height/threads.y + 1);
    Convolution_image_x<<<blocks, threads>>>(D_Input_Image, D_Kernel, D_Temp_Image, kernel_size, temp_width, temp_height);
    Convolution_image_y<<<blocks, threads>>>(D_Temp_Image, D_Kernel, D_Output_Image, kernel_size, conv_im_width, conv_im_height);
    HANDLE_ERROR(cudaEventRecord(D_stop));
    HANDLE_ERROR(cudaEventSynchronize(D_stop));
    
    //Copy Image from Device to Host
    H_Output_Image = (unsigned char*)malloc(3 * conv_im_height * conv_im_width * sizeof(unsigned char));
    HANDLE_ERROR(cudaMemcpy(H_Output_Image, D_Output_Image, 3 * conv_im_width * conv_im_height, cudaMemcpyDeviceToHost));


    //Export the image into file temp_width * temp_height conv_im_height * conv_im_height;
    pixel* H_output_GPU = (pixel*)malloc(conv_im_width * conv_im_height * sizeof(pixel));

    for(int i = 0; i < conv_im_width * conv_im_height; i++){
        H_output_GPU[i].red = H_Output_Image[3 * i];
        H_output_GPU[i].green = H_Output_Image[3 * i + 1];
        H_output_GPU[i].blue = H_Output_Image[3 * i + 2];
    }
    write_ppmFile(H_output_GPU, filename + "_GPUOutput" , conv_im_width, conv_im_height);
    //Set End Timer
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, D_start, D_stop));
    HANDLE_ERROR(cudaEventDestroy(D_start));
    HANDLE_ERROR(cudaEventDestroy(D_stop));

    std::cout <<  std::fixed << std::setprecision(2) << "GPU time used: "
                  << milliseconds / 1000 << "s\n"
                  << "Generate Successfully." << '\n';
    
    free(input);
    free(H_Kernel);
    free(output);
    free(H_output_GPU);
    free(H_Output_Image);
    free(H_Kernel);
    HANDLE_ERROR(cudaFree(D_Kernel));
    HANDLE_ERROR(cudaFree(D_Input_Image));
    HANDLE_ERROR(cudaFree(D_Output_Image));
    HANDLE_ERROR(cudaFree(D_Temp_Image));
 
    return 0;
}