// by Cody Rivera <cjrivera1@crimson.ua.edu>

/*
  Testbed for integer histogramming

  Options: -a n, where n is an algorithm
              1: Current Imp. (default)
              2: 2007 paper.
              3: 2013 paper.
           -f filename, file input -- text: numValues, numBuckets, values
           or random-generated input
           -b buckets
           -v values
           -d n, where n is a distribution
              0 - uniform (default)
              1 - binomial
           -t number of trials
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "histogram.cuh"

/* declarations */
int* readFile(char* filename, int& numValues, int& numBuckets);
int* genValues(int numValues, int numBuckets);
int* genNormalValues(int numValues, int numBuckets);
void runTest(int algNumber, int numTrials, int* inputValues, int numValues, int* outputBuckets, int numBuckets, double& gbPerSecond);

int main(int argc, char** argv) {
    /* Argument info */
    int   algNumber = 1;
    bool  fileMode  = false;
    char* filename;
    bool  randMode    = false;
    int   randBuckets = -1;
    int   randValues  = -1;
    int   distNumber  = 0;
    int   numTrials   = 100;

    { /* Parse the arguments */
        int i = 1;
        while (i < argc) {
            if (argv[i][0] == '-') {
                switch (argv[i][1]) {
                    case 'a':
                        if (i + 1 < argc) algNumber = atoi(argv[(i++) + 1]);
                        break;
                    case 'f':
                        fileMode = true;
                        if (i + 1 < argc) filename = argv[(i++) + 1];
                        break;
                    case 'b':
                        randMode = true;
                        if (i + 1 < argc) randBuckets = atoi(argv[(i++) + 1]);
                        break;
                    case 'v':
                        randMode = true;
                        if (i + 1 < argc) randValues = atoi(argv[(i++) + 1]);
                        break;
                    case 'd':
                        if (i + 1 < argc) distNumber = atoi(argv[(i++) + 1]);
                        break;
                    case 't':
                        if (i + 1 < argc) numTrials = atoi(argv[(i++) + 1]);
                        break;
                    default:
                        fprintf(stderr, "Invalid switch at position %d: %s\n", i, argv[i]);
                        return -1;
                }
            } else {
                fprintf(stderr, "Invalid argument at position %d: %s\n", i, argv[i]);
                return -1;
            }
            ++i;
        }
    }

    int* inputValues;
    int  numValues;

    int* outputBuckets;
    int  numBuckets;

    if (fileMode == randMode) {
        fprintf(stderr, "Invalid mode\n");
        return -1;
    } else if (fileMode) {
        inputValues = readFile(filename, numValues, numBuckets);
    } else /* randMode */
    {
        numValues  = randValues;
        numBuckets = randBuckets;
        if (numValues < 1 || numBuckets < 1) {
            fprintf(stderr, "Invalid number of values or buckets\n");
            return -1;
        }
        switch (distNumber) {
            case 1:
                inputValues = genNormalValues(numValues, numBuckets);
                break;
            default:
                inputValues = genValues(numValues, numBuckets);
                break;
        }
    }

    outputBuckets = new int[numBuckets];

    int* trueOutputBuckets = new int[numBuckets];

    /* Perform true check */
    for (int i = 0; i < numBuckets; ++i) {
        trueOutputBuckets[i] = 0;
    }
    for (int i = 0; i < numValues; ++i) {
        ++trueOutputBuckets[inputValues[i]];
    }

    double gbPerSecond;

    runTest(algNumber, numTrials, inputValues, numValues, outputBuckets, numBuckets, gbPerSecond);

    for (int i = 0; i < numBuckets; ++i) {
        // fprintf(stderr, "%d: (naive) %d, (alg) %d", i, trueOutputBuckets[i], outputBuckets[i]);
        if (trueOutputBuckets[i] != outputBuckets[i]) {
            fprintf(stderr, "Don't match\n");
            return -1;
        } else {
            // fprintf(stderr, "\n");
        }
    }

    printf("algNumber(%d), numTrials(%d), numValues(%d), numBuckets(%d), gbPerSecond(%g)\n", algNumber, numTrials, numValues, numBuckets,
           gbPerSecond);

    delete[] inputValues;
    delete[] outputBuckets;

    return 0;
}

int* readFile(char* filename, int& numValues, int& numBuckets) {
    FILE* input = NULL;
    input       = fopen(filename, "r");
    if (input == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(-1);
    }

    scanf("%d", &numValues);
    scanf("%d", &numBuckets);

    if (numValues < 1 || numBuckets < 1) {
        fprintf(stderr, "Invalid number of values or buckets\n");
        exit(-1);
    }

    int* array = new int[numValues];

    for (int i = 0; i < numValues; ++i) {
        scanf("%d", &array[i]);
        if (array[i] < 0 || array[i] >= numBuckets) {
            fprintf(stderr, "Invalid input\n");
            exit(-1);
        }
    }

    fclose(input);
    return array;
}

int* genValues(int numValues, int numBuckets) {
    srand(time(NULL));
    int* array = new int[numValues];
    for (int i = 0; i < numValues; ++i) {
        array[i] = rand() % numBuckets;
    }
    return array;
}

int* genNormalValues(int numValues, int numBuckets) {
    int*                       array = new int[numValues];
    unsigned int               seed  = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    /* generate mostly from 0.0 to 1.0 */
    std::normal_distribution<double> distribution(0.5, 0.341);

    for (int i = 0; i < numValues; ++i) {
        double value = distribution(generator);
        while (value < 0.0 || value > 1.0) {
            value = distribution(generator);
        }
        array[i] = (int)(value * numBuckets);
    }
    return array;
}

/* CUDA algorithm testbed */
void runTest(int algNumber, int numTrials, int* inputValues, int numValues, int* outputBuckets, int numBuckets, double& gbPerSecond) {
    int* deviceInputValues;
    int* deviceOutputBuckets;

    cudaEvent_t start, end;

    cudaErrchk(cudaEventCreate(&start));
    cudaErrchk(cudaEventCreate(&end));

    /* Copy data to GPU */

    cudaErrchk(cudaMalloc(&deviceInputValues, sizeof(int) * numValues));
    cudaErrchk(cudaMalloc(&deviceOutputBuckets, sizeof(int) * numBuckets));

    cudaErrchk(cudaMemcpy(deviceInputValues, inputValues, sizeof(int) * numValues, cudaMemcpyHostToDevice));

    switch (algNumber) {
        case 1: {
            int numBlocks          = (numValues / (256 * 32)) + 1;
            int threadsPerBlock    = 256;
            int symbols_per_thread = 32;
            cudaErrchk(cudaEventRecord(start));
            for (int i = 0; i < numTrials; ++i) {
                cudaErrchk(cudaMemset(deviceOutputBuckets, 0, sizeof(int) * numBuckets));
                naiveHistogram<<<numBlocks, threadsPerBlock>>>(deviceInputValues, deviceOutputBuckets, numValues, symbols_per_thread);
            }
            cudaErrchk(cudaEventRecord(end));
        } break;

        case 3: {
            int maxbytes = 98304;
            cudaFuncSetAttribute(p2013Histogram, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

            /* Sets up parameters */
            int numSMs         = 84;
            int itemsPerThread = 1;

            int RPerBlock = (maxbytes / sizeof(int)) / (numBuckets + 1);
            int numBlocks = numSMs;

            /* Fits to size */
            int threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
            while (threadsPerBlock > 1024) {
                if (RPerBlock <= 1) {
                    threadsPerBlock = 1024;
                } else {
                    RPerBlock /= 2;
                    numBlocks *= 2;
                    threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
                }
            }

            cudaErrchk(cudaEventRecord(start));
            for (int i = 0; i < numTrials; ++i) {
                cudaErrchk(cudaMemset(deviceOutputBuckets, 0, sizeof(int) * numBuckets));
                p2013Histogram<<<numBlocks, threadsPerBlock, ((numBuckets + 1) * RPerBlock) * sizeof(int)>>>(deviceInputValues, deviceOutputBuckets,
                                                                                                             numValues, numBuckets, RPerBlock);
            }
            cudaErrchk(cudaEventRecord(end));
        } break;
        default:
            break;
    }

    cudaErrchk(cudaDeviceSynchronize());

    float  time;
    double dtime;
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    /* Calc GB/s*/
    dtime = time;
    dtime /= 1000;

    dtime /= numTrials;

    // printf("%g for %g\n", dtime, ((double) numValues * sizeof(int)));

    gbPerSecond = (((double)numValues * sizeof(int)) / (double)10e9) / dtime;

    cudaErrchk(cudaMemcpy(outputBuckets, deviceOutputBuckets, sizeof(int) * numBuckets, cudaMemcpyDeviceToHost));
    cudaErrchk(cudaFree(deviceInputValues));
    cudaErrchk(cudaFree(deviceOutputBuckets));
    cudaErrchk(cudaEventDestroy(start));
    cudaErrchk(cudaEventDestroy(end));
}
