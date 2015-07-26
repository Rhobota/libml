
/*
 * If defined, this file will take a long time to compile, but
 * will output very fast code. You should turn this on for
 * production code, but turn it off to do quick test iterations.
 */
#define COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE 1


/*
 * If defined, this code will refuse to run the fallback
 * implementation, which is a very slow implementation.
 * This is nice to turn on if you want to be sure your
 * code uses the fast templated versions of the functions
 * instead of the fallback.
 */
#define THROW_IF_FALLBACK_IMPL_NEEDED 0


/*
 * The execution config to use for these CUDA functions.
 * Choosing the right values for these is very important.
 *
 * A block size that is too big will decrease the number
 * of registers each thread has access to, which will force
 * it to use local memory, which will slow it down.
 *
 * A block size that is too small will cause redundant work
 * to be performed because each block copies an apron of
 * the input into shared memory. You don't want to copy that
 * apron more than you have to.
 *
 * The number of threads in a block should be a multiple of
 * 32 (aka, the warp size of every CUDA device right now).
 *
 * Increasing DESIRED_BLOCKS_PER_SM will increase the amount
 * of concurrency you get, but will decrease the number of registers
 * each thread gets, again causing each thread to use more local memory
 * and slowing it down.
 *
 * All that said, the following seems to be a happy medium for
 * my GTX 550 Ti GPU.
 */
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_X 32
#define DESIRED_BLOCKS_PER_SM 2


/*
 * These defs are effective only when using the non-templated version of the CUDA functions.
 */
#define MAX_KERNELS_SUPPORTED 100
#define MAX_INPUT_COMPONENTS_SUPPORTED 100

