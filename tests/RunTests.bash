#!/bin/bash

TEST_DIR="$(dirname "$0")"
INCLUDE_FLAGS="-I $TEST_DIR/../include -I $TEST_DIR/../../librho/include"
LIB_PATH="$TEST_DIR/../objects/libml.a $TEST_DIR/../../librho/objects/librho.a"
OUT_FILE="a.out"

CC="$TARGET""g++"
CC_FLAGS_LOCAL="$CC_FLAGS \
    -g -O0 -fvisibility=hidden -fno-inline -Wall -Wextra \
    -Wno-unused-parameter -Wno-long-long -Werror -pedantic \
    -D_FILE_OFFSET_BITS=64 \
    $INCLUDE_FLAGS"
CC_LIB_FLAGS="$LIB_PATH -fopenmp $CC_LIB_FLAGS"

if [ $(uname) == "Linux" ]
then
    CC_FLAGS_LOCAL+=" -rdynamic"
    CC_LIB_FLAGS+=" "
elif [ $(uname) == "Darwin" ]
then
    CC_FLAGS_LOCAL+=" -rdynamic -mmacosx-version-min=10.6"
    CC_LIB_FLAGS+=" -framework Foundation -framework AVFoundation"
    CC_LIB_FLAGS+=" -framework CoreVideo -framework CoreMedia"
    CC_LIB_FLAGS+=" -framework OpenGL -framework IOKit"
    CC_LIB_FLAGS+=" -framework Cocoa -framework GLUT"
else
    # Mingw
    CC_FLAGS_LOCAL+=" "
    CC_LIB_FLAGS+=" -lwsock32 -lws2_32"
fi

CC_CUDA="/usr/local/cuda-6.5/bin/nvcc"
CC_CUDA_FLAGS_LOCAL="$CC_FLAGS \
    -g -O0 -Xcompiler -fvisibility=hidden -Xcompiler -fno-inline -Xcompiler -Wall -Xcompiler -Wextra \
    -Xcompiler -Wno-unused-parameter -Xcompiler -Wno-long-long -Xcompiler -Werror \
    -D_FILE_OFFSET_BITS=64 \
    $INCLUDE_FLAGS"
CC_CUDA_LIB_FLAGS="$LIB_PATH \
    -lcublas_static -lculibos -lcurand_static -Xcompiler -fopenmp"

if [ -n "$1" ]
then
    TEST_DIR="$1"
fi

RUN_PREFIX="$2"

function getExtension
{
    filename="${1##*/}"
    extension="${filename##*.}"
    echo "$extension"
}

function cleanup
{
    rm -rf "$OUT_FILE"*
}

function gotSignal
{
    echo
    if [ -n $testPid ]
    then
        echo "Killing process $testPid ..."
        kill -SIGTERM $testPid
    fi
    cleanup
    echo "Exiting due to signal ..."
    exit 1
}

trap gotSignal SIGTERM SIGINT

somethingRun=0

for testPath in $(find "$TEST_DIR" -name '*.cpp' -o -name '*.cu' | sort)
do
    somethingRun=1
    echo "---- $testPath ----"
    if [ $(getExtension $testPath) = "cu" ]
    then
        $CC_CUDA $CC_CUDA_FLAGS_LOCAL $testPath $CC_CUDA_LIB_FLAGS -o "$OUT_FILE"
    else
        $CC $CC_FLAGS_LOCAL $testPath $CC_LIB_FLAGS -o "$OUT_FILE"
    fi
    compileStatus=$?
    if [ $compileStatus -eq 0 ]
    then
        if [ -n "$RUN_PREFIX" ]
        then
            "$RUN_PREFIX" ./"$OUT_FILE"
        else
            ./"$OUT_FILE" &
            testPid=$!
            wait $testPid
        fi
        testStatus=$?
        echo
        if [ $testStatus -ne 0 ]
        then
            cleanup
            echo "^^^^^^^^ TEST FAILURE ^^^^^^^^"
            exit 1
        fi
    else
        cleanup
        echo
        echo "^^^^^^^^ COMPILATION ERROR ^^^^^^^^"
        exit 1
    fi
done

if [ $somethingRun -eq 0 ]
then
    echo "^^^^^^^^ NO TESTS WHERE RUN ^^^^^^^^"
    exit 1
fi

cleanup
echo "____ ALL TESTS PASSED ____"
exit 0
