eigen=$(realpath ~/.conan/data/eigen/3.4.0/_/_/package/*/ | head -n 1)
libsvm=$(realpath ~/.conan/data/libsvm/330/_/_/package/*/ | head -n 1)
export CXXFLAGS="-I$eigen/include -I$libsvm/include"
export LDFLAGS="-L$libsvm/lib"
export LD_LIBRARY_PATH="$libsvm/lib"
