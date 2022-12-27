myrealpath() {
  OURPWD=$PWD
  cd "$(dirname "$1")"
  LINK=$(readlink "$(basename "$1")")
  while [ "$LINK" ]; do
    cd "$(dirname "$LINK")"
    LINK=$(readlink "$(basename "$1")")
  done
  REALPATH="$PWD/$(basename "$1")"
  cd "$OURPWD"
  echo "$REALPATH"
}

eigen=$(myrealpath ~/.conan/data/eigen/3.4.0/_/_/package/*/ | head -n 1)
libsvm=$(myrealpath ~/.conan/data/libsvm/330/_/_/package/*/ | head -n 1)

export CXXFLAGS="-I$eigen/include -I$libsvm/include"
export LDFLAGS="-L$libsvm/lib"
export LD_LIBRARY_PATH="$libsvm/lib"
export DYLD_LIBRARY_PATH="$libsvm/lib"
