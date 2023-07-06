rm -rf ./bin/*

shopt -s globstar
for i in $(find ./src -name '*.cpp'); do # Whitespace-safe and recursive
    file=${i##*/}
    base=${file%.*}
    g++ -c -I ./inc/ $i -o ./bin/$base.o
done

cc $(pkg-config --cflags glfw3 gl glew freetype2) -I ./inc/ main.cpp $(find ./bin/. \( -name  \*.o \)) -o main.exe -lm -lstdc++ $(pkg-config --libs glfw3 gl glew freetype2) -lOpenCL