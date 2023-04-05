#include "src/Graphics/Graphics.h"

int main() {

    Graphics graphics = Graphics();

    // Main loop
    while (graphics.isRunning()) {
        graphics.draw();
    }

    // Clean up
    return 0;
}