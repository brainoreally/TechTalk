#include "src/Graphics/Graphics.h"

int main() {

    Graphics graphics = Graphics();

    // Main loop
    while (graphics.is_running()) {
        graphics.draw();
    }

    // Clean up
    return 0;
}