/*
 * color.h - Color Utilities and PPM Output
 * 
 * Provides color type alias and functions for writing PPM image format.
 * Handles gamma correction and color clamping for proper image output.
 */

#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

// Type alias for clarity - colors are just vec3 with (r,g,b) components
typedef vec3 color;

/**
 * write_color - Write RGB color to output stream in PPM format
 * 
 * Applies gamma correction (gamma = 2.0) and clamps values to [0,255].
 * PPM format expects integer values 0-255 for each RGB component.
 * 
 * @param out Output stream (typically std::cout)
 * @param pixel_color Color value (typically in range [0,1])
 */
void write_color(std::ostream &out, color pixel_color) {
    // Apply gamma correction (gamma = 2.0 means square root)
    double r = sqrt(pixel_color.x());
    double g = sqrt(pixel_color.y());
    double b = sqrt(pixel_color.z());

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = int(256 * clamp(r, 0.0, 0.999));
    int gbyte = int(256 * clamp(g, 0.0, 0.999));
    int bbyte = int(256 * clamp(b, 0.0, 0.999));

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

/**
 * clamp - Clamp value to range [min, max]
 * Ensures color values stay within valid bounds
 */
double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#endif 