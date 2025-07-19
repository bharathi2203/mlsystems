/*
 * rtweekend.h - Common Utilities and Constants
 * 
 * Central header containing shared utilities, mathematical constants,
 * random number generation, and common type definitions used throughout
 * the ray tracer. Includes interval class for range operations.
 */

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <cstdlib>

// Type aliases for memory management
using std::shared_ptr;
using std::make_shared;

// Mathematical constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

/**
 * degrees_to_radians - Convert degrees to radians
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

/**
 * interval - Range/Interval Class
 * 
 * Represents a closed interval [min, max] with utility methods
 * for range checking and clamping. Used for ray parameter ranges
 * and color value clamping.
 */
class interval {
  public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // Default = empty interval
    interval(double _min, double _max) : min(_min), max(_max) {}

    /**
     * contains - Check if value is within interval [min, max]
     */
    bool contains(double x) const {
        return min <= x && x <= max;
    }

    /**
     * surrounds - Check if value is strictly within interval (min, max)
     */
    bool surrounds(double x) const {
        return min < x && x < max;
    }

    /**
     * clamp - Clamp value to interval bounds
     * @param x Value to clamp
     * @return Value clamped to [min, max]
     */
    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

// Static interval definitions
const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

/**
 * random_double - Generate random number in [0,1)
 * @return Random double in range [0, 1)
 */
inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

/**
 * random_double - Generate random number in [min,max)
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return Random double in range [min, max)
 */
inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

// Common Headers
#include "color.h"
#include "ray.h"
#include "vec3.h"

/**
 * random_in_unit_sphere - Generate random point inside unit sphere
 * 
 * Uses rejection sampling to generate uniformly distributed points
 * inside a unit sphere. Used for diffuse material scattering.
 */
inline vec3 random_in_unit_sphere() {
    // Generate random point inside unit sphere using rejection sampling
    while (true) {
        vec3 p = vec3(random_double(-1,1), random_double(-1,1), random_double(-1,1));
        if (p.length_squared() >= 1) continue;  // Reject if outside sphere
        return p;
    }
}

/**
 * random_unit_vector - Generate random unit vector (on sphere surface)
 * @return Random unit vector for true Lambertian distribution
 */
inline vec3 random_unit_vector() {
    // Generate random unit vector (on surface of unit sphere)
    return unit_vector(random_in_unit_sphere());
}

/**
 * random_on_hemisphere - Generate random vector on hemisphere
 * 
 * Generates random unit vector on hemisphere oriented around given normal.
 * Used for hemisphere sampling in diffuse materials.
 * 
 * @param normal Surface normal defining hemisphere orientation
 * @return Random vector on hemisphere around normal
 */
inline vec3 random_on_hemisphere(const vec3& normal) {
    // Generate random vector on hemisphere around the normal
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/**
 * random_in_unit_disk - Generate random point inside unit disk
 * 
 * Uses rejection sampling to generate uniformly distributed points
 * inside a unit disk in the XY plane. Used for camera defocus blur.
 */
inline vec3 random_in_unit_disk() {
    // Generate random point inside unit disk using rejection sampling
    while (true) {
        vec3 p = vec3(random_double(-1,1), random_double(-1,1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

#endif
