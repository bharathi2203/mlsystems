/*
 * ray.h - Ray Class Implementation
 * 
 * Defines a ray as a function P(t) = A + t*B where:
 * A = ray origin (point3)
 * B = ray direction (vec3)  
 * t = parameter (double)
 * 
 * Forms the foundation for all ray-object intersection calculations.
 */

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

/**
 * ray - Ray Class
 * 
 * Represents a ray as a parametric line P(t) = origin + t * direction.
 * Used for primary camera rays, shadow rays, and reflected/refracted rays.
 * The parameter t represents distance along the ray from the origin.
 */
class ray {
  public:
    point3 orig;  // Ray origin point
    vec3 dir;     // Ray direction vector

    ray() {}
    
    /**
     * Constructor - Creates ray with given origin and direction
     * @param origin Starting point of the ray
     * @param direction Direction vector (doesn't need to be normalized)
     */
    ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction) {}

    // Accessor methods
    point3 origin() const  { return orig; }
    vec3 direction() const { return dir; }

    /**
     * at - Point Along Ray
     * Formula: P(t) = A + t*B
     * @param t Parameter value (distance along ray)
     * @return Point at parameter t along the ray
     */
    point3 at(double t) const {
        return orig + t*dir;
    }
};

#endif 