/*
 * shapes.h - Geometric Shape Implementations
 * 
 * Implements concrete geometric shapes that can be intersected by rays.
 * Currently includes sphere implementation using optimized quadratic formula.
 */

#ifndef SHAPES_H
#define SHAPES_H

#include "hittable.h"
#include "material.h"

/**
 * sphere - Sphere Geometry Implementation
 * 
 * Represents a sphere defined by center point and radius.
 * Uses optimized ray-sphere intersection based on quadratic formula.
 * Supports both solid spheres (positive radius) and hollow spheres (negative radius).
 */
class sphere : public hittable {
  public:
    point3 center;               // Sphere center point
    double radius;               // Sphere radius (can be negative for hollow)
    shared_ptr<material> mat;    // Surface material

    /**
     * Constructor - Create sphere with center, radius, and material
     * @param _center Center point of the sphere
     * @param _radius Radius (positive for solid, negative for hollow interior)
     * @param _material Surface material for shading calculations
     */
    sphere(const point3& _center, double _radius, shared_ptr<material> _material)
      : center(_center), radius(_radius), mat(_material) {}

    /**
     * hit - Ray-Sphere Intersection Test
     * 
     * Uses optimized quadratic formula for ray-sphere intersection:
     * Ray: P(t) = A + t*B
     * Sphere: (P-C)·(P-C) = r²
     * 
     * Solving gives quadratic: at² + bt + c = 0
     * where: a = B·B, b = 2B·(A-C), c = (A-C)·(A-C) - r²
     * 
     * Optimized form uses half_b = B·(A-C) to reduce computation.
     */
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        double a = r.direction().length_squared();
        double half_b = dot(oc, r.direction());
        double c = oc.length_squared() - radius*radius;

        // Discriminant determines if intersection exists
        double discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;

        // Find the nearest root that lies in the acceptable range.
        double sqrtd = sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        // Fill hit record with intersection information
        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }
};

// ============================================================================
// Future shapes can be added here:
// - class plane : public hittable { ... };
// - class triangle : public hittable { ... };
// - class box : public hittable { ... };
// ============================================================================

#endif
