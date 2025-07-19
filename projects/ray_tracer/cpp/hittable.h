/*
 * hittable.h - Ray-Object Intersection Interface
 * 
 * Defines the abstract hittable interface for ray-object intersection testing
 * and the hit_record structure for storing intersection information.
 * Also includes hittable_list for managing collections of objects.
 */

#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"

// Forward declaration
class material;

/**
 * hit_record - Intersection Information
 * 
 * Stores all relevant information about a ray-object intersection:
 * - Hit point, surface normal, distance, and material
 * - Front/back face information for proper normal orientation
 */
struct hit_record {
    point3 p;                    // Hit point in world coordinates
    vec3 normal;                 // Surface normal at hit point
    shared_ptr<material> mat;    // Material at hit point
    double t;                    // Distance along ray to hit point
    bool front_face;             // True if ray hits outside surface

    /**
     * set_face_normal - Set normal direction based on ray direction
     * 
     * Ensures normal always points "outward" from surface, and sets
     * front_face flag to indicate which side of surface was hit.
     * 
     * @param r The incident ray
     * @param outward_normal The geometric outward normal (unit length)
     */
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to be unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

/**
 * hittable - Abstract Ray-Object Intersection Interface
 * 
 * Base class for all objects that can be intersected by rays.
 * Defines the interface for ray-object intersection testing.
 */
class hittable {
  public:
    virtual ~hittable() {}

    /**
     * hit - Test ray-object intersection
     * 
     * @param r The ray to test intersection with
     * @param ray_t Valid t-parameter range for intersection
     * @param rec Output parameter filled with hit information
     * @return true if intersection found in valid range, false otherwise
     */
    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

/**
 * hittable_list - Collection of Hittable Objects
 * 
 * Container class that manages multiple hittable objects and finds
 * the closest intersection among all objects in the scene.
 */
class hittable_list : public hittable {
  public:
    std::vector<shared_ptr<hittable> > objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    /**
     * hit - Find closest intersection among all objects
     * 
     * Tests intersection with all objects and returns information
     * about the closest hit (smallest positive t value).
     */
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = ray_t.max;

        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif
