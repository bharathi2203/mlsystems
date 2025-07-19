/*
 * material.h - Material System for Surface Scattering
 * 
 * Implements polymorphic material system for physically-based light scattering.
 * Includes Lambertian diffuse, metal reflection, and dielectric refraction materials.
 * Each material defines how light rays scatter when hitting surfaces.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"

// Forward declaration
struct hit_record;

/**
 * material - Abstract Material Base Class
 * 
 * Defines interface for all material types. Materials determine how light
 * scatters when rays hit surfaces, controlling the appearance and behavior
 * of objects in the scene.
 */
class material {
  public:
    virtual ~material() {}

    /**
     * scatter - Determine ray scattering behavior
     * 
     * @param r_in Incident ray hitting the surface
     * @param rec Hit record containing surface information
     * @param attenuation Output: color filter applied to scattered light
     * @param scattered Output: new ray direction after scattering
     * @return true if light scatters, false if absorbed
     */
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const = 0;
};

/**
 * lambertian - Diffuse Material (Matte Surfaces)
 * 
 * Implements ideal Lambertian reflection where light scatters equally
 * in all directions. Creates matte, non-reflective surfaces like paper,
 * concrete, or unpolished wood.
 */
class lambertian : public material {
  public:
    color albedo;  // Surface color (reflectance)

    lambertian(const color& a) : albedo(a) {}

    /**
     * scatter - Lambertian Diffuse Scattering
     * 
     * Scatters light randomly in hemisphere around surface normal.
     * Uses random unit vector + normal for true Lambertian distribution.
     */
    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const {
        vec3 scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

/**
 * metal - Reflective Material (Mirror Surfaces)
 * 
 * Implements specular reflection with optional surface roughness.
 * Perfect mirrors (fuzz=0) to brushed metals (fuzz=1).
 */
class metal : public material {
  public:
    color albedo;   // Surface color tint
    double fuzz;    // Surface roughness (0=perfect mirror, 1=completely rough)

    metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    /**
     * scatter - Metal Reflection
     * 
     * Uses reflection formula: r = v - 2(v·n)n
     * Adds random perturbation based on fuzz parameter for surface roughness.
     */
    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_unit_vector());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
};

/**
 * dielectric - Transparent Material (Glass, Water, etc.)
 * 
 * Implements refraction and reflection based on Snell's law and Fresnel equations.
 * Handles total internal reflection and uses Schlick's approximation for
 * realistic glass appearance.
 */
class dielectric : public material {
  public:
    double ir; // Index of Refraction (air=1.0, glass≈1.5, water≈1.33)

    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    /**
     * scatter - Dielectric Refraction and Reflection
     * 
     * Implements Snell's law: η₁sin(θ₁) = η₂sin(θ₂)
     * Uses Schlick's approximation for Fresnel reflectance
     * Handles total internal reflection when sin(θ₂) > 1
     */
    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::min(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

  private:
    /**
     * reflectance - Schlick's Approximation for Fresnel Reflectance
     * 
     * Approximates the fraction of light reflected vs refracted at interface.
     * Formula: R(θ) = R₀ + (1-R₀)(1-cos(θ))⁵
     * where R₀ = ((1-n)/(1+n))²
     */
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        double r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

#endif
