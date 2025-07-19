/*
 * ray_tracer.cpp - Main Ray Tracer Program
 * 
 * Complete implementation of "Ray Tracing in One Weekend" featuring:
 * - Physically-based materials (diffuse, metal, dielectric)  
 * - Advanced camera with positioning and depth of field
 * - Antialiasing and recursive ray tracing
 * - Configurable scenes demonstrating various features
 * 
 * This scene showcases defocus blur with mixed materials:
 * - Hollow glass sphere (left) with total internal reflection
 * - Blue diffuse sphere (center) as focus target
 * - Gold metal sphere (right) with perfect reflection
 * - Positioned camera with narrow FOV and large aperture
 */

#include "rtweekend.h"
#include "camera.h"
#include "hittable.h"
#include "shapes.h"
#include "material.h"

/**
 * main - Ray Tracer Entry Point
 * 
 * Sets up scene geometry, materials, and camera configuration,
 * then renders the scene to PPM format on stdout.
 * 
 * Current scene: Defocus blur demonstration with hollow glass sphere
 */
int main() {
    // World
    hittable_list world;

    // SCENE 7: Defocus Blur (Depth of Field) Demo
    shared_ptr<lambertian> ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    shared_ptr<lambertian> center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    shared_ptr<dielectric> left   = make_shared<dielectric>(1.5);                    
    shared_ptr<dielectric> bubble = make_shared<dielectric>(1.5);                    
    shared_ptr<metal> right       = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);  

    // Objects at different distances for depth of field effect
    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, center));      // Focus target
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, left));        
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),  -0.4, bubble));      
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, right));

    // Camera with defocus blur
    camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;  // High quality for blur effects
    cam.max_depth         = 50;
    
    cam.vfov     = 20;                          
    cam.lookfrom = point3(-2, 2, 1);           
    cam.lookat   = point3(0, 0, -1);           
    cam.vup      = vec3(0, 1, 0);              

    cam.defocus_angle = 10.0;                   // Large aperture for dramatic blur
    cam.focus_dist    = 3.4;                    // Focus on the center sphere

    cam.render(world);
}
